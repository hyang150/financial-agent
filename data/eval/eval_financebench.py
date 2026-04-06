#!/usr/bin/env python3
"""
FinanceBench 基准评估脚本。

从 HuggingFace 加载 FinanceBench（150 条专家标注的 SEC 10-K/10-Q QA），
转为本项目的 golden_queries 格式，并对 RAG 检索 + 端到端答案进行量化评估。

====================================================================
功能一览
====================================================================
  1. download   — 下载 FinanceBench 数据集并转换为本地 golden JSON
  2. retrieval  — 纯检索评估（不需要 LLM，本地 CPU 可跑）
  3. e2e        — 端到端评估（需要 LLM，需 GPU 或 API）
  4. report     — 汇总检索 + 端到端评估，输出对比报告
  5. show       — 查看已有报告的摘要

====================================================================
用法（在 finance_agent 根目录）
====================================================================
  # Step 1: 下载 FinanceBench 并生成 golden 文件
  python data/eval/eval_financebench.py download

  # Step 2: 仅检索评估（本地 CPU 可跑）
  python data/eval/eval_financebench.py retrieval --db-path data/vector_db

  # Step 2b: 只评估指定 ticker（过滤掉你没有数据的公司）
  python data/eval/eval_financebench.py retrieval --tickers AAPL MSFT TSLA

  # Step 3: 端到端评估（需 GPU 加载 LLM）
  python data/eval/eval_financebench.py e2e --db-path data/vector_db

  # Step 3b: 端到端只评估指定 ticker
  python data/eval/eval_financebench.py e2e --tickers AAPL MSFT TSLA

  # Step 4: 查看报告
  python data/eval/eval_financebench.py show --report data/eval/financebench_report.json

  # 对比两次运行
  python data/eval/eval_financebench.py report \
      --current data/eval/financebench_report.json \
      --previous data/eval/financebench_report_prev.json

====================================================================
依赖
====================================================================
  pip install datasets    # HuggingFace datasets（仅 download 子命令需要）
  其余依赖与主项目一致
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------- 项目路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ================================================================
#  常量
# ================================================================
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_GOLDEN = EVAL_DIR / "financebench_golden.json"
DEFAULT_REPORT = EVAL_DIR / "financebench_report.json"

# FinanceBench 中的公司 → 股票代码映射（覆盖数据集中出现的主要公司）
COMPANY_TICKER_MAP = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "AMAZON": "AMZN",
    "ALPHABET": "GOOGL", "GOOGLE": "GOOGL", "META": "META",
    "FACEBOOK": "META", "TESLA": "TSLA", "NVIDIA": "NVDA",
    "JOHNSON & JOHNSON": "JNJ", "JPMORGAN": "JPM", "JP MORGAN": "JPM",
    "WALMART": "WMT", "VISA": "V", "PROCTER & GAMBLE": "PG",
    "MASTERCARD": "MA", "UNITEDHEALTH": "UNH", "HOME DEPOT": "HD",
    "BANK OF AMERICA": "BAC", "PFIZER": "PFE", "ABBVIE": "ABBV",
    "COCA-COLA": "KO", "COCACOLA": "KO", "PEPSICO": "PEP",
    "COSTCO": "COST", "CISCO": "CSCO", "MERCK": "MRK",
    "THERMO FISHER": "TMO", "MCDONALD": "MCD", "MCDONALDS": "MCD",
    "WALT DISNEY": "DIS", "DISNEY": "DIS", "INTEL": "INTC",
    "VERIZON": "VZ", "COMCAST": "CMCSA", "ADOBE": "ADBE",
    "NETFLIX": "NFLX", "SALESFORCE": "CRM", "AMGEN": "AMGN",
    "AMD": "AMD", "BROADCOM": "AVGO", "STARBUCKS": "SBUX",
    "GENERAL ELECTRIC": "GE", "3M": "MMM", "BOEING": "BA",
    "CATERPILLAR": "CAT", "HONEYWELL": "HON", "GOLDMAN SACHS": "GS",
    "MORGAN STANLEY": "MS", "CITIGROUP": "C", "WELLS FARGO": "WFC",
    "AMERICAN EXPRESS": "AXP", "LOCKHEED MARTIN": "LMT",
    "RAYTHEON": "RTX", "GENERAL DYNAMICS": "GD",
    "CORNING": "GLW", "ACTIVISION": "ATVI", "BLOCK": "SQ",
    "PAYPAL": "PYPL", "QUALCOMM": "QCOM", "TEXAS INSTRUMENTS": "TXN",
    "GENERAL MILLS": "GIS", "KELLOGG": "K", "KRAFT HEINZ": "KHC",
    "TARGET": "TGT", "BEST BUY": "BBY", "CVS": "CVS",
    "WALGREENS": "WBA", "LOWE'S": "LOW", "LOWES": "LOW",
    "NIKE": "NKE", "ORACLE": "ORCL", "IBM": "IBM",
    "SERVICENOW": "NOW", "WORKDAY": "WDAY", "SNOWFLAKE": "SNOW",
    "UBER": "UBER", "AIRBNB": "ABNB", "DOORDASH": "DASH",
    "RIVIAN": "RIVN", "LUCID": "LCID",
}


# ================================================================
#  子命令 1: download — 下载 FinanceBench 并转为 golden JSON
# ================================================================

def _guess_ticker(question: str, answer: str, doc_name: str) -> Optional[str]:
    """从问题/答案/文档名中猜测股票代码。"""
    combined = f"{question} {answer} {doc_name}".upper()
    for company, ticker in sorted(COMPANY_TICKER_MAP.items(), key=lambda x: -len(x[0])):
        if company in combined:
            return ticker
    # 尝试直接匹配 ticker 格式 (全大写 2-5 字母)
    for token in combined.split():
        clean = re.sub(r"[^A-Z]", "", token)
        if 2 <= len(clean) <= 5 and clean in COMPANY_TICKER_MAP.values():
            return clean
    return None


def _extract_keywords(question: str, answer: str) -> List[str]:
    """从问题和答案中提取关键词，用于检索召回评估。"""
    keywords = set()

    # 从答案中提取数字（财报 QA 的核心）
    numbers = re.findall(r"\$[\d,]+\.?\d*|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+%?", answer)
    for n in numbers[:3]:  # 最多取 3 个数字
        keywords.add(n.replace(",", ""))

    # 从问题中提取实体关键词（去掉停用词）
    stopwords = {
        "what", "was", "is", "the", "a", "an", "of", "in", "for", "to",
        "and", "or", "how", "much", "did", "does", "do", "were", "are",
        "its", "it", "their", "they", "this", "that", "which", "who",
        "from", "by", "with", "as", "at", "on", "has", "had", "have",
        "been", "be", "will", "would", "could", "should", "can", "may",
        "about", "than", "more", "most", "any", "some", "all", "each",
        "between", "through", "during", "before", "after", "above",
        "below", "up", "down", "out", "into", "over", "under",
        "fiscal", "year", "quarter", "annual", "report", "according",
        "total", "net", "gross", "per", "share", "based", "company",
    }
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", question)
    for t in tokens:
        if t.lower() not in stopwords and len(t) > 2:
            keywords.add(t)

    # 从答案提取关键短语
    answer_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", answer)
    for t in answer_tokens:
        if t.lower() not in stopwords and len(t) > 3:
            keywords.add(t)

    return list(keywords)[:8]  # 限制数量，避免过于严格


def cmd_download(args) -> int:
    """从 HuggingFace 下载 FinanceBench 数据集并转换为本项目的 golden JSON。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("  pip install datasets")
        return 1

    print("正在从 HuggingFace 下载 FinanceBench 数据集...")
    try:
        ds = load_dataset("PatronusAI/financebench", split="train")
    except Exception as e:
        print(f"下载失败: {e}")
        print("备用方案: 访问 https://huggingface.co/datasets/PatronusAI/financebench 手动下载")
        return 1

    print(f"成功加载 {len(ds)} 条数据")

    # ---------- 查看数据集的字段结构 ----------
    print(f"\n数据集字段: {ds.column_names}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"示例条目字段:")
        for k, v in sample.items():
            preview = str(v)[:120] + ("..." if len(str(v)) > 120 else "")
            print(f"  {k}: {preview}")

    # ---------- 转换为 golden_queries 格式 ----------
    items = []
    ticker_counter = Counter()
    skipped = 0

    for idx, row in enumerate(ds):
        # FinanceBench 字段名可能变化，尝试多种命名
        question = (
            row.get("question")
            or row.get("query")
            or row.get("Question")
            or ""
        )
        answer = (
            row.get("answer")
            or row.get("gold_answer")
            or row.get("Answer")
            or ""
        )
        evidence = (
            row.get("evidence")
            or row.get("evidence_text")
            or row.get("Evidence")
            or ""
        )
        doc_name = (
            row.get("doc_name")
            or row.get("document")
            or row.get("source")
            or row.get("doc_link")
            or ""
        )
        question_type = row.get("question_type") or row.get("type") or ""
        doc_period = row.get("doc_period") or row.get("period") or ""

        if not question:
            skipped += 1
            continue

        ticker = _guess_ticker(question, str(answer), str(doc_name))
        if ticker:
            ticker_counter[ticker] += 1

        must_contain = _extract_keywords(question, str(answer))
        source_should = [ticker] if ticker else []

        item = {
            "id": f"fb_{idx:03d}",
            "query": question,
            "must_contain": must_contain,
            "source_should_contain": source_should,
            # 扩展字段（用于端到端评估）
            "reference_answer": str(answer),
            "evidence_text": str(evidence),
            "doc_name": str(doc_name),
            "question_type": str(question_type),
            "doc_period": str(doc_period),
        }
        items.append(item)

    golden = {
        "version": "2",
        "source": "FinanceBench (PatronusAI/financebench @ HuggingFace)",
        "description": (
            "由 FinanceBench 自动转换。150 条专家标注的 SEC 10-K/10-Q 问答。"
            "must_contain 从问题+答案自动提取，reference_answer 为黄金答案。"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_items": len(items),
            "skipped": skipped,
            "tickers_covered": dict(ticker_counter.most_common()),
        },
        "items": items,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已生成 golden 文件: {out_path}")
    print(f"  总条目: {len(items)}")
    print(f"  跳过:   {skipped}")
    print(f"  涉及 ticker: {dict(ticker_counter.most_common(10))}")
    print(f"\n下一步:")
    print(f"  1. 确保向量库已含对应 ticker 的财报数据")
    print(f"  2. 运行: python data/eval/eval_financebench.py retrieval")
    return 0


# ================================================================
#  子命令 2: retrieval — 纯检索评估（不需要 LLM）
# ================================================================

def _normalize(s: str) -> str:
    return (s or "").lower().strip()


def _keyword_recall(context: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    ctx = _normalize(context)
    hit = sum(1 for k in keywords if _normalize(k) in ctx)
    return hit / len(keywords)


def _source_match(sources: List[Dict[str, Any]], needles: List[str]) -> float:
    if not needles:
        return 1.0
    paths = [_normalize(s.get("source", "")) for s in sources]
    hit = sum(1 for n in needles if any(_normalize(n) in p for p in paths))
    return hit / len(needles)


def _evidence_overlap(context: str, evidence: str) -> float:
    """计算证据文本与检索上下文的词级 F1 重叠度。"""
    if not evidence or not context:
        return 0.0
    ctx_tokens = set(_normalize(context).split())
    evi_tokens = set(_normalize(evidence).split())
    if not evi_tokens:
        return 0.0
    overlap = ctx_tokens & evi_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(ctx_tokens) if ctx_tokens else 0.0
    recall = len(overlap) / len(evi_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cmd_retrieval(args) -> int:
    """对 FinanceBench golden 文件执行纯检索评估。"""
    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(f"错误: golden 文件不存在: {golden_path}")
        print(f"请先运行: python data/eval/eval_financebench.py download")
        return 1

    data = json.loads(golden_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if not items:
        print("错误: golden 文件中无 items")
        return 1

    # ---------- ticker 过滤 ----------
    if args.tickers:
        tickers_upper = {t.upper() for t in args.tickers}
        original_count = len(items)
        items = [
            it for it in items
            if any(t.upper() in tickers_upper
                   for t in it.get("source_should_contain", []))
        ]
        print(f"Ticker 过滤: {tickers_upper}")
        print(f"  原始 {original_count} 条 → 筛选后 {len(items)} 条")
        if not items:
            print("错误: 过滤后无匹配条目。请检查 ticker 是否正确。")
            return 1

    print(f"加载 {len(items)} 条 FinanceBench 查询")
    print(f"向量库: {args.db_path}")
    print(f"Rerank: {'关闭' if args.no_rerank else '开启'}")
    print()

    # 加载 RAG 链
    from config.model_config import EmbeddingConfig, RAGConfig
    from src.rag_chain import create_rag_chain

    rag_cfg = RAGConfig(verbose=False)
    rag = create_rag_chain(
        vector_db_path=args.db_path,
        embedding_config=EmbeddingConfig(device="cpu"),
        rag_config=rag_cfg,
    )

    # ---------- 逐条评估 ----------
    results: List[Dict[str, Any]] = []
    keyword_recalls: List[float] = []
    source_matches: List[float] = []
    evidence_overlaps: List[float] = []
    latencies: List[float] = []
    rerank_scores: List[float] = []

    # 按 question_type 分组统计
    type_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    total = len(items)
    for i, item in enumerate(items):
        query = item.get("query", "")
        must = item.get("must_contain", [])
        src_need = item.get("source_should_contain", [])
        evidence = item.get("evidence_text", "")
        q_type = item.get("question_type", "unknown")

        print(f"\r  [{i+1}/{total}] 评估中...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            ctx_result = rag.get_context(query, with_rerank=not args.no_rerank)
        except Exception as e:
            print(f"\n  警告: 查询失败 [{item.get('id')}]: {e}")
            results.append({
                "id": item.get("id"),
                "query": query,
                "error": str(e),
            })
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        ctx = ctx_result.get("context", "")
        sources = ctx_result.get("sources", [])

        kr = _keyword_recall(ctx, must)
        sm = _source_match(sources, src_need)
        eo = _evidence_overlap(ctx, evidence)

        keyword_recalls.append(kr)
        source_matches.append(sm)
        evidence_overlaps.append(eo)

        top_score = None
        if sources and sources[0].get("relevance_score") != "N/A":
            try:
                top_score = float(sources[0]["relevance_score"])
                rerank_scores.append(top_score)
            except (TypeError, ValueError):
                pass

        row = {
            "id": item.get("id"),
            "query": query,
            "question_type": q_type,
            "keyword_recall": round(kr, 4),
            "source_match": round(sm, 4),
            "evidence_overlap_f1": round(eo, 4),
            "top_rerank_score": round(top_score, 4) if top_score is not None else None,
            "num_docs_retrieved": ctx_result.get("num_documents", 0),
            "latency_sec": round(elapsed, 3),
        }
        results.append(row)
        type_stats[q_type].append({"kr": kr, "sm": sm, "eo": eo})

    print(f"\r  [{total}/{total}] 完成!                    ")

    # ---------- 汇总指标 ----------
    summary = {
        "mean_keyword_recall": round(float(np.mean(keyword_recalls)), 4) if keyword_recalls else 0.0,
        "mean_source_match": round(float(np.mean(source_matches)), 4) if source_matches else 0.0,
        "mean_evidence_overlap_f1": round(float(np.mean(evidence_overlaps)), 4) if evidence_overlaps else 0.0,
        "mean_rerank_score": round(float(np.mean(rerank_scores)), 4) if rerank_scores else None,
        "mean_latency_sec": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "p50_latency_sec": round(float(np.percentile(latencies, 50)), 3) if latencies else 0.0,
        "p95_latency_sec": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        "total_queries": total,
        "successful_queries": len(keyword_recalls),
    }

    # 按题型汇总
    per_type_summary = {}
    for qtype, stats_list in sorted(type_stats.items()):
        per_type_summary[qtype] = {
            "count": len(stats_list),
            "mean_keyword_recall": round(float(np.mean([s["kr"] for s in stats_list])), 4),
            "mean_source_match": round(float(np.mean([s["sm"] for s in stats_list])), 4),
            "mean_evidence_overlap_f1": round(float(np.mean([s["eo"] for s in stats_list])), 4),
        }

    # ---------- 输出报告 ----------
    report = {
        "eval_type": "retrieval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "db_path": args.db_path,
            "golden_path": str(golden_path),
            "with_rerank": not args.no_rerank,
        },
        "summary": summary,
        "per_question_type": per_type_summary,
        "per_query": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 终端打印 ----------
    print("\n" + "=" * 65)
    print("  FinanceBench 检索评估结果")
    print("=" * 65)
    print(f"  查询总数:              {summary['total_queries']}")
    print(f"  成功查询:              {summary['successful_queries']}")
    print(f"  关键词召回率 (mean):   {summary['mean_keyword_recall']:.4f}")
    print(f"  来源命中率 (mean):     {summary['mean_source_match']:.4f}")
    print(f"  证据重叠 F1 (mean):    {summary['mean_evidence_overlap_f1']:.4f}")
    if summary.get("mean_rerank_score") is not None:
        print(f"  重排分数 (mean):       {summary['mean_rerank_score']:.4f}")
    print(f"  平均延迟:              {summary['mean_latency_sec']:.3f}s")
    print(f"  P95 延迟:              {summary['p95_latency_sec']:.3f}s")

    if per_type_summary:
        print(f"\n{'题型':<25} {'数量':>5} {'召回率':>8} {'来源':>8} {'证据F1':>8}")
        print("-" * 58)
        for qtype, s in per_type_summary.items():
            label = qtype[:24] if qtype else "unknown"
            print(f"  {label:<23} {s['count']:>5} {s['mean_keyword_recall']:>8.4f} "
                  f"{s['mean_source_match']:>8.4f} {s['mean_evidence_overlap_f1']:>8.4f}")

    print(f"\n报告已写入: {out_path}")
    return 0


# ================================================================
#  子命令 3: e2e — 端到端评估（需要 LLM）
# ================================================================

def _simple_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = set(_normalize(prediction).split())
    ref_tokens = set(_normalize(reference).split())
    if not ref_tokens or not pred_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    """宽松精确匹配：答案中是否包含参考答案的核心内容。"""
    pred = _normalize(prediction)
    ref = _normalize(reference)
    # 完全匹配
    if ref in pred:
        return 1.0
    # 尝试提取参考答案中的数字进行匹配
    ref_numbers = re.findall(r"[\d,]+\.?\d*", ref)
    if ref_numbers:
        pred_numbers = re.findall(r"[\d,]+\.?\d*", pred)
        for rn in ref_numbers:
            rn_clean = rn.replace(",", "")
            if any(pn.replace(",", "") == rn_clean for pn in pred_numbers):
                return 1.0
    return 0.0


def cmd_e2e(args) -> int:
    """端到端评估：加载 LLM → 对每条 FinanceBench 查询生成答案 → 与黄金答案对比。"""
    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(f"错误: golden 文件不存在: {golden_path}")
        return 1

    data = json.loads(golden_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if not items:
        print("错误: golden 文件中无 items")
        return 1

    # ---------- ticker 过滤 ----------
    if args.tickers:
        tickers_upper = {t.upper() for t in args.tickers}
        original_count = len(items)
        items = [
            it for it in items
            if any(t.upper() in tickers_upper
                   for t in it.get("source_should_contain", []))
        ]
        print(f"Ticker 过滤: {tickers_upper}")
        print(f"  原始 {original_count} 条 → 筛选后 {len(items)} 条")
        if not items:
            print("错误: 过滤后无匹配条目。请检查 ticker 是否正确。")
            return 1

    # 可选限制评估条数（端到端较慢）
    if args.limit and args.limit < len(items):
        items = items[:args.limit]
        print(f"限制评估条数: {args.limit}")

    print(f"加载 {len(items)} 条查询进行端到端评估")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")

    # 尝试加载 Agent
    try:
        from config.model_config import ModelConfig, AgentConfig
        from src.agent import create_agent

        model_config = ModelConfig(
            model_name=args.model,
            device=args.device,
            temperature=0.1,
        )
        agent_config = AgentConfig(verbose=False)
        agent = create_agent(
            model_config=model_config,
            agent_config=agent_config,
            vector_db_path=args.db_path,
        )
    except Exception as e:
        print(f"错误: 无法加载 Agent: {e}")
        print("端到端评估需要 GPU 和完整模型。如果没有 GPU，请使用 retrieval 子命令。")
        return 1

    results: List[Dict[str, Any]] = []
    f1_scores: List[float] = []
    em_scores: List[float] = []
    latencies: List[float] = []
    type_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    total = len(items)
    for i, item in enumerate(items):
        query = item.get("query", "")
        ref_answer = item.get("reference_answer", "")
        q_type = item.get("question_type", "unknown")

        print(f"\r  [{i+1}/{total}] 生成答案中...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            agent_result = agent.query(query)
            pred_answer = agent_result.get("answer", "")
        except Exception as e:
            print(f"\n  警告: 查询失败 [{item.get('id')}]: {e}")
            results.append({"id": item.get("id"), "query": query, "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        f1 = _simple_f1(pred_answer, ref_answer)
        em = _exact_match(pred_answer, ref_answer)
        f1_scores.append(f1)
        em_scores.append(em)

        row = {
            "id": item.get("id"),
            "query": query,
            "question_type": q_type,
            "predicted_answer": pred_answer[:500],
            "reference_answer": ref_answer[:500],
            "token_f1": round(f1, 4),
            "exact_match": round(em, 4),
            "latency_sec": round(elapsed, 3),
        }
        results.append(row)
        type_stats[q_type].append({"f1": f1, "em": em})

    print(f"\r  [{total}/{total}] 完成!                    ")

    summary = {
        "mean_token_f1": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "mean_exact_match": round(float(np.mean(em_scores)), 4) if em_scores else 0.0,
        "mean_latency_sec": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "total_queries": total,
        "successful_queries": len(f1_scores),
    }

    per_type_summary = {}
    for qtype, stats_list in sorted(type_stats.items()):
        per_type_summary[qtype] = {
            "count": len(stats_list),
            "mean_token_f1": round(float(np.mean([s["f1"] for s in stats_list])), 4),
            "mean_exact_match": round(float(np.mean([s["em"] for s in stats_list])), 4),
        }

    report = {
        "eval_type": "end_to_end",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "device": args.device,
            "db_path": args.db_path,
            "golden_path": str(golden_path),
        },
        "summary": summary,
        "per_question_type": per_type_summary,
        "per_query": results,
    }

    out_path = Path(args.output.replace(".json", "_e2e.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("  FinanceBench 端到端评估结果")
    print("=" * 65)
    print(f"  查询总数:             {summary['total_queries']}")
    print(f"  成功:                 {summary['successful_queries']}")
    print(f"  Token F1 (mean):      {summary['mean_token_f1']:.4f}")
    print(f"  Exact Match (mean):   {summary['mean_exact_match']:.4f}")
    print(f"  平均延迟:             {summary['mean_latency_sec']:.3f}s")

    if per_type_summary:
        print(f"\n{'题型':<25} {'数量':>5} {'F1':>8} {'EM':>8}")
        print("-" * 50)
        for qtype, s in per_type_summary.items():
            label = qtype[:24] if qtype else "unknown"
            print(f"  {label:<23} {s['count']:>5} {s['mean_token_f1']:>8.4f} "
                  f"{s['mean_exact_match']:>8.4f}")

    print(f"\n报告已写入: {out_path}")
    return 0


# ================================================================
#  子命令 4: report — 对比两份报告
# ================================================================

def _diff(name: str, old: Any, new: Any) -> str:
    if old is None or new is None:
        return f"  {name}: 旧={old} 新={new}"
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        delta = new - old
        arrow = "^" if delta > 0 else ("v" if delta < 0 else "=")
        return f"  {name}: {old} -> {new} ({arrow} {delta:+.4f})"
    return f"  {name}: {old} -> {new}"


def cmd_report(args) -> int:
    """对比两份评估报告（current vs previous）。"""
    cur_path = Path(args.current)
    prev_path = Path(args.previous) if args.previous else None

    if not cur_path.is_file():
        print(f"错误: 报告不存在: {cur_path}")
        return 1

    cur = json.loads(cur_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 65)
    print(f"  当前报告: {cur_path.name}")
    print(f"  类型: {cur.get('eval_type', 'unknown')}")
    print(f"  生成时间: {cur.get('generated_at', 'N/A')}")
    print("=" * 65)

    s = cur.get("summary", {})
    for k, v in s.items():
        print(f"  {k}: {v}")

    if prev_path and prev_path.is_file():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        ps = prev.get("summary", {})
        print(f"\n--- 与 {prev_path.name} 对比 ---")
        all_keys = set(list(s.keys()) + list(ps.keys()))
        for k in sorted(all_keys):
            if k in s and k in ps:
                print(_diff(k, ps[k], s[k]))
    elif args.previous:
        print(f"\n警告: 上一份报告不存在: {args.previous}")

    return 0


# ================================================================
#  子命令 5: show — 查看报告摘要
# ================================================================

def cmd_show(args) -> int:
    """查看已有评估报告的摘要。"""
    report_path = Path(args.report)
    if not report_path.is_file():
        # 尝试列出 eval 目录下所有报告
        eval_dir = EVAL_DIR
        reports = sorted(eval_dir.glob("financebench_report*.json"))
        if reports:
            print(f"可用的报告文件:")
            for r in reports:
                size = r.stat().st_size
                print(f"  {r.name} ({size:,} bytes)")
            print(f"\n使用: python data/eval/eval_financebench.py show --report <文件路径>")
        else:
            print("未找到任何报告。请先运行 retrieval 或 e2e 评估。")
        return 1

    data = json.loads(report_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 65)
    print(f"  报告: {report_path.name}")
    print(f"  类型: {data.get('eval_type', 'unknown')}")
    print(f"  时间: {data.get('generated_at', 'N/A')}")
    print("=" * 65)

    s = data.get("summary", {})
    for k, v in s.items():
        print(f"  {k}: {v}")

    ptype = data.get("per_question_type", {})
    if ptype:
        print(f"\n  按题型分:")
        for qtype, stats in ptype.items():
            print(f"    {qtype}: {stats}")

    # 找出表现最差的 5 条
    per_query = data.get("per_query", [])
    if per_query:
        # 按 keyword_recall 或 token_f1 排序
        sort_key = "keyword_recall" if "keyword_recall" in per_query[0] else "token_f1"
        valid = [r for r in per_query if sort_key in r and "error" not in r]
        if valid:
            worst = sorted(valid, key=lambda x: x.get(sort_key, 0))[:5]
            print(f"\n  表现最差的 5 条 (按 {sort_key}):")
            for w in worst:
                print(f"    [{w.get('id')}] {sort_key}={w.get(sort_key, 'N/A'):.4f} "
                      f"— {w.get('query', '')[:60]}...")

    return 0


# ================================================================
#  CLI 主入口
# ================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinanceBench 基准评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1. 下载数据集
  python data/eval/eval_financebench.py download

  # 2. 检索评估（本地 CPU）
  python data/eval/eval_financebench.py retrieval --db-path data/vector_db

  # 3. 端到端评估（需 GPU）
  python data/eval/eval_financebench.py e2e --model Qwen/Qwen2.5-7B-Instruct

  # 4. 对比报告
  python data/eval/eval_financebench.py report \\
      --current data/eval/financebench_report.json \\
      --previous data/eval/financebench_report_prev.json

  # 5. 查看报告
  python data/eval/eval_financebench.py show
        """,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- download ---
    p_dl = sub.add_parser("download", help="下载 FinanceBench 并生成 golden JSON")
    p_dl.add_argument(
        "--output", "-o",
        default=str(DEFAULT_GOLDEN),
        help=f"输出 golden JSON 路径 (默认: {DEFAULT_GOLDEN.name})",
    )

    # --- retrieval ---
    p_ret = sub.add_parser("retrieval", help="纯检索评估（不需要 LLM）")
    p_ret.add_argument("--db-path", default="data/vector_db", help="Chroma 向量库路径")
    p_ret.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="FinanceBench golden JSON")
    p_ret.add_argument("--output", "-o", default=str(DEFAULT_REPORT), help="输出报告路径")
    p_ret.add_argument("--no-rerank", action="store_true", help="关闭 Cross-Encoder 重排")
    p_ret.add_argument(
        "--tickers", "-t", nargs="+", default=None,
        help="只评估指定 ticker 的问题，例如: --tickers AAPL MSFT TSLA",
    )

    # --- e2e ---
    p_e2e = sub.add_parser("e2e", help="端到端评估（需要 LLM + GPU）")
    p_e2e.add_argument("--db-path", default="data/vector_db", help="Chroma 向量库路径")
    p_e2e.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="FinanceBench golden JSON")
    p_e2e.add_argument("--output", "-o", default=str(DEFAULT_REPORT), help="输出报告路径")
    p_e2e.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct", help="LLM 模型")
    p_e2e.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="设备")
    p_e2e.add_argument("--limit", type=int, default=None, help="限制评估条数（端到端较慢）")
    p_e2e.add_argument(
        "--tickers", "-t", nargs="+", default=None,
        help="只评估指定 ticker 的问题，例如: --tickers AAPL MSFT TSLA",
    )

    # --- report ---
    p_rpt = sub.add_parser("report", help="对比两份评估报告")
    p_rpt.add_argument("--current", required=True, help="当前报告 JSON")
    p_rpt.add_argument("--previous", default=None, help="上一份报告 JSON")

    # --- show ---
    p_show = sub.add_parser("show", help="查看报告摘要")
    p_show.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="报告 JSON 路径",
    )

    args = parser.parse_args()

    if args.cmd == "download":
        return cmd_download(args)
    elif args.cmd == "retrieval":
        return cmd_retrieval(args)
    elif args.cmd == "e2e":
        return cmd_e2e(args)
    elif args.cmd == "report":
        return cmd_report(args)
    elif args.cmd == "show":
        return cmd_show(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
