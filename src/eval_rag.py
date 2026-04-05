#!/usr/bin/env python3
"""
流水线量化评估：向量库中的「切分」统计 + 基于 golden 文件的 RAG 检索质量。

与历史报告 JSON 对比（--compare），用于对比不同 ingestion 参数或 RAG 配置下的产出。

用法（在 finance_agent 根目录）:
  python src/eval_rag.py chunks --db-path data/vector_db
  python src/eval_rag.py rag --golden data/eval/golden_queries.json
  python src/eval_rag.py all --golden data/eval/golden_queries.json --output data/eval/last_report.json
  python src/eval_rag.py all --compare data/eval/prev_report.json

Golden JSON 格式见 data/eval/golden_queries.example.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.model_config import EmbeddingConfig, RAGConfig
from src.rag_chain import create_rag_chain


def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def eval_chunks(db_path: str, embedding_model: str = "BAAI/bge-small-en-v1.5") -> Dict[str, Any]:
    """从 Chroma 读出所有 chunk，统计长度分布与来源覆盖（反映切分+入库结果）。"""
    p = Path(db_path)
    if not p.exists() or not any(p.iterdir()):
        raise FileNotFoundError(f"向量库不存在或为空: {db_path}")

    emb = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(persist_directory=str(p), embedding_function=emb)
    raw = store.get(include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    lengths = np.array([len(d or "") for d in docs], dtype=np.float64)
    sources = []
    for m in metas:
        if isinstance(m, dict):
            sources.append(str(m.get("source", "")))
        else:
            sources.append("")

    # 粗略 token 估计（英文财报场景常用 ~4 字符/token）
    token_est = lengths / 4.0

    unique_sources = len({s for s in sources if s})
    source_counts = Counter(sources)
    top_sources = source_counts.most_common(8)

    empty_ratio = float(np.mean(lengths == 0)) if lengths.size else 0.0

    return {
        "vector_db_path": str(p.resolve()),
        "chunk_count": int(lengths.size),
        "unique_source_paths": unique_sources,
        "char_length": {
            "mean": float(lengths.mean()) if lengths.size else 0.0,
            "std": float(lengths.std()) if lengths.size else 0.0,
            "min": float(lengths.min()) if lengths.size else 0.0,
            "max": float(lengths.max()) if lengths.size else 0.0,
            **_percentiles(lengths),
        },
        "token_length_est": {
            "mean": float(token_est.mean()) if token_est.size else 0.0,
            **_percentiles(token_est),
        },
        "empty_chunk_ratio": empty_ratio,
        "top_sources_by_chunk_count": [{"source": s, "chunks": c} for s, c in top_sources],
    }


def _normalize(s: str) -> str:
    return (s or "").lower()


def _keyword_recall(context: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    ctx = _normalize(context)
    hit = sum(1 for k in keywords if _normalize(k) in ctx)
    return hit / len(keywords)


def _source_should_match(sources: List[Dict[str, Any]], needles: List[str]) -> float:
    if not needles:
        return 1.0
    paths = [_normalize(s.get("source", "")) for s in sources]
    hit = 0
    for n in needles:
        nn = _normalize(n)
        if any(nn in p for p in paths):
            hit += 1
    return hit / len(needles)


def eval_rag_golden(
    db_path: str,
    golden_path: Path,
    with_rerank: bool = True,
    embedding_config: Optional[EmbeddingConfig] = None,
) -> Dict[str, Any]:
    """对 golden 中每条 query 跑 RAG，统计关键词召回与来源匹配（轻量、可复现）。"""
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if not items:
        raise ValueError("golden 文件缺少 items 数组")

    rag_cfg = RAGConfig(verbose=False)
    rag = create_rag_chain(
        vector_db_path=db_path,
        embedding_config=embedding_config or EmbeddingConfig(device="cpu"),
        rag_config=rag_cfg,
    )

    per_query: List[Dict[str, Any]] = []
    latencies: List[float] = []
    krs: List[float] = []
    srs: List[float] = []

    for it in items:
        q = it.get("query") or ""
        must = list(it.get("must_contain") or [])
        src_need = list(it.get("source_should_contain") or [])

        t0 = time.perf_counter()
        result = rag.get_context(q, with_rerank=with_rerank)
        latencies.append(time.perf_counter() - t0)

        ctx = result.get("context") or ""
        sources = result.get("sources") or []

        kr = _keyword_recall(ctx, must)
        sr = _source_should_match(sources, src_need)
        krs.append(kr)
        srs.append(sr)

        top_score = None
        if sources and sources[0].get("relevance_score") != "N/A":
            try:
                top_score = float(sources[0]["relevance_score"])
            except (TypeError, ValueError):
                top_score = None

        per_query.append(
            {
                "id": it.get("id"),
                "query": q,
                "keyword_recall": kr,
                "source_should_match_rate": sr,
                "num_documents": result.get("num_documents", 0),
                "top_rerank_score": top_score,
            }
        )

    return {
        "golden_path": str(golden_path.resolve()),
        "with_rerank": with_rerank,
        "query_count": len(items),
        "mean_keyword_recall": float(np.mean(krs)) if krs else 0.0,
        "mean_source_should_match": float(np.mean(srs)) if srs else 0.0,
        "mean_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
        "per_query": per_query,
    }


def _diff_scalar(name: str, old: Any, new: Any) -> str:
    if old is None or new is None:
        return f"  {name}: 旧={old} 新={new}"
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        delta = new - old
        return f"  {name}: {old} -> {new} (Δ {delta:+.4g})"
    return f"  {name}: {old} -> {new}"


def print_compare(prev: Dict[str, Any], cur: Dict[str, Any]) -> None:
    """对比两份报告中的核心标量。"""
    print("\n=== 与历史报告对比 ===")
    pc, cc = prev.get("chunks"), cur.get("chunks")
    if pc and cc:
        print("chunks:")
        print(_diff_scalar("chunk_count", pc.get("chunk_count"), cc.get("chunk_count")))
        print(
            _diff_scalar(
                "char_length.mean",
                pc.get("char_length", {}).get("mean"),
                cc.get("char_length", {}).get("mean"),
            )
        )
        print(
            _diff_scalar(
                "empty_chunk_ratio",
                pc.get("empty_chunk_ratio"),
                cc.get("empty_chunk_ratio"),
            )
        )

    pr, cr = prev.get("rag"), cur.get("rag")
    if pr and cr:
        print("rag:")
        print(
            _diff_scalar(
                "mean_keyword_recall",
                pr.get("mean_keyword_recall"),
                cr.get("mean_keyword_recall"),
            )
        )
        print(
            _diff_scalar(
                "mean_source_should_match",
                pr.get("mean_source_should_match"),
                cr.get("mean_source_should_match"),
            )
        )
        print(
            _diff_scalar(
                "mean_latency_sec",
                pr.get("mean_latency_sec"),
                cr.get("mean_latency_sec"),
            )
        )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Finance Agent — 切分与 RAG 量化评估")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db-path", default="data/vector_db", help="Chroma 持久化目录")
    common.add_argument(
        "--golden",
        default="data/eval/golden_queries.json",
        help="RAG 评估用 golden JSON（可复制 golden_queries.example.json）",
    )
    common.add_argument(
        "--output",
        "-o",
        default="data/eval/last_report.json",
        help="写入本次完整报告的路径",
    )
    common.add_argument(
        "--compare",
        type=str,
        default=None,
        help="上一份报告 JSON，用于打印关键指标差异",
    )
    common.add_argument(
        "--no-rerank",
        action="store_true",
        help="RAG 评估时不做 cross-encoder 重排",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chunks", parents=[common], help="仅统计向量库内 chunk 分布")
    sub.add_parser("rag", parents=[common], help="仅跑 golden RAG 指标")
    sub.add_parser("all", parents=[common], help="chunks + rag")

    args = parser.parse_args()

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunks": None,
        "rag": None,
    }

    if args.cmd in ("chunks", "all"):
        report["chunks"] = eval_chunks(args.db_path)

    if args.cmd in ("rag", "all"):
        gpath = Path(args.golden)
        if not gpath.is_file():
            print(
                f"未找到 golden 文件: {gpath}\n"
                f"请复制 data/eval/golden_queries.example.json 为 {args.golden} 并编辑。",
                file=sys.stderr,
            )
            return 1
        report["rag"] = eval_rag_golden(
            args.db_path,
            gpath,
            with_rerank=not args.no_rerank,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入: {out_path.resolve()}")

    if args.compare:
        prev_path = Path(args.compare)
        if prev_path.is_file():
            print_compare(json.loads(prev_path.read_text(encoding="utf-8")), report)
        else:
            print(f"对比文件不存在，已跳过: {prev_path}", file=sys.stderr)

    # 简要终端摘要
    if report["chunks"]:
        c = report["chunks"]
        print(f"\n[切分/向量库] chunks={c['chunk_count']} 均长={c['char_length']['mean']:.1f} 字符")
    if report["rag"]:
        r = report["rag"]
        print(
            f"[RAG] mean_keyword_recall={r['mean_keyword_recall']:.3f} "
            f"mean_source_match={r['mean_source_should_match']:.3f} "
            f"latency={r['mean_latency_sec']:.3f}s/query"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
