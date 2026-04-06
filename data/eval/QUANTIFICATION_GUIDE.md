# Finance Agent 产出量化指南

本文件说明如何量化本项目的两大核心产出：**切分（Chunking）质量** 与 **RAG 检索质量**，并与历史报告对比分析。

---

## 1. 快速使用：运行评估

### 前置：已有向量库 + golden 文件

```bash
# 仅统计 chunk 分布
python src/eval_rag.py chunks

# 仅跑 RAG golden 指标
python src/eval_rag.py rag --golden data/eval/golden_queries.json

# 两者全跑，写入报告
python src/eval_rag.py all \
    --golden data/eval/golden_queries.json \
    --output data/eval/last_report.json

# 与上一次报告对比（看指标变化）
python src/eval_rag.py all \
    --golden data/eval/golden_queries.json \
    --output data/eval/this_run.json \
    --compare data/eval/last_report.json
```

---

## 2. 切分（Chunking）量化指标

`eval_rag.py chunks` 统计向量库内所有 chunk 的以下指标：

| 指标 | 字段名 | 说明 | 参考目标 |
|------|--------|------|----------|
| **chunk 总数** | `chunk_count` | 向量库中的文本块数量 | 越多代表覆盖越全，但不是越多越好 |
| **去重来源文件数** | `unique_source_paths` | 来自几个不同原始文件 | 应与下载文件数一致 |
| **平均字符长度** | `char_length.mean` | 每个 chunk 平均字符数 | 建议 600–1200（当前配置 1000） |
| **标准差** | `char_length.std` | 长度分布的一致性 | 越小说明切分越均匀 |
| **最短/最长** | `char_length.min/max` | 极值检测 | min < 50 说明有碎片 chunk |
| **p50/p90/p99** | `char_length.p*` | 长度分位数 | p99 >> 1000 说明有超长块 |
| **token 估计均值** | `token_length_est.mean` | 约 `char/4`（英文财报） | LLM context window 规划参考 |
| **空 chunk 比例** | `empty_chunk_ratio` | 空白/无效块占比 | 应 < 0.01（1%） |
| **各来源 chunk 数** | `top_sources_by_chunk_count` | Top 8 来源的 chunk 分布 | 严重不均衡说明某文件过大 |

### 示例报告片段

```json
{
  "chunks": {
    "chunk_count": 3842,
    "unique_source_paths": 45,
    "char_length": {
      "mean": 952.3,
      "std": 187.6,
      "min": 12.0,
      "max": 2341.0,
      "p50": 998.0,
      "p90": 1150.0,
      "p99": 1843.0
    },
    "token_length_est": {
      "mean": 238.1,
      "p50": 249.5,
      "p90": 287.5,
      "p99": 460.8
    },
    "empty_chunk_ratio": 0.003,
    "top_sources_by_chunk_count": [
      {"source": "data/sec_filings/.../AAPL/10-K/.../full-submission.htm", "chunks": 621}
    ]
  }
}
```

### 如何解读 chunk 质量

```
chunk_count 过少（< 500）  → 文件下载不完整，或 HTML 解析失败
empty_chunk_ratio > 0.05   → HTML 中有大量无效节点，考虑清洗逻辑
char_length.std 很大       → 切分参数不稳定，检查 separators 配置
unique_source_paths << 预期 → 部分 ticker 下载失败
```

---

## 3. RAG 检索量化指标

`eval_rag.py rag` 对 `golden_queries.json` 中的每条查询执行检索，统计：

| 指标 | 字段名 | 说明 | 参考目标 |
|------|--------|------|----------|
| **关键词召回率** | `mean_keyword_recall` | `must_contain` 词在上下文中出现的比例 | > 0.85 |
| **来源命中率** | `mean_source_should_match` | 检索结果来源路径包含指定 ticker 的比例 | > 0.80 |
| **平均检索延迟** | `mean_latency_sec` | 每次 `get_context()` 的耗时（含重排） | CPU < 3s，GPU < 1s |
| **最高重排分数** | `top_rerank_score` | Cross-Encoder 给 rank=1 结果的分数 | > 0.5 说明语义高度匹配 |

### golden_queries.json 格式

```json
{
  "version": "1",
  "items": [
    {
      "id": "apple_revenue_2023",
      "query": "What was Apple's total revenue in fiscal year 2023?",
      "must_contain": ["revenue", "Apple", "2023"],
      "source_should_contain": ["AAPL"]
    },
    {
      "id": "msft_cloud_growth",
      "query": "Microsoft Azure cloud revenue growth rate",
      "must_contain": ["Azure", "cloud", "revenue"],
      "source_should_contain": ["MSFT"]
    }
  ]
}
```

> 复制 `golden_queries.example.json` → `golden_queries.json`，然后按实际下载的 ticker 扩充。

---

## 4. 与历史报告对比（`--compare`）

对比输出示例：

```
=== 与历史报告对比 ===
chunks:
  chunk_count: 3421 -> 3842 (Δ +421)
  char_length.mean: 945.2 -> 952.3 (Δ +7.1)
  empty_chunk_ratio: 0.012 -> 0.003 (Δ -0.009)
rag:
  mean_keyword_recall: 0.733 -> 0.867 (Δ +0.134)
  mean_source_should_match: 0.800 -> 0.933 (Δ +0.133)
  mean_latency_sec: 2.341 -> 1.892 (Δ -0.449)
```

**典型优化场景**：

| 操作 | 预期变化 |
|------|----------|
| 增加 ticker 或下载更多期数 | `chunk_count` ↑，`unique_source_paths` ↑ |
| 调大 `chunk_size`（如 1500） | `char_length.mean` ↑，`chunk_count` ↓ |
| 调小 `chunk_overlap`（如 50） | `chunk_count` ↓，可能 `keyword_recall` ↓ |
| 开启 rerank（默认开启） | `keyword_recall` ↑，`latency` ↑ |
| 改用更强嵌入模型 | `keyword_recall` ↑，`top_rerank_score` ↑ |

---

## 5. 可量化的综合维度总览

```
┌─────────────────────────────────────────────┐
│          Finance Agent 产出量化体系          │
├──────────────┬──────────────────────────────┤
│  A. 切分质量  │ chunk_count                  │
│              │ char_length 分布 (均/std/分位) │
│              │ token 估计                    │
│              │ empty_chunk_ratio             │
│              │ source 覆盖均匀度             │
├──────────────┼──────────────────────────────┤
│  B. 检索质量  │ keyword_recall (召回)         │
│              │ source_should_match (来源命中) │
│              │ top_rerank_score (语义相关性)  │
│              │ latency (检索速度)             │
├──────────────┼──────────────────────────────┤
│  C. 端到端    │ 需 LLM 生成答案对比           │
│  (需扩展)    │ - ROUGE / BLEU vs 参考答案    │
│              │ - Faithfulness（答案是否来自  │
│              │   检索文档，非 LLM 幻觉）     │
│              │ - Answer Relevance            │
└──────────────┴──────────────────────────────┘
```

### A+B 已由 `eval_rag.py` 实现（无需 LLM，轻量可复现）
### C 端到端需要扩展（见下方说明）

---

## 6. 端到端量化（扩展方向）

如需评估最终答案质量，可在 `data/eval/golden_queries.json` 增加 `reference_answer` 字段，配合以下方案：

```python
# 伪代码：端到端答案质量
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
result = agent.query(query)
scores = scorer.score(reference_answer, result['answer'])
# scores['rougeL'].fmeasure → 答案与参考答案的相似度
```

| 端到端指标 | 实现方式 | 成本 |
|-----------|---------|------|
| ROUGE-L | `rouge-score` 库，离线 | 低 |
| Faithfulness | 用 LLM 判断答案是否来自 context | 中（需 API） |
| Answer Relevance | 用 LLM 打分 1-5 | 中（需 API） |
| 人工评分 | 人工标注 | 高，最准确 |

---

## 7. 与现有文件对比速查

| 文件 | 用途 |
|------|------|
| `src/eval_rag.py` | 评估脚本主体（chunks + RAG 两部分） |
| `data/eval/golden_queries.example.json` | golden 格式模板，复制后填充 |
| `data/eval/golden_queries.json` | 你的实际 golden 测试集（需手动创建） |
| `data/eval/last_report.json` | 上一次运行结果（自动生成） |
| `config/model_config.py` | `RAGConfig` 中的切分/检索参数 |
| `src/ingestion.py` | 控制切分参数（chunk_size=1000, overlap=150） |
| `src/rag_chain.py` | 检索逻辑（混合搜索 + Cross-Encoder 重排） |
