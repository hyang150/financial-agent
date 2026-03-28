"""
Finance Agent 全局配置数据类。

定义主 LLM、嵌入模型、RAG 检索参数、Agent 行为及从环境变量读取键值对的辅助函数。
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """主对话模型名称、生成参数及（预留）LoRA 相关字段。"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True

    # LoRA Configuration (for fine-tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None

    def __post_init__(self):
        """初始化默认 LoRA 目标层列表（与 Qwen2.5 结构常见命名一致）。"""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

@dataclass
class EmbeddingConfig:
    """向量嵌入模型、设备与归一化等选项。"""
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32

@dataclass
class RAGConfig:
    """RAG 分块、Top-K、重排模型与混合检索权重配置。"""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_k: int = 3
    reranker_model: str = "BAAI/bge-reranker-base"

    # Hybrid search weights
    use_hybrid_search: bool = True
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7

@dataclass
class AgentConfig:
    """Agent 迭代上限、超时、详细日志与是否允许危险代码等。"""
    max_iterations: int = 10
    max_execution_time: int = 30  # seconds
    verbose: bool = True
    allow_dangerous_code: bool = False  # Safety flag for Python REPL

def get_config():
    """从环境变量读取常用密钥与模型相关字符串，返回字典。"""
    return {
        "hf_token": os.getenv("HF_TOKEN"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "sec_email": os.getenv("SEC_EMAIL", "student@university.edu"),
        "device": os.getenv("DEVICE", "cuda"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "llm_model": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    }
