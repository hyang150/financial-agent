"""
Model configuration for Finance Agent
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """LLM Model Configuration"""
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
        if self.lora_target_modules is None:
            # Default LoRA target modules for Qwen2.5
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

@dataclass
class EmbeddingConfig:
    """Embedding Model Configuration"""
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32

@dataclass
class RAGConfig:
    """RAG Pipeline Configuration"""
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
    """Agent Configuration"""
    max_iterations: int = 10
    max_execution_time: int = 30  # seconds
    verbose: bool = True
    allow_dangerous_code: bool = False  # Safety flag for Python REPL

def get_config():
    """Load configuration from environment variables"""
    return {
        "hf_token": os.getenv("HF_TOKEN"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "sec_email": os.getenv("SEC_EMAIL", "student@university.edu"),
        "device": os.getenv("DEVICE", "cuda"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "llm_model": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    }
