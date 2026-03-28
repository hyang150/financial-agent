"""
`config` 包：聚合 `model_config` 中的各数据类与 `get_config` 工厂。

供 Agent、RAG、训练脚本统一导入配置类型。
"""

from .model_config import (
    ModelConfig,
    EmbeddingConfig,
    RAGConfig,
    AgentConfig,
    get_config
)

__all__ = [
    "ModelConfig",
    "EmbeddingConfig",
    "RAGConfig",
    "AgentConfig",
    "get_config"
]
