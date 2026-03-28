"""
`src` 包入口：对外导出 Agent、RAG 链与工具工厂函数。

便于 `from src import create_agent` 等用法；版本号与作者元数据仅作包标识。
"""

__version__ = "0.1.0"
__author__ = "Finance Agent Team"

from src.agent import create_agent, FinanceAgent
from src.rag_chain import create_rag_chain, AdvancedRAGChain
from src.tools import create_tools

__all__ = [
    "create_agent",
    "FinanceAgent",
    "create_rag_chain",
    "AdvancedRAGChain",
    "create_tools",
]
