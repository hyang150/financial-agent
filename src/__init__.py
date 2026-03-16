"""
Finance Agent - Source Package
Multi-Tool RAG & Fine-tuned LLM for Financial Analysis
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
