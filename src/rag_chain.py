"""
RAG 检索链模块。

负责从 Chroma 向量库加载文档，支持语义检索 + BM25 混合检索，
以及 Cross-Encoder 重排序；对外提供检索、拼上下文等能力，供 Agent 与 CLI 调用。
"""
import os
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config.model_config import EmbeddingConfig, RAGConfig

class AdvancedRAGChain:
    """
    高级 RAG 流水线：混合检索（BM25 + 向量）+ Cross-Encoder 重排序 + 来源标注。
    """

    def __init__(
        self,
        vector_db_path: str = "data/vector_db",
        embedding_config: Optional[EmbeddingConfig] = None,
        rag_config: Optional[RAGConfig] = None
    ):
        """
        初始化 RAG 链：加载嵌入模型、向量库、检索器与（可选）重排序模型。

        参数:
            vector_db_path: Chroma 持久化目录路径。
            embedding_config: 嵌入模型配置，默认使用 EmbeddingConfig。
            rag_config: 检索与重排参数，默认使用 RAGConfig。
        """
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.rag_config = rag_config or RAGConfig()
        self.vector_db_path = vector_db_path

        if self.rag_config.verbose:
            print("📦 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_config.model_name,
            model_kwargs={'device': self.embedding_config.device},
            encode_kwargs={'normalize_embeddings': self.embedding_config.normalize_embeddings}
        )

        if self.rag_config.verbose:
            print("📚 Loading vector database...")
        if not os.path.exists(vector_db_path):
            raise ValueError(
                f"Vector database not found at {vector_db_path}. "
                "Please run ingestion.py first to create the database."
            )

        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )

        self._init_retrievers()

        if self.rag_config.rerank_top_k > 0:
            if self.rag_config.verbose:
                print("🔄 Loading reranker model...")
            self._init_reranker()
        else:
            self.reranker_model = None
            self.reranker_tokenizer = None

    def _init_retrievers(self):
        """初始化语义检索器；若开启混合检索则再构建 BM25 与融合检索器。"""
        self.semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.rag_config.retrieval_top_k}
        )

        if self.rag_config.use_hybrid_search:
            if self.rag_config.verbose:
                print("🔍 Initializing hybrid search (Semantic + BM25)...")
            all_docs = self.vectorstore.get()
            if all_docs and 'documents' in all_docs:
                docs_for_bm25 = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(
                        all_docs['documents'],
                        all_docs['metadatas'] if all_docs['metadatas'] else [{}] * len(all_docs['documents'])
                    )
                ]
                self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
                self.bm25_retriever.k = self.rag_config.retrieval_top_k

                self.retriever = EnsembleRetriever(
                    retrievers=[self.semantic_retriever, self.bm25_retriever],
                    weights=[self.rag_config.semantic_weight, self.rag_config.bm25_weight]
                )
            else:
                if self.rag_config.verbose:
                    print("⚠️ No documents found for BM25, using semantic search only")
                self.retriever = self.semantic_retriever
        else:
            self.retriever = self.semantic_retriever

    def _init_reranker(self):
        """加载 Cross-Encoder 重排序模型与分词器；若存在 GPU 则放到 CUDA。"""
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.rag_config.reranker_model
        )
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self.rag_config.reranker_model
        )
        self.reranker_model.eval()

        if torch.cuda.is_available():
            self.reranker_model = self.reranker_model.to('cuda')

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        使用 Cross-Encoder 对候选文档打分并按分数降序截取 top-k。

        参数:
            query: 用户查询。
            documents: 初检得到的文档列表。

        返回:
            重排后的文档列表（长度不超过 rag_config.rerank_top_k），
            并在 metadata 中写入与文档对齐的 relevance_score 与 rank。
        """
        if not self.reranker_model or len(documents) == 0:
            return documents

        pairs = [[query, doc.page_content] for doc in documents]

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1).float()
            scores = scores.cpu().numpy()

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 按排序后的 (文档, 分数) 对截取 top-k，保证 metadata 中的分数与文档一一对应
        top_pairs = doc_score_pairs[:self.rag_config.rerank_top_k]
        reranked_docs = [doc for doc, _ in top_pairs]

        for i, (doc, score) in enumerate(top_pairs):
            doc.metadata['relevance_score'] = float(score)
            doc.metadata['rank'] = i + 1

        return reranked_docs

    def retrieve(self, query: str, with_rerank: bool = True) -> List[Document]:
        """
        根据查询检索相关文档；可选在初检后做 Cross-Encoder 重排。

        参数:
            query: 用户问题。
            with_rerank: 是否启用重排序（需已加载 reranker）。

        返回:
            文档列表。
        """
        if self.rag_config.verbose:
            print(f"🔍 Retrieving documents for: '{query}'")

        documents = self.retriever.invoke(query)
        if self.rag_config.verbose:
            print(f"📄 Retrieved {len(documents)} initial documents")

        if with_rerank and self.reranker_model:
            if self.rag_config.verbose:
                print(f"🔄 Reranking to top {self.rag_config.rerank_top_k}...")
            documents = self._rerank_documents(query, documents)
            if self.rag_config.verbose:
                print(f"✅ Reranked to {len(documents)} documents")

        return documents

    def get_context(self, query: str, with_rerank: bool = True) -> Dict[str, Any]:
        """
        检索并拼接可供 LLM 使用的上下文字符串，同时返回结构化来源信息。

        参数:
            query: 用户问题。
            with_rerank: 是否对初检结果做重排。

        返回:
            包含 context、documents、sources、num_documents 的字典。
        """
        documents = self.retrieve(query, with_rerank)

        context_parts = []
        sources = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            relevance = doc.metadata.get('relevance_score', 'N/A')

            context_parts.append(f"[Document {i}]\n{doc.page_content}\n")
            sources.append({
                'rank': i,
                'source': source,
                'relevance_score': relevance,
                'content_preview': doc.page_content[:200] + "..."
            })

        return {
            'context': '\n'.join(context_parts),
            'documents': documents,
            'sources': sources,
            'num_documents': len(documents)
        }

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        纯向量相似度检索，不经过 BM25 融合与重排序。

        参数:
            query: 查询文本。
            k: 返回条数。

        返回:
            最相近的 k 条文档。
        """
        return self.vectorstore.similarity_search(query, k=k)


def create_rag_chain(
    vector_db_path: str = "data/vector_db",
    embedding_config: Optional[EmbeddingConfig] = None,
    rag_config: Optional[RAGConfig] = None
) -> AdvancedRAGChain:
    """
    工厂函数：创建并返回配置好的 AdvancedRAGChain 实例。

    参数:
        vector_db_path: 向量库路径。
        embedding_config: 嵌入配置。
        rag_config: RAG 配置。

    返回:
        AdvancedRAGChain 实例。
    """
    return AdvancedRAGChain(
        vector_db_path=vector_db_path,
        embedding_config=embedding_config,
        rag_config=rag_config
    )


if __name__ == "__main__":
    # 直接运行本文件时做一次 RAG 链路自检（需已执行 ingestion 构建向量库）
    print("🧪 Testing RAG Chain...")

    try:
        rag = create_rag_chain()

        test_query = "What was Apple's revenue in the last fiscal year?"
        result = rag.get_context(test_query)

        print(f"\n📊 Query: {test_query}")
        print(f"📚 Retrieved {result['num_documents']} documents")
        print(f"\n📝 Context Preview:\n{result['context'][:500]}...")
        print(f"\n🔗 Sources:")
        for source in result['sources']:
            print(f"  [{source['rank']}] {source['source']} (Score: {source['relevance_score']})")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've run ingestion.py first to create the vector database!")
