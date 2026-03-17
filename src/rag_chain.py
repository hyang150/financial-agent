"""
RAG Chain Implementation with Hybrid Search and Reranking
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
    Advanced RAG Pipeline with:
    - Hybrid Search (BM25 + Semantic)
    - Cross-Encoder Reranking
    - Source attribution
    """

    def __init__(
        self,
        vector_db_path: str = "data/vector_db",
        embedding_config: Optional[EmbeddingConfig] = None,
        rag_config: Optional[RAGConfig] = None
    ):
        """
        Initialize RAG Chain

        Args:
            vector_db_path: Path to the vector database
            embedding_config: Embedding model configuration
            rag_config: RAG pipeline configuration
        """
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.rag_config = rag_config or RAGConfig()
        self.vector_db_path = vector_db_path

        # Initialize embedding model
        print("📦 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_config.model_name,
            model_kwargs={'device': self.embedding_config.device},
            encode_kwargs={'normalize_embeddings': self.embedding_config.normalize_embeddings}
        )

        # Load vector database
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

        # Initialize retrievers
        self._init_retrievers()

        # Initialize reranker (cross-encoder)
        if self.rag_config.rerank_top_k > 0:
            print("🔄 Loading reranker model...")
            self._init_reranker()
        else:
            self.reranker_model = None
            self.reranker_tokenizer = None

    def _init_retrievers(self):
        """Initialize semantic and BM25 retrievers"""
        # Semantic retriever
        self.semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.rag_config.retrieval_top_k}
        )

        # BM25 retriever (keyword-based)
        if self.rag_config.use_hybrid_search:
            print("🔍 Initializing hybrid search (Semantic + BM25)...")
            all_docs = self.vectorstore.get()
            if all_docs and 'documents' in all_docs:
                # Create Document objects for BM25
                docs_for_bm25 = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(
                        all_docs['documents'],
                        all_docs['metadatas'] if all_docs['metadatas'] else [{}] * len(all_docs['documents'])
                    )
                ]
                self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
                self.bm25_retriever.k = self.rag_config.retrieval_top_k

                # Ensemble retriever combining both
                self.retriever = EnsembleRetriever(
                    retrievers=[self.semantic_retriever, self.bm25_retriever],
                    weights=[self.rag_config.semantic_weight, self.rag_config.bm25_weight]
                )
            else:
                print("⚠️ No documents found for BM25, using semantic search only")
                self.retriever = self.semantic_retriever
        else:
            self.retriever = self.semantic_retriever

    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.rag_config.reranker_model
        )
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self.rag_config.reranker_model
        )
        self.reranker_model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            self.reranker_model = self.reranker_model.to('cuda')

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-encoder

        Args:
            query: User query
            documents: List of retrieved documents

        Returns:
            Reranked list of documents
        """
        if not self.reranker_model or len(documents) == 0:
            return documents

        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc in documents]

        # Compute relevance scores
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

        # Sort documents by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k reranked documents
        reranked_docs = [doc for doc, score in doc_score_pairs[:self.rag_config.rerank_top_k]]

        # Add relevance scores to metadata
        for i, (doc, score) in enumerate(zip(reranked_docs, scores[:self.rag_config.rerank_top_k])):
            doc.metadata['relevance_score'] = float(score)
            doc.metadata['rank'] = i + 1

        return reranked_docs

    def retrieve(self, query: str, with_rerank: bool = True) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            with_rerank: Whether to apply reranking

        Returns:
            List of relevant documents
        """
        print(f"🔍 Retrieving documents for: '{query}'")

        # Initial retrieval
        documents = self.retriever.get_relevant_documents(query)
        print(f"📄 Retrieved {len(documents)} initial documents")

        # Rerank if enabled
        if with_rerank and self.reranker_model:
            print(f"🔄 Reranking to top {self.rag_config.rerank_top_k}...")
            documents = self._rerank_documents(query, documents)
            print(f"✅ Reranked to {len(documents)} documents")

        return documents

    def get_context(self, query: str, with_rerank: bool = True) -> Dict[str, Any]:
        """
        Get retrieval context for a query

        Args:
            query: User query
            with_rerank: Whether to apply reranking

        Returns:
            Dictionary with context and source information
        """
        documents = self.retrieve(query, with_rerank)

        # Format context
        context_parts = []
        sources = []

        for i, doc in enumerate(documents, 1):
            # Extract source info
            source = doc.metadata.get('source', 'Unknown')
            relevance = doc.metadata.get('relevance_score', 'N/A')

            # Format context entry
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
        Simple similarity search without reranking

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(query, k=k)


def create_rag_chain(
    vector_db_path: str = "data/vector_db",
    embedding_config: Optional[EmbeddingConfig] = None,
    rag_config: Optional[RAGConfig] = None
) -> AdvancedRAGChain:
    """
    Factory function to create a RAG chain

    Args:
        vector_db_path: Path to vector database
        embedding_config: Embedding configuration
        rag_config: RAG configuration

    Returns:
        Configured RAG chain
    """
    return AdvancedRAGChain(
        vector_db_path=vector_db_path,
        embedding_config=embedding_config,
        rag_config=rag_config
    )


if __name__ == "__main__":
    # Test the RAG chain
    print("🧪 Testing RAG Chain...")

    try:
        # Create RAG chain
        rag = create_rag_chain()

        # Test query
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
