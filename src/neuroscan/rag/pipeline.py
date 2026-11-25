"""
NeuroScan AI — RAG Pipeline Orchestrator
Coordinates document loading, chunking, vector indexing, retrieval,
and answer generation with strict educational guardrails.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path

from .document_loader import load_knowledge_base
from .chunker import chunk_documents
from .vector_store import (
    EducationalVectorStore,
    is_vector_store_available,
    get_dependency_status,
)
from .retriever import EducationalRetriever
from .generator import generate_rag_answer, sanitize_input


class RAGPipeline:
    """End-to-end RAG workflow for educational knowledge queries."""

    def __init__(
        self,
        vector_store: Optional[EducationalVectorStore] = None,
        retriever: Optional[EducationalRetriever] = None,
    ):
        self.vector_store = vector_store or EducationalVectorStore()
        self.retriever = retriever or EducationalRetriever(self.vector_store)

    def initialize_knowledge_base(self, recreate: bool = False) -> Dict[str, Any]:
        """
        Loads all educational documents from data/knowledge_base/, chunks them,
        and indexes them into the local ChromaDB collection (if ChromaDB is installed).
        """
        docs = load_knowledge_base()
        if not docs:
            return {
                "success": False,
                "indexed_chunks": 0,
                "documents_loaded": 0,
                "message": "No documents found in knowledge base directory.",
            }

        chunks = chunk_documents(docs)

        if not is_vector_store_available():
            return {
                "success": False,
                "indexed_chunks": 0,
                "documents_loaded": len(docs),
                "chunks_prepared": len(chunks),
                "message": "ChromaDB or sentence-transformers is missing from the environment.",
                "dependencies": get_dependency_status(),
            }

        indexed_count = self.vector_store.index_chunks(chunks, recreate=recreate)
        return {
            "success": True,
            "indexed_chunks": indexed_count,
            "documents_loaded": len(docs),
            "chunks_prepared": len(chunks),
            "message": f"Successfully indexed {indexed_count} chunks into ChromaDB.",
        }

    def ask(
        self,
        query: str,
        language: str = "english",
        top_k: int = 3,
        use_fallback_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Answers a user's educational query using retrieved knowledge base context.

        Args:
            query: The user's question string.
            language: "english" or "hinglish".
            top_k: Number of chunks to retrieve.
            use_fallback_on_error: Whether to return rule-based fallback if LLM fails.

        Returns:
            Dict containing query, answer, sources, status, retrieval_method, and error.
        """
        clean_query = sanitize_input(query)

        # 1. Retrieve relevant chunks
        retrieval_res = self.retriever.retrieve(clean_query, top_k=top_k)
        retrieved_chunks = retrieval_res["chunks"]
        retrieval_method = retrieval_res["method"]

        # 2. Generate grounded answer
        answer_res = generate_rag_answer(
            query=clean_query,
            retrieved_chunks=retrieved_chunks,
            language=language,
            use_fallback_on_error=use_fallback_on_error,
        )

        return {
            "query": clean_query,
            "answer": answer_res["answer"],
            "sources": answer_res["sources"],
            "status": answer_res["status"],
            "retrieval_method": retrieval_method,
            "language": language,
            "error": answer_res.get("error"),
        }


# Global singleton instance for clean application and test imports
_default_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Returns the shared RAGPipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def ask_knowledge_assistant(
    query: str,
    language: str = "english",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Convenience function to query the educational knowledge assistant."""
    pipeline = get_rag_pipeline()
    return pipeline.ask(query=query, language=language, top_k=top_k)
