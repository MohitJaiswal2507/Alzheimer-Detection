"""
NeuroScan AI — Educational RAG Knowledge Assistant Subpackage
Provides document loading, chunking, vector indexing, retrieval,
and grounded question answering.
"""
from .document_loader import Document, load_knowledge_document, load_knowledge_base
from .chunker import DocumentChunk, chunk_text, chunk_document, chunk_documents
from .vector_store import (
    EducationalVectorStore,
    is_vector_store_available,
    get_dependency_status,
)
from .retriever import EducationalRetriever, format_source_citations
from .generator import generate_rag_answer, build_rag_system_prompt, build_rag_user_prompt
from .pipeline import RAGPipeline, get_rag_pipeline, ask_knowledge_assistant

__all__ = [
    "Document",
    "load_knowledge_document",
    "load_knowledge_base",
    "DocumentChunk",
    "chunk_text",
    "chunk_document",
    "chunk_documents",
    "EducationalVectorStore",
    "is_vector_store_available",
    "get_dependency_status",
    "EducationalRetriever",
    "format_source_citations",
    "generate_rag_answer",
    "build_rag_system_prompt",
    "build_rag_user_prompt",
    "RAGPipeline",
    "get_rag_pipeline",
    "ask_knowledge_assistant",
]
