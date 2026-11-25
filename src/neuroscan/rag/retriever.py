"""
NeuroScan AI — RAG Retriever Module
Retrieves relevant educational chunks from the vector store with formatted source metadata.
Includes an intelligent lexical keyword search with relevance thresholds and deduplication
when ChromaDB is not installed or available.
"""
from typing import List, Dict, Any, Optional, Set
import re
from pathlib import Path

from .vector_store import (
    EducationalVectorStore,
    is_vector_store_available,
)
from .document_loader import load_knowledge_base
from .chunker import chunk_documents, DocumentChunk

# Common stopwords to exclude from domain keyword search
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "tell", "explain", "please", "give", "show", "describe",
}

# Domain keyword synonyms and expansions
SYNONYM_MAP = {
    "gradcam": ["grad-cam", "gradcam", "heatmap", "saliency"],
    "grad-cam": ["grad-cam", "gradcam", "heatmap", "saliency"],
    "efficientnet": ["efficientnet", "efficientnet-b0", "compound scaling", "mbconv"],
    "b0": ["efficientnet", "efficientnet-b0"],
    "axial": ["axial", "transverse", "slice", "plane"],
    "categories": ["categories", "classes", "mild_demented", "moderate_demented", "non_demented", "very_mild_demented"],
    "classes": ["categories", "classes", "mild_demented", "moderate_demented", "non_demented", "very_mild_demented"],
    "leakage": ["leakage", "patient-level", "data leakage", "slice leakage"],
    "alzheimer": ["alzheimer's", "alzheimer", "dementia", "neurodegenerative"],
    "alzheimers": ["alzheimer's", "alzheimer", "dementia", "neurodegenerative"],
}


def format_source_citations(chunks_or_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extracts and deduplicates source citations from retrieved chunks.
    Guarantees every cited source has title, publisher, URL, and access date.
    """
    seen_urls: Set[str] = set()
    sources = []

    for item in chunks_or_results:
        meta = item.get("metadata", {})
        url = meta.get("source_url", "").strip()
        title = meta.get("title", "Educational Reference").strip()
        publisher = meta.get("publisher", "NeuroScan AI Knowledge Base").strip()
        access_date = meta.get("access_date", "2026-10-29").strip()
        category = meta.get("category", "General Medical Education").strip()

        dedup_key = f"{title}_{url}"
        if dedup_key not in seen_urls:
            seen_urls.add(dedup_key)
            sources.append({
                "title": title,
                "publisher": publisher,
                "source_url": url or "https://github.com/MohitJaiswal2507/Alzheimer-Detection",
                "access_date": access_date,
                "category": category,
            })

    return sources


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculates word-level Jaccard similarity between two texts to filter near-duplicates."""
    words1 = set(re.findall(r"\b\w{3,}\b", text1.lower()))
    words2 = set(re.findall(r"\b\w{3,}\b", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return float(intersection) / float(union) if union > 0 else 0.0


class EducationalRetriever:
    """
    Unified retriever supporting dense ChromaDB similarity search with automatic
    keyword fallback when vector database dependencies are not installed.
    Enforces minimum relevance thresholding and near-duplicate removal.
    """

    def __init__(self, vector_store: Optional[EducationalVectorStore] = None):
        self.vector_store = vector_store or EducationalVectorStore()
        self._cached_fallback_chunks: Optional[List[DocumentChunk]] = None

    def _get_fallback_chunks(self) -> List[DocumentChunk]:
        """Loads and chunks raw knowledge base documents for keyword matching fallback."""
        if self._cached_fallback_chunks is None:
            docs = load_knowledge_base()
            self._cached_fallback_chunks = chunk_documents(docs)
        return self._cached_fallback_chunks

    def _extract_query_keywords(self, query: str) -> Set[str]:
        """Extracts meaningful, non-stopword tokens and expands domain synonyms."""
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query.lower())
        keywords: Set[str] = set()

        for token in raw_tokens:
            if token not in STOPWORDS and len(token) >= 3:
                keywords.add(token)
                if token in SYNONYM_MAP:
                    keywords.update(SYNONYM_MAP[token])

        # Also check for key phrases
        q_lower = query.lower()
        if "four categories" in q_lower or "4 categories" in q_lower or "four classes" in q_lower:
            keywords.update(["categories", "classes", "moderate_demented", "mild_demented", "non_demented"])
        if "grad-cam" in q_lower or "gradcam" in q_lower:
            keywords.update(["grad-cam", "gradcam", "heatmap", "saliency"])
        if "axial" in q_lower:
            keywords.update(["axial", "transverse", "slice", "plane"])
        if "data leakage" in q_lower or "patient leakage" in q_lower:
            keywords.update(["leakage", "patient-level", "slice"])

        return keywords

    def _keyword_search_fallback(
        self,
        query: str,
        top_k: int = 3,
        min_relevance_score: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Lexical token-matching search over knowledge base chunks with relevance
        thresholds and near-duplicate filtering.
        """
        chunks = self._get_fallback_chunks()
        if not chunks:
            return []

        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            return []

        scored_chunks = []
        for c in chunks:
            text_lower = c.text.lower()
            title_lower = str(c.metadata.get("title", "")).lower()
            category_lower = str(c.metadata.get("category", "")).lower()

            score = 0
            for kw in query_keywords:
                # Strong boost for match in document title or category
                if kw in title_lower:
                    score += 10
                if kw in category_lower:
                    score += 4
                # Header matches
                if f"#{kw}" in text_lower or f"## {kw}" in text_lower or f"### {kw}" in text_lower:
                    score += 6
                # Text matches
                if kw in text_lower:
                    score += text_lower.count(kw)

            # Apply relevance threshold: only keep chunks with meaningful score
            if score >= min_relevance_score:
                scored_chunks.append((score, c))

        if not scored_chunks:
            return []

        # Sort descending by relevance score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Select top-k while filtering near-duplicates
        results: List[Dict[str, Any]] = []
        selected_texts: List[str] = []

        for score, c in scored_chunks:
            # Check near-duplicate overlap with already selected chunks
            is_duplicate = False
            for prev_text in selected_texts:
                if calculate_jaccard_similarity(c.text, prev_text) > 0.40:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            results.append({
                "text": c.text,
                "metadata": c.metadata,
                "distance": max(0.0, 1.0 - (score / 20.0)),
                "match_type": "keyword_fallback",
                "score": score,
            })
            selected_texts.append(c.text)

            if len(results) >= top_k:
                break

        return results

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves relevant educational chunks for a query.

        Returns:
            Dict containing:
            - chunks: List of matched chunk texts and metadata
            - sources: Deduplicated list of authoritative source citations
            - method: 'chromadb_dense' or 'keyword_fallback'
            - query: original query
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "chunks": [],
                "sources": [],
                "method": "none",
                "query": query,
            }

        chunks: List[Dict[str, Any]] = []
        method = "none"

        # 1. Try ChromaDB dense semantic search if available and indexed
        if is_vector_store_available() and self.vector_store.count() > 0:
            try:
                raw_chunks = self.vector_store.query_similarity(clean_query, top_k=top_k * 2)
                # Deduplicate retrieved dense chunks
                selected = []
                selected_texts = []
                for item in raw_chunks:
                    text = item.get("text", "")
                    if any(calculate_jaccard_similarity(text, prev) > 0.45 for prev in selected_texts):
                        continue
                    selected.append(item)
                    selected_texts.append(text)
                    if len(selected) >= top_k:
                        break
                chunks = selected
                method = "chromadb_dense"
            except Exception:
                chunks = []

        # 2. If vector store is empty, unavailable, or failed, use keyword fallback
        if not chunks:
            chunks = self._keyword_search_fallback(clean_query, top_k=top_k)
            method = "keyword_fallback" if chunks else "none"

        sources = format_source_citations(chunks)

        return {
            "chunks": chunks,
            "sources": sources,
            "method": method,
            "query": clean_query,
        }
