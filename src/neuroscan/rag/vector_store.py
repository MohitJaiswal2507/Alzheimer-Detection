"""
NeuroScan AI — RAG Vector Store
Manages local ChromaDB persistence, collection indexing, and embedding queries.
Features graceful fallback when ChromaDB or sentence-transformers dependencies are not installed.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os

from ..config import CHROMA_DB_DIR, DEFAULT_EMBEDDING_MODEL
from .chunker import DocumentChunk

# Check dependency availability safely
CHROMA_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore


COLLECTION_NAME = "neuroscan_educational_kb"


def is_vector_store_available() -> bool:
    """Returns True only if both chromadb and sentence-transformers are installed."""
    return CHROMA_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE


def get_dependency_status() -> Dict[str, Any]:
    """Returns diagnostic status of RAG dependencies."""
    return {
        "chromadb_installed": CHROMA_AVAILABLE,
        "sentence_transformers_installed": SENTENCE_TRANSFORMERS_AVAILABLE,
        "ready": is_vector_store_available(),
        "db_directory": str(CHROMA_DB_DIR.resolve()),
        "collection_name": COLLECTION_NAME,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


class LocalEmbeddingFunction:
    """Wrapper adapting SentenceTransformer to ChromaDB's embedding function protocol."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed in the active environment. "
                    "Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(input_texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()


class EducationalVectorStore:
    """
    Manages a persistent local ChromaDB collection containing educational chunks.
    Provides indexing, deduplication, and similarity queries.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = Path(persist_dir or CHROMA_DB_DIR)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _init_client(self):
        """Initializes Chroma client and persistent collection."""
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is not installed in the active Python environment. "
                "Run: pip install chromadb"
            )

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )

        if self._embedding_fn is None:
            self._embedding_fn = LocalEmbeddingFunction(self.embedding_model_name)

        if self._collection is None:
            # Upsert collection with embedding function
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"description": "NeuroScan AI Curated Educational Knowledge Base"}
            )

    def count(self) -> int:
        """Returns the number of indexed chunks in the collection."""
        if not is_vector_store_available():
            return 0
        try:
            self._init_client()
            return self._collection.count()
        except Exception:
            return 0

    def index_chunks(self, chunks: List[DocumentChunk], recreate: bool = False) -> int:
        """
        Indexes a list of DocumentChunks into ChromaDB.
        Deduplicates chunks based on chunk_id.
        Returns the total number of items indexed.
        """
        if not is_vector_store_available():
            raise ImportError(
                "Cannot index chunks: chromadb or sentence-transformers is missing. "
                "Run: pip install chromadb sentence-transformers"
            )

        self._init_client()

        if recreate:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"description": "NeuroScan AI Curated Educational Knowledge Base"}
            )

        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        # Metadata dictionary values must be str, int, float, or bool for ChromaDB
        metadatas = [
            {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in c.metadata.items()}
            for c in chunks
        ]

        # Use upsert to avoid duplicate key errors
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        return self._collection.count()

    def query_similarity(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector store for the top-k most semantically similar chunks.
        Returns a list of result dictionaries containing text, metadata, and distance.
        """
        if not is_vector_store_available():
            return []

        query = query.strip()
        if not query:
            return []

        self._init_client()
        total_items = self._collection.count()
        if total_items == 0:
            return []

        actual_k = min(top_k, total_items)
        results = self._collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            formatted_results.append({
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 0.0,
            })

        return formatted_results
