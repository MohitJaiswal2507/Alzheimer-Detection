"""
NeuroScan AI — RAG Document Loader
Loads educational Markdown documents from data/knowledge_base/,
parses frontmatter metadata, and validates document integrity.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from ..config import KNOWLEDGE_BASE_DIR


class Document:
    """Represents an ingested educational document with content and metadata."""

    def __init__(self, content: str, metadata: Dict[str, Any], doc_id: str):
        self.content = content.strip()
        self.metadata = metadata
        self.doc_id = doc_id

    def __repr__(self) -> str:
        return f"<Document id='{self.doc_id}' title='{self.metadata.get('title', 'Untitled')}' chars={len(self.content)}>"


def parse_frontmatter(raw_text: str) -> tuple[Dict[str, Any], str]:
    """
    Extracts YAML-style frontmatter enclosed in '---' markers.
    Returns (metadata_dict, body_content).
    """
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, raw_text, re.DOTALL)
    if not match:
        return {}, raw_text.strip()

    frontmatter_text = match.group(1)
    body = raw_text[match.end():].strip()

    metadata: Dict[str, Any] = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            metadata[key] = val

    return metadata, body


def load_knowledge_document(file_path: Path) -> Optional[Document]:
    """
    Loads and validates a single Markdown file.
    Rejects empty documents or files without content.
    """
    file_path = Path(file_path)
    if not file_path.is_file() or file_path.suffix.lower() != ".md":
        return None

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise IOError(f"Could not read knowledge file {file_path}: {e}")

    metadata, body = parse_frontmatter(raw_text)

    # Require non-empty content
    if not body or len(body.strip()) < 20:
        return None

    # Guarantee required metadata fields
    if "title" not in metadata:
        # Fallback to first H1 in body or file stem
        h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        metadata["title"] = h1_match.group(1).strip() if h1_match else file_path.stem.replace("_", " ").title()

    metadata["source_file"] = file_path.name
    metadata.setdefault("publisher", "NeuroScan AI Educational Reference")
    metadata.setdefault("source_url", "https://github.com/MohitJaiswal2507/Alzheimer-Detection")
    metadata.setdefault("category", "General Medical Education")

    return Document(content=body, metadata=metadata, doc_id=file_path.stem)


def load_knowledge_base(kb_dir: Optional[Path] = None) -> List[Document]:
    """
    Scans the knowledge base directory and returns all valid Documents.
    Rejects empty or non-markdown files.
    """
    if kb_dir is None:
        kb_dir = KNOWLEDGE_BASE_DIR

    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return []

    documents = []
    for md_file in sorted(kb_path.glob("*.md")):
        doc = load_knowledge_document(md_file)
        if doc is not None:
            documents.append(doc)

    return documents
