"""
NeuroScan AI — RAG Document Chunker
Splits educational Markdown documents into semantically coherent, sentence-aware chunks
while preserving complete sentences, paragraph boundaries, and source metadata.
Eliminates sentence fragmentation, cut-off words, and repetitive paragraphs.
"""
from typing import List, Dict, Any, Optional
import re
from .document_loader import Document


class DocumentChunk:
    """Represents a discrete text chunk with associated source metadata."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: str,
        chunk_index: int,
        parent_doc_id: str,
    ):
        self.text = text.strip()
        self.metadata = metadata.copy()
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index
        self.parent_doc_id = parent_doc_id

        # Enrich metadata for retrieval transparency
        self.metadata["chunk_id"] = chunk_id
        self.metadata["chunk_index"] = chunk_index
        self.metadata["parent_doc_id"] = parent_doc_id

    def __repr__(self) -> str:
        return f"<DocumentChunk id='{self.chunk_id}' chars={len(self.text)} parent='{self.parent_doc_id}'>"


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into complete sentences without cutting words or abbreviations.
    Uses positive lookbehind for sentence-ending punctuation and lookahead for
    capitalized start of next sentence or list item.
    """
    text = text.strip()
    if not text:
        return []

    # Split on sentence terminators followed by whitespace and a capital letter or list marker
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9#\-\*\(])', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


def chunk_text(
    text: str,
    chunk_size: int = 850,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    Splits text into complete-sentence, paragraph-aware chunks.
    Preserves whole paragraphs whenever possible, merging lone headings with their content
    and avoiding mid-sentence cuts or repetitive paragraphs.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Split text into natural semantic blocks (by double newlines)
    raw_blocks = re.split(r'\n{2,}', text)
    blocks = [b.strip() for b in raw_blocks if b.strip()]

    # Merge lone headings with subsequent content blocks
    merged_blocks: List[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.startswith("#") and "\n" not in b and i + 1 < len(blocks):
            merged_blocks.append(f"{b}\n\n{blocks[i+1]}")
            i += 2
        else:
            merged_blocks.append(b)
            i += 1

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for block in merged_blocks:
        block_len = len(block)

        # If block fits in current buffer, append it
        if current_len + block_len + 2 <= chunk_size:
            current_chunk.append(block)
            current_len += block_len + 2
        else:
            # Flush current buffer if it has content
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            # If the block itself fits within chunk_size, make it current
            if block_len <= chunk_size:
                current_chunk.append(block)
                current_len = block_len
            else:
                # Block is larger than chunk_size: split cleanly by full sentences
                sentences = split_into_sentences(block)
                sub_chunk: List[str] = []
                sub_len = 0

                for sent in sentences:
                    if sub_len + len(sent) + 1 <= chunk_size:
                        sub_chunk.append(sent)
                        sub_len += len(sent) + 1
                    else:
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
                        # Handle sentence overlap if requested
                        if chunk_overlap > 0 and sub_chunk:
                            sub_chunk = [sub_chunk[-1], sent]
                            sub_len = len(sub_chunk[0]) + len(sent) + 1
                        else:
                            sub_chunk = [sent]
                            sub_len = len(sent)

                if sub_chunk:
                    current_chunk = [" ".join(sub_chunk)]
                    current_len = sub_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 850,
    chunk_overlap: int = 0,
) -> List[DocumentChunk]:
    """
    Chunks a single Document into DocumentChunk instances,
    preserving all document metadata on each chunk.
    """
    text_chunks = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_chunks = []

    for idx, text in enumerate(text_chunks):
        chunk_id = f"{doc.doc_id}_c{idx:03d}"
        doc_chunk = DocumentChunk(
            text=text,
            metadata=doc.metadata,
            chunk_id=chunk_id,
            chunk_index=idx,
            parent_doc_id=doc.doc_id,
        )
        document_chunks.append(doc_chunk)

    return document_chunks


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 850,
    chunk_overlap: int = 0,
) -> List[DocumentChunk]:
    """Chunks a list of Document instances into a flat list of DocumentChunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
