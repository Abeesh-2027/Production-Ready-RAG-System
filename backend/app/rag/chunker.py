from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.loader import LoadedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    page_number: int | None
    chunk_index: int
    text: str

    def metadata(self) -> dict:
        """Flat metadata dict for ChromaDB (values must be str/int/float/bool)."""
        meta = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
        }
        if self.page_number is not None:  # Chroma rejects None values
            meta["page_number"] = self.page_number
        return meta


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Break an oversized paragraph into pieces of at most `chunk_size` words."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        n = _word_count(sentence)
        if n > chunk_size:  # a single enormous "sentence": cut on words
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                pieces.append(" ".join(words[i:i + chunk_size]))
            continue
        if count + n > chunk_size and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one block of text into overlapping, paragraph-aware chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    # Overlap must be smaller than the chunk, otherwise we would never advance.
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    units: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if _word_count(paragraph) > chunk_size:
            units.extend(_split_long_paragraph(paragraph, chunk_size))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        n = _word_count(unit)
        if current and count + n > chunk_size:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            tail = chunk_text.split()[-chunk_overlap:] if chunk_overlap else []
            current = [" ".join(tail)] if tail else []
            count = len(tail)
        current.append(unit)
        count += n
    if current:
        final = "\n\n".join(current)
        # Avoid emitting a chunk that is nothing but the previous chunk's overlap.
        if not chunks or final.strip() != " ".join(chunks[-1].split()[-chunk_overlap:]).strip():
            chunks.append(final)
    return [c for c in chunks if c.strip()]


def chunk_document(
    document: LoadedDocument,
    cleaned_pages: list[tuple[int | None, str]],
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk every page and attach source metadata to each chunk."""
    chunks: list[Chunk] = []
    index = 0
    for page_number, text in cleaned_pages:
        for piece in split_text(text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chunk_{index:04d}",
                    document_id=document_id,
                    filename=document.filename,
                    file_type=document.file_type,
                    page_number=page_number,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks
