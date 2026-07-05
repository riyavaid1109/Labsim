"""
chunking.py — recursive, markdown-aware text chunking.

Strategy: split on the largest structural boundary that keeps chunks under
`chunk_size`, recursing down the separator hierarchy when a piece is too big:

    markdown headers → blank lines (paragraphs) → sentences → words → chars

Overlap is applied between adjacent chunks so retrieval doesn't lose context
at chunk boundaries.

Defaults (chunk_size=512, overlap=64 chars) were chosen because:
  - nomic-embed-text has a 2048-token context but embedding quality degrades
    on long passages; ~512 chars ≈ 100–130 tokens keeps chunks semantically
    focused.
  - 64-char overlap (~12%) preserves sentence continuity across boundaries
    without inflating the index much.
"""

import re

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64

# Ordered from coarsest to finest structural boundary.
_SEPARATORS = [
    r"\n#{1,6} ",   # markdown headers
    r"\n\n+",       # paragraphs
    r"(?<=[.!?])\s+",  # sentences
    r"\s+",         # words
]


def _split(text: str, sep_index: int, chunk_size: int) -> list[str]:
    """Recursively split text until every piece fits in chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if sep_index >= len(_SEPARATORS):
        # No separators left — hard cut.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = re.split(_SEPARATORS[sep_index], text)
    out: list[str] = []
    buf = ""
    for piece in pieces:
        if not piece.strip():
            continue
        candidate = f"{buf}\n{piece}".strip() if buf else piece
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if len(piece) <= chunk_size:
                buf = piece
            else:
                out.extend(_split(piece, sep_index + 1, chunk_size))
                buf = ""
    if buf:
        out.append(buf)
    return out


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks along structural boundaries.

    Returns a list of chunk strings. Overlap is taken from the tail of the
    previous chunk (rounded back to a word boundary).
    """
    if not text or not text.strip():
        return []
    base = _split(text.strip(), 0, chunk_size)
    if overlap <= 0 or len(base) <= 1:
        return base

    chunks = [base[0]]
    for prev, cur in zip(base, base[1:]):
        tail = prev[-overlap:]
        # Round back to a word boundary so overlap doesn't start mid-word.
        space = tail.find(" ")
        if 0 <= space < len(tail) - 1:
            tail = tail[space + 1:]
        chunks.append(f"{tail} {cur}".strip())
    return chunks


def chunk_documents(
    docs: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    source_names: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Chunk a list of documents. Returns (chunks, metadatas) ready for
    RAGRetriever.index(). Metadata records source doc and chunk position.
    """
    all_chunks: list[str] = []
    all_meta: list[dict] = []
    for i, doc in enumerate(docs):
        source = source_names[i] if source_names else f"doc_{i}"
        pieces = chunk_text(doc, chunk_size, overlap)
        for j, piece in enumerate(pieces):
            all_chunks.append(piece)
            all_meta.append({"source": source, "chunk_index": j, "n_chunks": len(pieces)})
    return all_chunks, all_meta
