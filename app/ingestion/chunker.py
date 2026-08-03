"""✂️ Semantic Chunking (DOCS/02_INGESTION_ENGINE.md §2).

* Chunk size: 1500 characters.
* Overlap: NATURAL — we split by paragraph breaks (\\n\\n), never mid-sentence.
* Goal: keep paragraphs together so no chunk is cut off mid-sentence and the
  LLM never receives "hallucinated" fragments.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1500) -> list[str]:
    """Split text into paragraph-aware chunks of ~`chunk_size` characters.

    Strategy:
      1. Normalise the whitespace.
      2. Group paragraphs greedily, never exceeding `chunk_size`.
      3. A single paragraph longer than `chunk_size` is hard-split at word
         boundaries (rare, but must never be dropped).
    """
    # Normalise: collapse stray newlines, keep paragraph breaks as \n\n
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the "\n\n" separator

        if current and current_len + para_len > chunk_size:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0

        if para_len > chunk_size:
            # Oversized paragraph — hard split at word boundaries.
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_split(para, chunk_size))
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


def _hard_split(paragraph: str, chunk_size: int) -> list[str]:
    """Split a single over-long paragraph without ever splitting a word."""
    words = paragraph.split(" ")
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in words:
        word_len = len(word) + 1  # +1 for the space
        if buf and buf_len + word_len > chunk_size:
            pieces.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += word_len
    if buf:
        pieces.append(" ".join(buf))
    return pieces
