"""🌍 Universal Ingestion Command (DOCS/02_INGESTION_ENGINE.md).

Usage:
    python -m app.ingestion.processor DATA --wipe

Scans the folder structure, auto-detects file types, parses each file,
semantic-chunks the text, embeds with Gemini, and uploads to Qdrant Cloud.

Folder names are mapped to metadata automatically:
    DATA/true_data/   →  {"data_type": "true",   "folder": "true_data"}
    DATA/noisy_data/  →  {"data_type": "noisy",  "folder": "noisy_data"}
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.config import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.parsers import parse_file
from app.services.retrieval.embedding import embed_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("ingestion")


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_CLUSTER_ENDPOINT,
        api_key=settings.QDRANT_API_KEY,
    )


def ensure_collection(client: QdrantClient, wipe: bool) -> None:
    """Create the collection if missing (or wipe + recreate when --wipe)."""
    exists = client.collection_exists(settings.QDRANT_COLLECTION)
    if exists and wipe:
        log.info("🗑️  --wipe: deleting collection '%s'", settings.QDRANT_COLLECTION)
        client.delete_collection(settings.QDRANT_COLLECTION)
        exists = False
    if not exists:
        log.info("🆕 Creating collection '%s' (dim=%d, distance=COSINE)",
                 settings.QDRANT_COLLECTION, settings.EMBEDDING_DIM)
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )


def folder_metadata(rel_path: Path) -> dict:
    """Map folder structure to metadata. Top-level folder = data_type."""
    parts = rel_path.parts
    if len(parts) >= 2:
        top = parts[-2]
        if top.endswith("_data") or top.endswith("_datas"):
            return {"data_type": top.replace("_data", "").replace("_datas", ""), "folder": top}
    return {"data_type": "generic", "folder": parts[-2] if len(parts) >= 2 else ""}


def collect_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".html", ".htm", ".txt", ".md", ".docx", ".pptx"}:
            files.append(path)
    return sorted(files)


def run(data_dir: str, wipe: bool = False) -> None:
    root = Path(data_dir)
    if not root.is_dir():
        log.error("Data directory not found: %s", root)
        sys.exit(1)

    client = get_client()
    ensure_collection(client, wipe)

    files = collect_files(root)
    if not files:
        log.warning("No supported files found under %s", root)
        return

    log.info("📄 Found %d file(s) to ingest", len(files))
    point_id = 0
    total_start = time.perf_counter()

    for file in files:
        rel = file.relative_to(root)
        text = parse_file(file)
        if not text.strip():
            log.warning("  ⚠️  %s — no text extracted, skipping", rel)
            continue

        chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE)
        meta = folder_metadata(rel)
        log.info("  📄 %s → %d chunk(s) [%s]", rel, len(chunks), meta["data_type"])

        # Embed + upload in batches (avoids giant single requests)
        batch = 32
        for i in range(0, len(chunks), batch):
            batch_chunks = chunks[i : i + batch]
            vectors = embed_texts(batch_chunks)
            payloads = [
                {
                    "text": c,
                    "source": str(rel),
                    "chunk_index": point_id + j,
                    **meta,
                }
                for j, c in enumerate(batch_chunks)
            ]
            client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[
                    models.PointStruct(id=point_id + j, vector=v, payload=p)
                    for j, (v, p) in enumerate(zip(vectors, payloads))
                ],
            )
            point_id += len(batch_chunks)
            log.info("      ↑ uploaded %d chunk(s) (total %d)", len(batch_chunks), point_id)

    elapsed = time.perf_counter() - total_start
    log.info("✅ Ingestion complete: %d chunks in %.1fs → collection '%s'",
             point_id, elapsed, settings.QDRANT_COLLECTION)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal RAG ingestion engine")
    parser.add_argument("data_dir", help="Folder containing source documents (e.g. DATA)")
    parser.add_argument("--wipe", action="store_true", help="Recreate the collection from scratch")
    args = parser.parse_args()
    run(args.data_dir, wipe=args.wipe)


if __name__ == "__main__":
    main()
