"""📄 Lightweight parser for golden-dataset source documents (DOCS/11 §Part 1).

We reuse the ingestion parsers directly (python-docx, python-pptx,
BeautifulSoup, pypdf) instead of the `unstructured` library so the golden
dataset builder stays fast and dependency-light.
"""
from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers import parse_file


def parse_document(path: Path) -> str:
    """Return the full extracted text of a source document."""
    text = parse_file(path)
    return text.strip()


# The 5 golden sources used by DOCS/11 (plus architecture.pptx exists too).
TRUE_DATA_FILES = [
    "parallel_work_queue.txt",
    "pods_autoscale.html",
    "job_management.html",
    "cronjobs.docx",
    "monitor_job.docx",
]


def true_data_paths(data_dir: Path) -> list[Path]:
    """Resolve the golden source files inside a DATA/true_data directory."""
    paths = []
    for name in TRUE_DATA_FILES:
        p = data_dir / name
        if p.exists():
            paths.append(p)
    return paths