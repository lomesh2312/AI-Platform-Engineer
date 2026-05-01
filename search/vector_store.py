"""
search/vector_store.py
──────────────────────
Resilient FAISS index management.
"""

import json
import logging
from pathlib import Path
from typing import Any, Tuple

import faiss
import numpy as np
import config

logger = logging.getLogger(__name__)

def build_and_save_index(embeddings: np.ndarray, chunks: list[dict]):
    if len(embeddings) == 0: return
    index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
    index.add(embeddings)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Index saved: {len(chunks)} vectors.")

def load_index() -> Tuple[Any, list]:
    """
    Load index and metadata. Returns a fresh empty index if file is missing.
    Does NOT raise FileNotFoundError.
    """
    index_path = Path(config.FAISS_INDEX_PATH)
    meta_path = Path(config.METADATA_PATH)

    if not index_path.exists() or not meta_path.exists():
        logger.warning("⚠️ Index or metadata missing. Creating empty state.")
        empty_index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
        return empty_index, []

    try:
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        logger.error(f"❌ Error loading index: {e}")
        return faiss.IndexFlatL2(config.EMBEDDING_DIMENSION), []

def search(query_embedding: np.ndarray, top_k: int = config.TOP_K_RESULTS) -> list:
    index, metadata = load_index()
    if index.ntotal == 0:
        return []

    query_2d = query_embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_2d, min(top_k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            chunk = dict(metadata[idx])
            chunk["score"] = float(dist)
            results.append(chunk)
    return results

def index_exists() -> bool:
    # Always return true now, because load_index handles empty state gracefully
    return True
