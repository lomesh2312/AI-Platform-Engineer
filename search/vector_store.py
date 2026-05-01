"""
search/vector_store.py
──────────────────────
FAISS index management: build, save, load, and search.

Responsibilities:
  • Build a fresh FAISS IndexFlatL2 index from embeddings + chunk metadata.
  • Persist the index (binary) and metadata (JSON) to disk.
  • Load an existing index and metadata from disk.
  • Search the index with a query embedding and return ranked chunk dicts.

Design decisions:
  • IndexFlatL2 — brute-force exact search. Simple and reliable for a demo
    with < 100k vectors. Swap to IndexIVFFlat / HNSW later for scale.
  • Metadata is stored separately in JSON so it's human-readable and easy
    to inspect / migrate to a database (e.g. PostgreSQL) later.
"""

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config

logger = logging.getLogger(__name__)


# ─── Build & Save ─────────────────────────────────────────────────────────────

def build_and_save_index(
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    index_path: str = config.FAISS_INDEX_PATH,
    metadata_path: str = config.METADATA_PATH,
) -> None:
    """
    Build a FAISS index from embeddings and save both index + metadata to disk.

    On every full sync we clear the old index and rebuild from scratch.
    This keeps the index consistent with whatever is currently on Drive.

    Args:
        embeddings:    Float32 array of shape (N, EMBEDDING_DIMENSION).
        chunks:        List of chunk dicts (same order as embeddings rows).
        index_path:    File path for the FAISS binary index.
        metadata_path: File path for the JSON metadata file.

    Raises:
        ValueError: If embeddings and chunks have different lengths.
    """
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunks."
        )

    if len(embeddings) == 0:
        logger.warning("No embeddings provided — nothing to index.")
        return

    # ── Build the FAISS index ─────────────────────────────────────────────────
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)   # Euclidean distance, exact search
    index.add(embeddings)                  # bulk add all vectors at once

    logger.info(
        "✅ FAISS index built. Vectors: %d, Dimension: %d",
        index.ntotal,
        dimension,
    )

    # ── Persist the FAISS index ───────────────────────────────────────────────
    faiss.write_index(index, index_path)
    logger.info("✅ FAISS index saved → %s", index_path)

    # ── Persist chunk metadata as JSON ────────────────────────────────────────
    # Strip the "text" field from stored metadata to keep the JSON small;
    # the text is only needed at query time and we store it anyway for context.
    # Actually we DO keep it because we need the text to build the LLM prompt.
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("✅ Metadata saved → %s (%d entries)", metadata_path, len(chunks))


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_index(
    index_path: str = config.FAISS_INDEX_PATH,
    metadata_path: str = config.METADATA_PATH,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Load an existing FAISS index and its metadata from disk.

    Args:
        index_path:    Path to the FAISS binary index file.
        metadata_path: Path to the JSON metadata file.

    Returns:
        A tuple of (faiss_index, list_of_chunk_dicts).

    Raises:
        FileNotFoundError: If either file doesn't exist yet.
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Run POST /api/v1/sync-drive first."
        )
    if not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Metadata file not found at '{metadata_path}'. "
            "Run POST /api/v1/sync-drive first."
        )

    index = faiss.read_index(index_path)
    logger.info("✅ FAISS index loaded from '%s' (%d vectors).", index_path, index.ntotal)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info("✅ Metadata loaded from '%s' (%d chunks).", metadata_path, len(metadata))

    return index, metadata


# ─── Search ───────────────────────────────────────────────────────────────────

def search(
    query_embedding: np.ndarray,
    top_k: int = config.TOP_K_RESULTS,
    index_path: str = config.FAISS_INDEX_PATH,
    metadata_path: str = config.METADATA_PATH,
) -> list[dict[str, Any]]:
    """
    Search the FAISS index and return the top-k most relevant chunks.

    Args:
        query_embedding: Float32 1-D array of shape (EMBEDDING_DIMENSION,).
        top_k:           How many results to return.
        index_path:      Path to FAISS index file.
        metadata_path:   Path to metadata JSON file.

    Returns:
        List of up to `top_k` chunk dicts, ordered by similarity (best first).
        Each dict has all original metadata fields plus:
          "score": float  — L2 distance (lower = more similar)

    Raises:
        FileNotFoundError: If the index hasn't been built yet.
    """
    index, metadata = load_index(index_path, metadata_path)

    # FAISS expects a 2-D array: (num_queries, dimension)
    query_2d = query_embedding.reshape(1, -1).astype(np.float32)

    # Clamp top_k to the number of vectors in the index
    actual_k = min(top_k, index.ntotal)
    distances, indices = index.search(query_2d, actual_k)

    results: list[dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 when fewer than k results exist
            continue
        chunk = dict(metadata[idx])  # copy so we don't mutate the cached list
        chunk["score"] = float(dist)
        results.append(chunk)

    logger.info("Search returned %d result(s) for top_k=%d.", len(results), top_k)
    return results


def index_exists(
    index_path: str = config.FAISS_INDEX_PATH,
    metadata_path: str = config.METADATA_PATH,
) -> bool:
    """
    Quick check: do both the index and metadata files exist on disk?

    Used by the /ask endpoint to return a helpful error before attempting
    a search on a non-existent index.
    """
    return Path(index_path).exists() and Path(metadata_path).exists()
