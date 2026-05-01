"""
embedding/embedder.py
─────────────────────
SentenceTransformer embedding logic.

Responsibilities:
  • Load the all-MiniLM-L6-v2 model once and cache it in memory.
  • Embed a list of text chunks in a single batch (efficient).
  • Embed a single query string at inference time.
  • Return numpy float32 arrays that FAISS can consume directly.
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model and cache it for the process lifetime.

    lru_cache(maxsize=1) ensures the model is loaded only once no matter
    how many times _load_model() is called — avoids repeated disk I/O.

    Returns:
        Loaded SentenceTransformer model.
    """
    logger.info("Loading embedding model: %s ...", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info("✅ Embedding model loaded.")
    return model


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Convert a list of chunk dicts into a 2-D numpy embedding matrix.

    All chunks are embedded in a single batch call for efficiency.
    The resulting matrix has shape (num_chunks, EMBEDDING_DIMENSION).

    Args:
        chunks: List of chunk dicts — each must have a "text" key.

    Returns:
        Float32 numpy array of shape (N, 384).

    Raises:
        ValueError: If the chunks list is empty.
    """
    if not chunks:
        raise ValueError("embed_chunks() received an empty list.")

    model = _load_model()
    texts = [chunk["text"] for chunk in chunks]

    logger.info("Embedding %d chunk(s) in batch ...", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=64,           # process up to 64 texts at once
        show_progress_bar=True,  # shows tqdm bar in terminal during sync
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-norm → cosine similarity via dot product
    )

    # Ensure float32 — FAISS requires it
    embeddings = embeddings.astype(np.float32)

    logger.info(
        "✅ Batch embedding done. Shape: %s, dtype: %s",
        embeddings.shape,
        embeddings.dtype,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query string into a 1-D vector ready for FAISS search.

    Args:
        query: The user's natural-language question.

    Returns:
        Float32 numpy array of shape (384,).

    Raises:
        ValueError: If the query string is empty.
    """
    if not query.strip():
        raise ValueError("embed_query() received an empty query string.")

    model = _load_model()

    embedding = model.encode(
        [query],                     # encode always wants a list
        convert_to_numpy=True,
        normalize_embeddings=True,   # must match how chunks were embedded
    )

    return embedding[0].astype(np.float32)  # shape: (384,)
