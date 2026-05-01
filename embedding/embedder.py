"""
embedding/embedder.py
─────────────────────
FastEmbed embedding logic (Lightweight, No PyTorch).

Responsibilities:
  • Load the BAAI/bge-small-en-v1.5 model once and cache it.
  • Embed a list of text chunks (efficient list generator).
  • Embed a single query string at inference time.
  • Return numpy float32 arrays (384-dim) for FAISS.
"""

import logging
from functools import lru_cache
import numpy as np
from fastembed import TextEmbedding

import config

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    """
    Load the FastEmbed model and cache it.
    Note: The first call will download the model files (~130MB).
    """
    logger.info("Loading FastEmbed model: BAAI/bge-small-en-v1.5 ...")
    # fastembed downloads models automatically to ~/.cache/fastembed/
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    logger.info("✅ FastEmbed model loaded.")
    return model

def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Convert a list of chunk dicts into a 2-D numpy embedding matrix.
    """
    if not chunks:
        raise ValueError("embed_chunks() received an empty list.")

    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]

    logger.info("Embedding %d chunk(s) via FastEmbed...", len(texts))
    
    # .embed returns a generator, we convert to list then numpy
    embeddings_gen = model.embed(texts)
    embeddings_list = list(embeddings_gen)
    embeddings = np.array(embeddings_list).astype(np.float32)

    logger.info("✅ Embedding done. Shape: %s", embeddings.shape)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query string.
    """
    if not query.strip():
        raise ValueError("embed_query() received an empty query string.")

    model = _get_model()
    
    # FastEmbed's .embed always expects a list
    embeddings_gen = model.embed([query])
    embedding = list(embeddings_gen)[0]
    
    return np.array(embedding).astype(np.float32)
