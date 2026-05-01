"""
embedding/embedder.py
─────────────────────
HuggingFace Inference API embedding logic (Zero Memory).

Responsibilities:
  • Call HF Inference API to generate 384-dim embeddings.
  • No local model loading = tiny RAM footprint (~150MB total).
  • Uses all-MiniLM-L6-v2 via API for perfect FAISS compatibility.
"""

import logging
import requests
import numpy as np
import time
from typing import List

import config

logger = logging.getLogger(__name__)

# Correct HF Inference API URL structure
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

def _call_hf_api(texts: List[str]) -> np.ndarray:
    """
    Core helper to call HuggingFace Inference API.
    Handles 'wait_for_model' to ensure the API is ready.
    """
    headers = {}
    if config.HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_API_TOKEN}"
    
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True}
    }
    
    logger.info("Calling HF Inference API for %d texts...", len(texts))
    
    # Simple retry loop for 'loading' states or temporary issues
    for attempt in range(3):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
            
            # If the model is still loading, wait and retry
            if response.status_code == 503:
                logger.warning(f"HF Model is loading... attempt {attempt+1}/3. Waiting 10s.")
                time.sleep(10)
                continue
                
            response.raise_for_status()
            embeddings = response.json()
            return np.array(embeddings).astype(np.float32)
            
        except Exception as e:
            if attempt == 2:
                logger.error(f"❌ HuggingFace API Error: {str(e)}")
                raise
            time.sleep(2)

def embed_chunks(chunks: List[dict]) -> np.ndarray:
    """
    Convert a list of chunk dicts into a 2-D numpy embedding matrix.
    """
    if not chunks:
        raise ValueError("embed_chunks() received an empty list.")

    texts = [chunk["text"] for chunk in chunks]
    
    # HF Inference API handles batching, but for very large syncs 
    # we process in chunks of 50 to avoid API timeouts
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_vecs = _call_hf_api(batch_texts)
        all_embeddings.append(batch_vecs)
        
        # Tiny sleep to avoid hitting free-tier rate limits
        if len(texts) > batch_size:
            time.sleep(0.5)

    final_matrix = np.vstack(all_embeddings)
    logger.info("✅ Batch embedding done via API. Shape: %s", final_matrix.shape)
    return final_matrix

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query string.
    """
    if not query.strip():
        raise ValueError("embed_query() received an empty query string.")

    # Returns shape (384,)
    embedding = _call_hf_api([query])[0]
    return embedding
