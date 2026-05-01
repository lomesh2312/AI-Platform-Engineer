"""
config.py
─────────
Resilient configuration module for Cloud/Render deployment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists (local development)
load_dotenv()

# ─── Google Drive ─────────────────────────────────────────────────────────────

# This can be a file path OR the raw JSON string content
GOOGLE_SERVICE_ACCOUNT_JSON: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

SUPPORTED_MIME_TYPES: list[str] = [
    "application/pdf",
    "application/vnd.google-apps.document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
]

# ─── Groq LLM ─────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Embedding (HF API) ───────────────────────────────────────────────────────

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")

# ─── FAISS / Storage ──────────────────────────────────────────────────────────

FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
METADATA_PATH: str = os.getenv("METADATA_PATH", "metadata.json")

# ─── RAG Settings ─────────────────────────────────────────────────────────────

TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── Startup Validation ───────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Non-crashing validation. Just logs warnings instead of raising SystemExit.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not GOOGLE_DRIVE_FOLDER_ID:
        logger.warning("⚠️ GOOGLE_DRIVE_FOLDER_ID is missing.")
    if not GROQ_API_KEY:
        logger.warning("⚠️ GROQ_API_KEY is missing.")
    if not HF_API_TOKEN:
        logger.warning("⚠️ HF_API_TOKEN is missing. HF API calls may fail.")
