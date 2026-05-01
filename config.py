"""
config.py
─────────
Central configuration module.

Reads all settings from environment variables (via .env file).
Every other module imports from here — never reads .env directly.
This makes it trivial to swap values or move to a secrets manager later.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level up from this file if needed)
load_dotenv()

# ─── Google Drive ─────────────────────────────────────────────────────────────

# Absolute or relative path to the Service Account JSON key file
GOOGLE_SERVICE_ACCOUNT_JSON: str = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_JSON", "credentials/service_account.json"
)

# The Drive folder whose contents will be indexed
GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

# MIME types we care about — everything else is skipped
SUPPORTED_MIME_TYPES: list[str] = [
    "application/pdf",                                          # PDF files
    "application/vnd.google-apps.document",                    # Google Docs
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document",                              # .docx files
    "text/plain",                                              # .txt files
]

# ─── Groq LLM ─────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Embedding ────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION: int = 384   # fixed output size for all-MiniLM-L6-v2

# ─── FAISS / Storage ──────────────────────────────────────────────────────────

# Where the FAISS binary index is saved on disk
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")

# Where chunk metadata (doc_id, file_name, text, etc.) is stored as JSON
METADATA_PATH: str = os.getenv("METADATA_PATH", "metadata.json")

# ─── RAG Settings ─────────────────────────────────────────────────────────────

# How many chunks to retrieve from FAISS for each user query
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# ─── Chunking Settings ────────────────────────────────────────────────────────

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))      # words per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50")) # overlap in words

# ─── Validation ───────────────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Call this at startup to fail fast if critical config is missing.
    Raises ValueError with a descriptive message for each missing value.
    """
    errors: list[str] = []

    if not GOOGLE_DRIVE_FOLDER_ID:
        errors.append("GOOGLE_DRIVE_FOLDER_ID is not set in .env")

    if not Path(GOOGLE_SERVICE_ACCOUNT_JSON).exists():
        errors.append(
            f"Service account file not found at: {GOOGLE_SERVICE_ACCOUNT_JSON}"
        )

    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set in .env")

    if errors:
        raise ValueError(
            "Configuration errors found:\n" + "\n".join(f"  • {e}" for e in errors)
        )
