"""
processing/chunker.py
─────────────────────
Text extraction + cleaning + chunking pipeline.

Responsibilities:
  • Extract raw text from PDF (via PyMuPDF), .docx (via python-docx),
    and plain text (.txt / Google Docs export) files.
  • Clean the extracted text (collapse whitespace, remove junk).
  • Split cleaned text into overlapping word-based chunks.
  • Attach metadata to every chunk so we can trace answers back to sources.
"""

import io
import logging
import re
from typing import Any

import fitz  # PyMuPDF
from docx import Document

import config

logger = logging.getLogger(__name__)


# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(raw_bytes: bytes) -> str:
    """
    Extract all text from a PDF file.

    Uses PyMuPDF (fitz) which is fast and handles most PDF variants.
    Falls back to an empty string if extraction fails.

    Args:
        raw_bytes: Raw PDF file content.

    Returns:
        Concatenated text from all pages.
    """
    try:
        pdf = fitz.open(stream=raw_bytes, filetype="pdf")
        pages_text = [page.get_text() for page in pdf]
        full_text = "\n".join(pages_text)
        logger.info("  PDF extraction: %d pages, %d chars", len(pages_text), len(full_text))
        return full_text
    except Exception as exc:
        logger.error("  ❌ PDF extraction failed: %s", exc)
        return ""


def extract_text_from_docx(raw_bytes: bytes) -> str:
    """
    Extract all paragraph text from a .docx file.

    Args:
        raw_bytes: Raw .docx file content.

    Returns:
        Newline-joined paragraph text.
    """
    try:
        doc = Document(io.BytesIO(raw_bytes))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs)
        logger.info("  DOCX extraction: %d paragraphs, %d chars", len(paragraphs), len(full_text))
        return full_text
    except Exception as exc:
        logger.error("  ❌ DOCX extraction failed: %s", exc)
        return ""


def extract_text_from_txt(raw_bytes: bytes) -> str:
    """
    Decode plain text bytes (handles UTF-8 and Latin-1 fallback).

    Args:
        raw_bytes: Raw text file content.

    Returns:
        Decoded string.
    """
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def extract_text(file_meta: dict, raw_bytes: bytes) -> str:
    """
    Route a file to the correct extractor based on its MIME type.

    Args:
        file_meta: Drive file metadata (must have "mimeType" and "name").
        raw_bytes: Raw downloaded file bytes.

    Returns:
        Extracted text string, or "" if unsupported / failed.
    """
    mime: str = file_meta.get("mimeType", "")
    name: str = file_meta.get("name", "unknown")

    logger.info("Extracting text from '%s' (mime=%s) ...", name, mime)

    if mime == "application/pdf":
        return extract_text_from_pdf(raw_bytes)

    elif mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        return extract_text_from_docx(raw_bytes)

    elif mime in ("text/plain", "application/vnd.google-apps.document"):
        # Google Docs are already exported as plain text by the connector
        return extract_text_from_txt(raw_bytes)

    else:
        logger.warning("  Unsupported MIME type '%s' for file '%s'. Skipping.", mime, name)
        return ""


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw extracted text for embedding quality.

    Steps:
      1. Replace form-feeds and non-breaking spaces with regular spaces.
      2. Collapse multiple blank lines into a single blank line.
      3. Strip leading/trailing whitespace per line.
      4. Remove lines that are only punctuation or numbers (page numbers, etc.).
      5. Final strip.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    # Normalise whitespace characters
    text = text.replace("\f", "\n").replace("\xa0", " ")

    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip each line, drop lines that are pure punctuation / page numbers
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and not re.fullmatch(r"[\d\W]+", line):
            lines.append(line)

    return "\n".join(lines).strip()


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    file_meta: dict,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping word-based chunks with metadata.

    Strategy:
      • Split text into individual words.
      • Slide a window of `chunk_size` words with `overlap` words of stride.
      • Each chunk is stored as a dict with the text and source metadata.

    Args:
        text:       Cleaned document text.
        file_meta:  Drive file metadata dict (id, name, mimeType).
        chunk_size: Number of words per chunk (default from config).
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of chunk dicts:
        {
            "text":        str,   # the chunk's raw text
            "doc_id":      str,   # Drive file ID
            "file_name":   str,   # original file name
            "source":      str,   # always "gdrive"
            "chunk_index": int,   # 0-based position in the document
        }
    """
    if not text.strip():
        logger.warning("  Empty text for '%s' — skipping chunking.", file_meta.get("name"))
        return []

    words = text.split()
    stride = chunk_size - overlap  # how many words to advance each step

    if stride <= 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})."
        )

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "text": chunk_text_str,
                "doc_id": file_meta.get("id", "unknown"),
                "file_name": file_meta.get("name", "unknown"),
                "source": "gdrive",
                "chunk_index": chunk_index,
            }
        )

        chunk_index += 1
        start += stride

        # Stop if the remaining words are fewer than the overlap
        # (avoids creating a near-duplicate tiny last chunk)
        if start >= len(words):
            break

    logger.info(
        "  '%s' → %d chunks (chunk_size=%d, overlap=%d)",
        file_meta.get("name"),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


# ─── High-Level Pipeline Step ─────────────────────────────────────────────────

def process_file(file_meta: dict, raw_bytes: bytes) -> list[dict[str, Any]]:
    """
    Full processing pipeline for a single Drive file:
      extract → clean → chunk.

    Args:
        file_meta: Drive file metadata dict.
        raw_bytes: Raw file bytes from Drive.

    Returns:
        List of chunk dicts ready for embedding.
        Returns an empty list if extraction yields nothing.
    """
    raw_text = extract_text(file_meta, raw_bytes)
    if not raw_text:
        return []

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, file_meta)
    return chunks
