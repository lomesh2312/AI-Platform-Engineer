"""
connectors/gdrive.py
────────────────────
Google Drive connector using a Service Account (no OAuth needed).

Responsibilities:
  • Authenticate with Google Drive API via service account JSON key.
  • List all supported files (PDF, Google Docs, .docx, .txt) in a given folder.
  • Download each file's raw bytes into memory (no temp files on disk).
  • Export Google Docs as plain text (they have no direct binary download).
"""

import io
import logging
from typing import Generator

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import config

logger = logging.getLogger(__name__)

# Scopes required — read-only is sufficient for RAG
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Maps MIME types to how we'll handle them
MIME_EXPORT_MAP = {
    # Google Docs must be exported — we export as plain text
    "application/vnd.google-apps.document": "text/plain",
}


def _build_drive_service():
    """
    Authenticate and return a Google Drive API service client.
    Uses the service account JSON key defined in config.
    """
    credentials = service_account.Credentials.from_service_account_file(
        config.GOOGLE_SERVICE_ACCOUNT_JSON,
        scopes=SCOPES,
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    logger.info("✅ Google Drive service authenticated successfully.")
    return service


def list_files(folder_id: str) -> list[dict]:
    """
    List all supported files inside a Google Drive folder.

    Args:
        folder_id: The Drive folder ID to scan.

    Returns:
        A list of file metadata dicts, each containing:
          { "id": str, "name": str, "mimeType": str }
    """
    service = _build_drive_service()

    # Build a query that finds all files in the folder with supported MIME types
    mime_conditions = " or ".join(
        [f"mimeType='{m}'" for m in config.SUPPORTED_MIME_TYPES]
    )
    query = f"'{folder_id}' in parents and ({mime_conditions}) and trashed=false"

    results: list[dict] = []
    page_token: str | None = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            )
            .execute()
        )

        files = response.get("files", [])
        results.extend(files)
        logger.info("  Found %d file(s) in this page.", len(files))

        page_token = response.get("nextPageToken")
        if not page_token:
            break  # No more pages

    logger.info("✅ Total files found in folder: %d", len(results))
    return results


def download_file(file_meta: dict) -> bytes | None:
    """
    Download a single file from Google Drive into memory.

    Handles two cases:
      1. Regular binary files (PDF, .docx, .txt) — direct media download.
      2. Google Workspace files (Google Docs) — exported as plain text.

    Args:
        file_meta: Dict with keys "id", "name", "mimeType".

    Returns:
        Raw file bytes, or None if the download failed.
    """
    service = _build_drive_service()
    file_id: str = file_meta["id"]
    file_name: str = file_meta["name"]
    mime_type: str = file_meta["mimeType"]

    buffer = io.BytesIO()

    try:
        if mime_type in MIME_EXPORT_MAP:
            # ── Google Workspace file: must be exported ──────────────────────
            export_mime = MIME_EXPORT_MAP[mime_type]
            request = service.files().export_media(
                fileId=file_id, mimeType=export_mime
            )
            logger.info("  Exporting Google Doc '%s' as plain text...", file_name)
        else:
            # ── Regular binary file: direct download ─────────────────────────
            request = service.files().get_media(fileId=file_id)
            logger.info("  Downloading binary file '%s'...", file_name)

        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        logger.info("  ✅ Downloaded '%s' (%d bytes)", file_name, buffer.tell())
        return buffer.getvalue()

    except Exception as exc:
        logger.error("  ❌ Failed to download '%s': %s", file_name, exc)
        return None  # Caller decides what to do with None


def iter_drive_files(folder_id: str) -> Generator[tuple[dict, bytes], None, None]:
    """
    High-level generator: list all files then yield (metadata, raw_bytes) pairs.

    Skips files that fail to download so the rest of the pipeline continues.

    Args:
        folder_id: The Drive folder to read from.

    Yields:
        Tuples of (file_meta_dict, raw_bytes).
    """
    files = list_files(folder_id)

    for file_meta in files:
        raw_bytes = download_file(file_meta)
        if raw_bytes is None:
            logger.warning("  Skipping '%s' due to download failure.", file_meta["name"])
            continue
        yield file_meta, raw_bytes
