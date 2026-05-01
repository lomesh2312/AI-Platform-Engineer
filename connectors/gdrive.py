"""
connectors/gdrive.py
────────────────────
Cloud-resilient Google Drive connector.
"""

import io
import json
import logging
import os
from typing import Generator

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import config

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
MIME_EXPORT_MAP = {"application/vnd.google-apps.document": "text/plain"}

def _build_drive_service():
    """
    Authenticate via Service Account.
    Automatically detects if config is a file path or a raw JSON string.
    """
    val = config.GOOGLE_SERVICE_ACCOUNT_JSON
    
    try:
        # Check if it's a valid JSON string first (Render style)
        if val.strip().startswith("{"):
            info = json.loads(val)
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=SCOPES
            )
            logger.info("✅ Authenticated via Service Account JSON string.")
        # Otherwise treat as a file path (Local style)
        elif os.path.exists(val):
            credentials = service_account.Credentials.from_service_account_file(
                val, scopes=SCOPES
            )
            logger.info(f"✅ Authenticated via Service Account file: {val}")
        else:
            raise ValueError("No valid service account JSON or file found.")
            
    except Exception as e:
        logger.error(f"❌ Auth Failed: {str(e)}")
        raise

    return build("drive", "v3", credentials=credentials, cache_discovery=False)

def list_files(folder_id: str) -> list[dict]:
    if not folder_id:
        logger.error("No Folder ID provided.")
        return []
        
    service = _build_drive_service()
    mime_conditions = " or ".join([f"mimeType='{m}'" for m in config.SUPPORTED_MIME_TYPES])
    query = f"'{folder_id}' in parents and ({mime_conditions}) and trashed=false"

    results = []
    page_token = None
    while True:
        response = service.files().list(
            q=query, spaces="drive", fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token
        ).execute()
        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token: break
    return results

def download_file(file_meta: dict) -> bytes | None:
    service = _build_drive_service()
    file_id = file_meta["id"]
    mime_type = file_meta["mimeType"]
    buffer = io.BytesIO()

    try:
        if mime_type in MIME_EXPORT_MAP:
            request = service.files().export_media(fileId=file_id, mimeType=MIME_EXPORT_MAP[mime_type])
        else:
            request = service.files().get_media(fileId=file_id)
        
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Download failed for {file_meta['name']}: {e}")
        return None

def iter_drive_files(folder_id: str):
    for file_meta in list_files(folder_id):
        raw_bytes = download_file(file_meta)
        if raw_bytes: yield file_meta, raw_bytes
