"""
api/routes.py
─────────────
FastAPI route definitions for the Personal RAG System.

Endpoints:
  POST /sync-drive  — Fetch, process, embed, and index all Drive documents.
  POST /ask         — Answer a question using RAG over the indexed documents.

Each route is a thin orchestration layer:
  • It calls the appropriate modules (connector → chunker → embedder → store).
  • It never contains business logic itself.
  • It returns clean JSON responses with proper HTTP status codes.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from groq import Groq

import config
from connectors.gdrive import iter_drive_files
from processing.chunker import process_file
from embedding.embedder import embed_chunks, embed_query
from search.vector_store import build_and_save_index, search, index_exists

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialise the Groq client once (reused across requests)
groq_client = Groq(api_key=config.GROQ_API_KEY)


# ─── Request / Response Models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class SyncResponse(BaseModel):
    message: str
    documents_synced: int
    total_chunks: int


# ─── POST /sync-drive ─────────────────────────────────────────────────────────

@router.post("/sync-drive", response_model=SyncResponse, tags=["RAG"])
async def sync_drive() -> dict[str, Any]:
    """
    Sync all documents from Google Drive into the local FAISS index.

    Pipeline:
      1. List + download all supported files from the configured Drive folder.
      2. Extract and clean text from each file.
      3. Split text into overlapping chunks.
      4. Embed all chunks in a single batch.
      5. Build a fresh FAISS index and save to disk.

    Returns:
        Number of documents synced and total chunks indexed.
    """
    logger.info("━━━ Starting Drive sync ━━━")

    all_chunks: list[dict] = []
    documents_synced = 0

    try:
        for file_meta, raw_bytes in iter_drive_files(config.GOOGLE_DRIVE_FOLDER_ID):
            logger.info("Processing: %s", file_meta["name"])
            chunks = process_file(file_meta, raw_bytes)

            if not chunks:
                logger.warning("  No chunks extracted from '%s'. Skipping.", file_meta["name"])
                continue

            all_chunks.extend(chunks)
            documents_synced += 1
            logger.info("  ✅ %d chunk(s) from '%s'", len(chunks), file_meta["name"])

    except Exception as exc:
        logger.error("Drive sync failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Drive sync error: {str(exc)}")

    if not all_chunks:
        return {
            "message": "No documents found or no text could be extracted.",
            "documents_synced": 0,
            "total_chunks": 0,
        }

    # ── Embed all chunks in one batch ─────────────────────────────────────────
    logger.info("Embedding %d total chunks ...", len(all_chunks))
    try:
        embeddings = embed_chunks(all_chunks)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(exc)}")

    # ── Build and persist the FAISS index ─────────────────────────────────────
    try:
        build_and_save_index(embeddings, all_chunks)
    except Exception as exc:
        logger.error("FAISS index build failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Index build error: {str(exc)}")

    logger.info(
        "━━━ Sync complete: %d docs, %d chunks ━━━",
        documents_synced,
        len(all_chunks),
    )

    return {
        "message": "Drive sync completed successfully.",
        "documents_synced": documents_synced,
        "total_chunks": len(all_chunks),
    }


# ─── POST /ask ────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest) -> dict[str, Any]:
    """
    Answer a user question using RAG over the indexed Drive documents.

    Pipeline:
      1. Validate that the FAISS index exists.
      2. Embed the query.
      3. Search FAISS for the top-k most relevant chunks.
      4. Build a context-grounded prompt.
      5. Call Groq LLM API.
      6. Return the answer + unique source file names.

    Args:
        request: JSON body with a "query" field.

    Returns:
        { "answer": "...", "sources": ["file1.pdf", "file2.docx"] }
    """
    logger.info("━━━ /ask received: '%s' ━━━", request.query)

    # ── Guard: index must exist ───────────────────────────────────────────────
    if not index_exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge base is empty. "
                "Call POST /api/v1/sync-drive first to index your documents."
            ),
        )

    # ── Embed the query ───────────────────────────────────────────────────────
    try:
        query_vec = embed_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(exc)}")

    # ── Search FAISS ──────────────────────────────────────────────────────────
    try:
        top_chunks = search(query_vec, top_k=config.TOP_K_RESULTS)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)}")

    # ── Handle no results ─────────────────────────────────────────────────────
    if not top_chunks:
        return {"answer": "I don't know.", "sources": []}

    # ── Build the RAG prompt ──────────────────────────────────────────────────
    context_blocks = "\n\n---\n\n".join(
        f"[Source: {c['file_name']}]\n{c['text']}" for c in top_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not present in the context, respond with exactly: "I don't know."

Context:
{context_blocks}

Question: {request.query}
Answer:"""

    # ── Call Groq LLM ─────────────────────────────────────────────────────────
    try:
        logger.info("Calling Groq model: %s ...", config.GROQ_MODEL)
        completion = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,       # low temp → more factual, less creative
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(exc)}")

    # ── Collect unique source file names ──────────────────────────────────────
    sources = list(dict.fromkeys(c["file_name"] for c in top_chunks))

    logger.info("━━━ /ask response ready. Sources: %s ━━━", sources)
    return {"answer": answer, "sources": sources}
