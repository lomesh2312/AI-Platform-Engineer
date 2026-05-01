"""
main.py
───────
Application entry point.

Responsibilities:
  1. Validate all required environment config at startup (fail fast).
  2. Create the FastAPI app instance.
  3. Register API routes from api/routes.py.
  4. Expose a health-check endpoint.
  5. Run the Uvicorn dev server when executed directly.
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import validate_config
from api.routes import router

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Startup Validation ───────────────────────────────────────────────────────

try:
    validate_config()
    logger.info("✅ Configuration validated successfully.")
except ValueError as exc:
    logger.error("❌ Startup aborted due to config errors:\n%s", exc)
    raise SystemExit(1) from exc

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Personal RAG System",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that answers questions "
        "using your own Google Drive documents as the knowledge base."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# Allow all origins for local development — tighten this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────

# Register all API routes under the /api/v1 prefix
app.include_router(router, prefix="/api/v1")


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Quick liveness check — returns 200 if the server is running."""
    return {"status": "ok", "service": "Personal RAG System"}


# ─── Dev Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # hot-reload on file changes during development
        log_level="info",
    )
