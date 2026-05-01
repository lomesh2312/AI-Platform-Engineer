"""
main.py
───────
Minimal, cloud-safe FastAPI entry point.
"""

import logging
from fastapi import FastAPI
from api.routes import router
import config

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Validate config (non-crashing)
config.validate_config()

app = FastAPI(
    title="DriveRAG",
    description="Resilient RAG System for Google Drive"
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Health"])
def root_check():
    """Simple root health check for Render/Uptime services."""
    return {
        "status": "DriveRAG is running",
        "version": "1.0.1"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    import os
    # Read port from environment (Render injects $PORT)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
