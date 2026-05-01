# 🧠 Personal RAG System over Google Drive

A production-ready **Retrieval-Augmented Generation (RAG)** API that lets you ask questions in plain English and get answers grounded in your own Google Drive documents — no hallucinations, full source citations.

---

## 🏗️ Architecture

```
Google Drive Folder
        │
        ▼
 connectors/gdrive.py        ← Auth via Service Account, list + download files
        │
        ▼
 processing/chunker.py       ← Extract text (PDF/DOCX/TXT), clean, split into chunks
        │
        ▼
 embedding/embedder.py       ← Generate 384-dim embeddings via HuggingFace Inference API
        │
        ▼
 search/vector_store.py      ← Store in FAISS IndexFlatL2 (faiss_index.bin + metadata.json)
        │
        ▼
 api/routes.py               ← FastAPI endpoints: /sync-drive  and  /ask
        │
        ▼
    Groq LLM                 ← llama3-70b-8192 answers using retrieved context only
```

---

## 📁 Project Structure

```
project/
├── connectors/
│   └── gdrive.py           # Google Drive fetching logic
├── processing/
│   └── chunker.py          # Text extraction + cleaning + chunking
├── embedding/
│   └── embedder.py         # SentenceTransformer embedding logic
├── search/
│   └── vector_store.py     # FAISS index: save, load, search
├── api/
│   └── routes.py           # FastAPI route definitions
├── credentials/
│   └── service_account.json  ← Place your GCP key here (git-ignored)
├── main.py                 # App entry point
├── config.py               # All config / env variable loading
├── requirements.txt        # All dependencies
├── .env                    # Your secrets (git-ignored)
└── .env.example            # Template — copy to .env and fill in
```

---

## ⚙️ Setup Instructions

### 1. Clone and Create Virtual Environment

```bash
git clone <your-repo-url>
cd project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First install takes ~3–5 minutes (PyTorch + FAISS are large).

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Path to your GCP Service Account JSON key |
| `GOOGLE_DRIVE_FOLDER_ID` | Folder ID from your Drive URL |
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama3-70b-8192` (or `llama3-8b-8192` for speed) |

### 4. Place Service Account Key

Copy your downloaded `service_account.json` into `credentials/`:

```bash
cp ~/Downloads/your-key.json credentials/service_account.json
```

> 🔒 This file is in `.gitignore` — it will never be committed.

### 5. Share Your Drive Folder

In Google Drive, right-click your test folder → **Share** → paste your service account email → give **Viewer** access.

Your service account email looks like:
```
your-name@your-project.iam.gserviceaccount.com
```

### 6. Start the Backend Server

```bash
python main.py
```

The API will be live at: **http://localhost:8000**

### 7. Start the Streamlit UI

In a **new terminal tab**, run:

```bash
source venv/bin/activate
streamlit run app_ui.py
```

The UI will be available at: **http://localhost:8501**


- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## 🚀 Using the API

### Step 1 — Sync Your Drive

This downloads, processes, and indexes all documents from your Drive folder.

> ℹ️ **Note:** Embeddings are generated via the **HuggingFace Inference API**. The first request may take a moment while the API wakes up. Make sure `HF_API_TOKEN` is set in your environment variables.

```bash
curl -X POST http://localhost:8000/api/v1/sync-drive
```

**Example response:**
```json
{
  "message": "Drive sync completed successfully.",
  "documents_synced": 3,
  "total_chunks": 47
}
```

---

### Step 2 — Ask Questions

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

**Example response:**
```json
{
  "answer": "According to the refund policy document, customers can request a full refund within 30 days of purchase by contacting support@company.com.",
  "sources": ["refund_policy.pdf"]
}
```

---

### Sample Queries to Test

```bash
# Policy questions
curl -X POST http://localhost:8000/api/v1/ask \
  -d '{"query": "What are the leave policy rules?"}' \
  -H "Content-Type: application/json"

# Compliance
curl -X POST http://localhost:8000/api/v1/ask \
  -d '{"query": "What does the SOP say about data handling?"}' \
  -H "Content-Type: application/json"

# Out-of-scope (should say I don't know)
curl -X POST http://localhost:8000/api/v1/ask \
  -d '{"query": "What is the weather in Tokyo?"}' \
  -H "Content-Type: application/json"
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Google Drive | google-api-python-client (Service Account) |
| PDF Extraction | PyMuPDF (fitz) |
| DOCX Extraction | python-docx |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS `IndexFlatL2` |
| LLM | Groq `llama3-70b-8192` |
| Config | python-dotenv |

---

## 🔄 Swapping Components

The modular design makes swapping easy:

**Replace FAISS with OpenSearch:**
> Implement the same `build_and_save_index()` / `search()` interface in a new `search/opensearch_store.py` and update the import in `routes.py`.

**Replace Groq with OpenAI:**
> Change `groq_client` in `routes.py` to `openai.OpenAI(api_key=...)` and update the `.chat.completions.create()` call signature (it's identical).

---

## 🛡️ Error Handling

| Scenario | Behavior |
|---|---|
| File can't be parsed | Logged and skipped; sync continues |
| Index not built yet | `/ask` returns HTTP 503 with helpful message |
| No relevant chunks found | Returns `"I don't know."` |
| Groq API failure | Returns HTTP 502 with error detail |
| Missing config | Server refuses to start with clear error message |
