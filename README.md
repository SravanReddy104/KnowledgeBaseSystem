# Knowledge Base Agent (Streamlit + LangChain + LangGraph + Chroma + Gemini)

A modern Streamlit app to build a local knowledge base from your documents, store them in a Chroma vector DB, and chat with them using Google Gemini. Workflows are orchestrated with LangGraph.

## Features
- Upload multiple files: PDF, DOCX, PPTX, TXT, MD, HTML, CSV
- Chunk + embed with Google Gemini embeddings
- Store/retrieve with Chroma (local, persisted)
- LangGraph retrieval-augmented generation flow
- Streamed chat responses in a modern UI

## Prerequisites
- Python 3.10+
- A Google API key with access to Gemini models

## Quickstart
```bash
# 1) Create env
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env  # On Windows: copy .env.example .env
# Edit .env and add GOOGLE_API_KEY

# 4) Run the app
streamlit run src/app.py
```

## Notes
- Default Chroma dir: `./chroma_db` (can change via `.env` or UI)
- Default models: `gemini-1.5-flash` (chat) and `models/embedding-001` (embeddings)
- First ingestion will create the Chroma store; subsequent uploads add to it.

## Troubleshooting
- If you see API errors, ensure `GOOGLE_API_KEY` is set and valid.
- If embedding or Chroma issues occur, try deleting the `chroma_db/` directory and re-ingesting.
