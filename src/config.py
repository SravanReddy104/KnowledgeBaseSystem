import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def _get_secret(name: str, default: str | None = None):
    # Prefer Streamlit secrets if available (works locally with .streamlit/secrets.toml and on Cloud)
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets.get(name, default)
    except Exception:
        # st.secrets may not be available outside Streamlit runtime
        pass
    # Fallback to environment variables / .env
    return os.getenv(name, default)


# Directories
CHROMA_DIR = _get_secret("CHROMA_DIR", "./chroma_db")
UPLOAD_DIR = _get_secret("UPLOAD_DIR", "./uploads")

# Models
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = _get_secret("GEMINI_EMBED_MODEL", "models/embedding-001")

# Keys
GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")

# App
APP_TITLE = _get_secret("APP_TITLE", "Knowledge Base Agent")
APP_DESCRIPTION = _get_secret(
    "APP_DESCRIPTION",
    "Upload docs, embed to Chroma, and chat using Gemini (LangChain + LangGraph)",
)
