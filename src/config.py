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
UPLOAD_DIR = _get_secret("UPLOAD_DIR", "./uploads")

# Models
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = _get_secret("GEMINI_EMBED_MODEL", "models/embedding-001")

# Keys
GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = _get_secret("PINECONE_API_KEY")

# Pinecone
PINECONE_INDEX = _get_secret("PINECONE_INDEX", "kb-index")
# For serverless index creation (defaults suitable for free tier)
PINECONE_CLOUD = _get_secret("PINECONE_CLOUD", "aws")
PINECONE_REGION = _get_secret("PINECONE_REGION", "us-east-1")

# App
APP_TITLE = _get_secret("APP_TITLE", "Knowledge Base Agent")
APP_DESCRIPTION = _get_secret(
    "APP_DESCRIPTION",
    "Upload docs, embed to Pinecone, and chat using Gemini (LangChain + LangGraph)",
)
