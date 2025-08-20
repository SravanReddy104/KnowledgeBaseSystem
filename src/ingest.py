import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils import load_files, chunk_documents, ensure_dir
from config import CHROMA_DIR, GEMINI_EMBED_MODEL, GOOGLE_API_KEY


def _embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
        transport="rest",
    )


def ingest_files(
    file_paths: List[str],
    persist_dir: Optional[str] = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> str:
    """
    Ingest files into Chroma. Returns the persistence directory used.
    """
    persist = persist_dir or CHROMA_DIR
    ensure_dir(persist)

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")

    docs = load_files(file_paths)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = _embeddings()

    # Create/update persistent collection
    vs = Chroma(embedding_function=embeddings, persist_directory=persist)
    vs.add_documents(chunks)
    vs.persist()
    return persist


def get_vectorstore(persist_dir: Optional[str] = None) -> Chroma:
    persist = persist_dir or CHROMA_DIR
    if not os.path.isdir(persist):
        raise FileNotFoundError(f"Chroma directory not found at {persist}. Ingest documents first.")
    embeddings = _embeddings()
    return Chroma(embedding_function=embeddings, persist_directory=persist)


def get_retriever(persist_dir: Optional[str] = None):
    vs = get_vectorstore(persist_dir)
    return vs.as_retriever(search_kwargs={"k": 5})
