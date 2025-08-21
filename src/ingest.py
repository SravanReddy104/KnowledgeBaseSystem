import os
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils import load_files, chunk_documents
from config import (
    GEMINI_EMBED_MODEL,
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_CLOUD,
    PINECONE_REGION,
)


def _embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
        transport="rest",
    )


def _ensure_pinecone_index(pc: Pinecone, index_name: str, dim: int = 768) -> None:
    existing = {i["name"] for i in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def ingest_files(
    file_paths: List[str],
    persist_dir: Optional[str] = None,  # kept for backward compatibility; unused
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> str:
    """
    Ingest files into Pinecone. Returns the index name used.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to your environment or secrets.")
    if not PINECONE_INDEX:
        raise RuntimeError("PINECONE_INDEX is empty. Set it in .env or Streamlit secrets (e.g., 'kb-index').")

    docs = load_files(file_paths)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = _embeddings()

    # Init Pinecone and ensure index exists (serverless/free tier)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_pinecone_index(pc, PINECONE_INDEX, dim=768)

    # Upsert documents
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
    )

    return PINECONE_INDEX


def get_vectorstore(persist_dir: Optional[str] = None) -> PineconeVectorStore:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to your environment or secrets.")
    if not PINECONE_INDEX:
        raise RuntimeError("PINECONE_INDEX is empty. Set it in .env or Streamlit secrets.")
    embeddings = _embeddings()
    return PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)


def get_retriever(persist_dir: Optional[str] = None):
    vs = get_vectorstore(persist_dir)
    return vs.as_retriever(search_kwargs={"k": 5})
