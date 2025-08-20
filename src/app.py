import os
import sys
import pathlib
import json
import streamlit as st
from typing import List, Tuple
import uuid

# Ensure local imports work when running: `streamlit run src/app.py`
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR  # since we keep modules in the same folder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_TITLE, APP_DESCRIPTION, CHROMA_DIR, GEMINI_MODEL
from ingest import ingest_files, get_retriever, get_vectorstore
from graph import build_rag_graph, run_rag
from utils import ensure_dir
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

# ---------- Sidebar Config ----------
with st.sidebar:
    st.header("⚙️ Settings")
    chroma_dir = st.text_input("Chroma directory", value=CHROMA_DIR)
    model_name = st.text_input("Gemini chat model", value=GEMINI_MODEL, help="e.g., gemini-1.5-flash or gemini-1.5-pro")
    stream_mode = st.toggle("Stream responses", value=True)
    st.caption("Default model: gemini-1.5-flash")

st.title(APP_TITLE)
st.caption(APP_DESCRIPTION)

# Ensure directories
ensure_dir(chroma_dir)
uploads_root = os.path.join(".", "uploads")
ensure_dir(upload_root := uploads_root)

# ---------- Simple JSON persistence for chats ----------
CHAT_STORE = os.path.join(".streamlit", "chats.json")
ensure_dir(os.path.dirname(CHAT_STORE))

def load_chats_from_disk():
    try:
        if os.path.isfile(CHAT_STORE):
            with open(CHAT_STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("chats", []), data.get("active_chat_id")
    except Exception:
        pass
    return [], None

def save_chats_to_disk():
    try:
        with open(CHAT_STORE, "w", encoding="utf-8") as f:
            json.dump({
                "chats": st.session_state.get("chats", []),
                "active_chat_id": st.session_state.get("active_chat_id"),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------- Upload & Ingest ----------
with st.expander("📤 Upload and Ingest Documents", expanded=True):
    uploaded = st.file_uploader(
        "Upload multiple files (PDF, DOCX, PPTX, TXT, MD, HTML, CSV)",
        type=["pdf", "docx", "pptx", "txt", "md", "html", "htm", "csv"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        chunk_size = st.number_input("Chunk size", value=1200, step=100, min_value=200, max_value=4000)
        chunk_overlap = st.number_input("Chunk overlap", value=200, step=50, min_value=0, max_value=1000)
    with col2:
        st.markdown("""
        - Larger chunks improve context but may slow retrieval.
        - Overlap helps preserve context across chunks.
        - Tune per document size and question style.
        """)

    ingest_btn = st.button("Ingest to Chroma", type="primary", use_container_width=True)

    if ingest_btn:
        if not uploaded:
            st.warning("Please upload at least one file.")
        else:
            file_paths = []
            for uf in uploaded:
                save_path = os.path.join(upload_root, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                file_paths.append(save_path)
            with st.spinner("Embedding and indexing documents... this may take a moment"):
                used_dir = ingest_files(
                    file_paths,
                    persist_dir=chroma_dir,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
            st.success(f"Ingested {len(file_paths)} files into {used_dir}")

st.divider()


# Initialize chat sessions
if "chats" not in st.session_state:
    loaded_chats, loaded_active = load_chats_from_disk()
    st.session_state.chats = loaded_chats or []  # [{"id": str, "title": str, "history": List[{role, content}]}]
    st.session_state.active_chat_id = loaded_active
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# Controls row
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 2])
with ctrl_col1:
    if st.button("➕ New Chat", use_container_width=True):
        cid = uuid.uuid4().hex[:8]
        st.session_state.chats.append({"id": cid, "title": f"Chat {len(st.session_state.chats)+1}", "history": []})
        st.session_state.active_chat_id = cid
        save_chats_to_disk()

# Build selector options
ids = [c["id"] for c in st.session_state.chats]
titles = [c["title"] for c in st.session_state.chats]
idx = ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in ids else (0 if ids else -1)

with ctrl_col2:
    if ids:
        sel = st.selectbox("Active chat", options=list(range(len(ids))), format_func=lambda i: titles[i], index=idx)
        st.session_state.active_chat_id = ids[sel]
        save_chats_to_disk()
    else:
        st.info("Start by creating a new chat.")

with ctrl_col3:
    # Rename active chat
    if st.session_state.active_chat_id and ids:
        i = ids.index(st.session_state.active_chat_id)
        new_title = st.text_input("Rename chat", value=titles[i], key=f"ttl_{ids[i]}")
        if new_title and new_title != titles[i]:
            st.session_state.chats[i]["title"] = new_title
            save_chats_to_disk()

st.markdown("---")

# Active chat
active = None
for c in st.session_state.chats:
    if c["id"] == st.session_state.active_chat_id:
        active = c
        break

# Render history
if active and active["history"]:
    for m in active["history"]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

# Helper: compute citations with snippets
def compute_citations(question: str, vs, k: int = 5) -> List[Tuple[str, float, str]]:
    # returns list of (source, score, snippet)
    items: List[Tuple[str, float, str]] = []
    try:
        if hasattr(vs, "similarity_search_with_relevance_scores"):
            results = vs.similarity_search_with_relevance_scores(question, k=k)
            for doc, score in results:
                src = doc.metadata.get("source", "unknown")
                snippet = (doc.page_content or "").strip()[:400]
                items.append((src, float(score), snippet))
        else:
            results = vs.similarity_search_with_score(question, k=k)
            for doc, dist in results:
                src = doc.metadata.get("source", "unknown")
                rel = max(0.0, min(1.0, 1.0 - float(dist)))
                snippet = (doc.page_content or "").strip()[:400]
                items.append((src, rel, snippet))
    except Exception:
        return []
    return items

# Helper: streaming generation
def stream_answer(question: str, history: List[dict], retriever, model_name: str) -> str:
    # Build context (sync)
    docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join(d.page_content for d in docs)
    hist_lines = []
    for m in history[-4:]:
        hist_lines.append(f"{m['role']}: {m['content']}")
    hist = "\n".join(hist_lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context and chat history to answer. If unsure, say you don't know."),
        ("user", "Chat history (may be empty):\n{history}\n\nQuestion: {question}\n\nContext:\n{context}")
    ])
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2, streaming=True)
    chain = prompt | llm

    # Stream into the UI
    output_text = ""
    for chunk in chain.stream({"question": question, "context": context_text, "history": hist}):
        piece = getattr(chunk, "content", str(chunk))
        output_text += piece
        yield piece
    # Final full text is output_text (returned to caller if needed)

# Chat input
user_msg = st.chat_input("Message your knowledge base...")
if user_msg:
    # Ensure VS/retriever exists
    try:
        retriever = get_retriever(chroma_dir)
        vs = get_vectorstore(chroma_dir)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(str(e))
    else:
        # Create default chat if none
        if not active:
            cid = uuid.uuid4().hex[:8]
            active = {"id": cid, "title": f"Chat {len(st.session_state.chats)+1}", "history": []}
            st.session_state.chats.append(active)
            st.session_state.active_chat_id = cid

        # Append user turn and render
        active["history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        save_chats_to_disk()

        # Prepare assistant container
        assistant_box = st.chat_message("assistant")
        with assistant_box:
            if stream_mode:
                placeholder = st.empty()
                acc = ""
                for token in stream_answer(user_msg, active["history"], retriever, model_name):
                    acc += token
                    placeholder.markdown(acc)
                answer = acc
            else:
                graph = build_rag_graph(retriever, model_name=model_name)
                with st.spinner("Thinking..."):
                    out = run_rag(graph, user_msg, history=active["history"])
                answer = out.get("answer", "No answer")
                st.markdown(answer)

            # Compute citations and provenance
            citations = compute_citations(user_msg, vs, k=5)
            provenance = round(sum(s for _, s, _ in citations) / len(citations), 3) if citations else 0.0

            # Render inline markers and snippets
            if citations:
                markers = " ".join(f"[{i+1}]" for i in range(len(citations)))
                st.markdown(f"\n\n**Citations:** {markers}")
                with st.expander("🔎 Source snippets", expanded=False):
                    for i, (src, score, snippet) in enumerate(citations, start=1):
                        st.markdown(f"**[{i}] {src}** — similarity: {score:.3f}")
                        st.code(snippet)
                st.caption(f"Provenance score (0-1): {provenance:.3f}")
            else:
                st.caption("No citations available.")

        # Save assistant turn
        active["history"].append({"role": "assistant", "content": answer})
        save_chats_to_disk()

# ---------- Footer ----------
st.markdown("""
<style>
/* Simple modern look */
section.main > div {padding-top: 0.25rem;}
</style>
""", unsafe_allow_html=True)
