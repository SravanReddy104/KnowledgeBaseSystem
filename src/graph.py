from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY


class ChatMessage(BaseModel):
    role: str = Field(description="user or assistant")
    content: str


class RAGState(BaseModel):
    question: str
    context: List[Document] = Field(default_factory=list)
    answer: str = ""
    history: List[ChatMessage] = Field(default_factory=list)


def _format_history(history: List[ChatMessage], max_turns: int = 4) -> str:
    if not history:
        return ""
    turns = history[-max_turns:]
    lines = []
    for m in turns:
        lines.append(f"{m.role}: {m.content}")
    return "\n".join(lines)


def build_rag_graph(retriever, model_name: Optional[str] = None) -> StateGraph:
    graph = StateGraph(RAGState)

    def retrieve_node(state: RAGState) -> RAGState:
        docs = retriever.get_relevant_documents(state.question)  # sync for simplicity
        return state.model_copy(update={"context": docs})

    def generate_node(state: RAGState, config: RunnableConfig | None = None) -> RAGState:
        context_text = "\n\n".join([d.page_content for d in state.context])
        hist = _format_history(state.history)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context and chat history to answer. If unsure, say you don't know."),
            ("user", "Chat history (may be empty):\n{history}\n\nQuestion: {question}\n\nContext:\n{context}")
        ])
        llm = ChatGoogleGenerativeAI(
            model=model_name or GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            streaming=False,
        )
        chain = prompt | llm
        resp = chain.invoke({"question": state.question, "context": context_text, "history": hist})
        return state.model_copy(update={"answer": resp.content})

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_rag(graph, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    hist_models = [ChatMessage(**m) for m in (history or [])]
    init_state = RAGState(question=question, history=hist_models)
    result = graph.invoke(init_state)
    # Ensure dict output for Streamlit rendering
    if isinstance(result, RAGState):
        return result.model_dump()
    return result
