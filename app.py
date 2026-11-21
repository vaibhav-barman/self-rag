"""Streamlit UI for Self-RAG. Run with: uv run streamlit run app.py"""

from __future__ import annotations

import streamlit as st

import self_rag
from self_rag import GraphState, SelfRagError

st.set_page_config(page_title="Self-RAG", page_icon="🔎")

SUPPORT_LABELS = {
    "fully_supported": "✅ fully supported",
    "partially_supported": "⚠️ partially supported",
    "no_support": "❌ no support",
}


@st.cache_resource(show_spinner="Loading models and vector store...")
def get_graph():
    """Compile the graph once per server process (LLM client + Chroma handle are reused)."""
    return self_rag.build_graph()


def render_trace(result: GraphState) -> None:
    """Show the Self-RAG decisions that led to the answer."""
    retrieved = result.get("needs_retrieval", False)
    st.markdown(
        f"**Retrieve?** {'yes' if retrieved else 'no'} — {result.get('retrieval_reason', '')}"
    )
    if result.get("docs"):
        st.markdown("**Relevance check (ISREL)** on retrieved chunks")
        for doc, ok in zip(result["docs"], result["relevance"]):
            snippet = " ".join(doc.page_content.split())[:160]
            st.markdown(f"- {'✅' if ok else '❌'} `{self_rag.format_source(doc)}` — {snippet}…")
    if "support" in result:
        st.markdown(f"**Support check (ISSUP):** {SUPPORT_LABELS[result['support']]}")
    if "useful" in result:
        verdict = "✅ useful" if result["useful"] else "❌ not useful"
        st.markdown(f"**Usefulness check (ISUSE):** {verdict} — {result.get('usefulness_reason', '')}")
    col_rev, col_ret = st.columns(2)
    col_rev.metric("Revisions", result.get("revisions", 0))
    col_ret.metric("Query rewrites (retries)", result.get("retries", 0))
    if result.get("rewritten_question"):
        st.markdown(f"**Rewritten query:** {result['rewritten_question']}")
    if result.get("limited_context"):
        st.warning("Retries exhausted: this is a limited-context answer.")
    sources = result.get("sources") or []
    st.markdown("**Sources:** " + ("; ".join(sources) if sources else "none (no document used)"))


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            with st.expander("How Self-RAG answered this"):
                render_trace(message["result"])


# --- Sidebar: ingestion ------------------------------------------------------
with st.sidebar:
    st.header("Documents")
    force = st.checkbox("Force rebuild", help="Delete the existing index and re-embed everything.")
    if st.button("Ingest documents"):
        try:
            with st.spinner("Embedding documents with Ollama (this can take a few minutes)..."):
                total = self_rag.ingest(force=force)
            get_graph.clear()  # the old graph holds a handle to the pre-ingest collection
            st.success(f"Index ready: {total} chunks.")
        except SelfRagError as exc:
            st.error(str(exc))

# --- Chat --------------------------------------------------------------------
st.title("Self-RAG")
st.caption("Retrieval that decides, grades and checks itself.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    render_message(message)

if question := st.chat_input("Ask a question about your documents"):
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)
    try:
        with st.spinner("Thinking..."):
            result = self_rag.ask(question, get_graph())
        reply = {"role": "assistant", "content": result["final_answer"], "result": result}
    except SelfRagError as exc:
        reply = {"role": "assistant", "content": f"⚠️ {exc}"}
    st.session_state.messages.append(reply)
    render_message(reply)
