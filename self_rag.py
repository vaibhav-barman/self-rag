"""Self-RAG: retrieval that decides, grades and checks itself.

Run:
    uv run self_rag.py ingest [--force]
    uv run self_rag.py ask "your question"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

import ollama
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

DOCUMENTS_DIR = Path("documents")  # where the source PDFs / text files live
CHROMA_DIR = "./chroma_db"  # on-disk location of the persisted vector store
COLLECTION_NAME = "self_rag_docs"  # named Chroma collection

CHUNK_SIZE = 1000  # max characters per chunk: small enough to stay on-topic, big enough for context
CHUNK_OVERLAP = 200  # characters shared by neighbouring chunks so sentences cut at a boundary survive

EMBED_MODEL = "mxbai-embed-large"  # 1024-dim Ollama embedding model
# mxbai-embed-large is trained to see this prefix on *queries* (never on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH = 64  # chunks embedded per call to Ollama / written per call to Chroma

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL = os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL  # override via env var
TOP_K = 4  # chunks fetched per similarity search
MAX_CONCURRENCY = 3  # parallel Groq calls per batch (keeps us inside free-tier rate limits)


class SelfRagError(RuntimeError):
    """A user-facing error with a clear message (no traceback needed)."""


# ---------------------------------------------------------------------------
# 2. Embeddings & vector store
# ---------------------------------------------------------------------------


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """mxbai-embed-large embeddings: prefix queries, leave documents untouched."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(QUERY_PREFIX + text)

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(QUERY_PREFIX + text)


def check_ollama() -> None:
    """Fail fast with a clear message if Ollama is down or the model is missing."""
    try:
        installed = [m.model for m in ollama.Client().list().models]
    except Exception as exc:  # connection refused, etc.
        raise SelfRagError(
            "Cannot reach Ollama at http://localhost:11434. Start it (open the Ollama "
            "app or run `ollama serve`) and try again."
        ) from exc
    if not any(name and name.split(":")[0] == EMBED_MODEL for name in installed):
        raise SelfRagError(
            f"Embedding model '{EMBED_MODEL}' is not installed. Run: ollama pull {EMBED_MODEL}"
        )


def get_embeddings() -> PrefixedOllamaEmbeddings:
    """Build the embeddings client (after checking Ollama is ready)."""
    check_ollama()
    return PrefixedOllamaEmbeddings(model=EMBED_MODEL)


def get_vector_store(
    embeddings: PrefixedOllamaEmbeddings | None = None,
    persist_dir: str = CHROMA_DIR,
    collection: str = COLLECTION_NAME,
) -> Chroma:
    """Open (or create) the persisted Chroma collection."""
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=persist_dir,
    )


def collection_size(store: Chroma) -> int:
    """Number of vectors currently stored."""
    return store._collection.count()


# ---------------------------------------------------------------------------
# 3. Ingest
# ---------------------------------------------------------------------------


def load_pdf(path: Path) -> list[Document]:
    """One Document per non-empty PDF page; metadata has `source` and 0-based `page`."""
    pages = (page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return [
        Document(page_content=text, metadata={"source": str(path), "page": i})
        for i, text in enumerate(pages)
        if text.strip()
    ]


def load_text(path: Path) -> list[Document]:
    """A plain-text / markdown file as a single Document."""
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(path)})]


def load_documents(folder: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load every PDF / .txt / .md file in `folder` (one Document per PDF page)."""
    if not folder.is_dir():
        raise SelfRagError(f"Documents folder '{folder}' does not exist.")
    docs: list[Document] = []
    files = sorted(p for p in folder.iterdir() if p.is_file())
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = load_pdf(path)
        elif suffix in {".txt", ".md"}:
            loaded = load_text(path)
        else:
            continue
        print(f"  loaded {path.name}: {len(loaded)} page(s)/section(s)")
        docs.extend(loaded)
    if not docs:
        raise SelfRagError(f"No PDF, .txt or .md files found in '{folder}'.")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def ingest(force: bool = False, folder: Path = DOCUMENTS_DIR, store: Chroma | None = None) -> int:
    """Load, split, embed and store documents. Skips if already ingested unless `force`.

    Returns the number of vectors in the collection afterwards.
    """
    store = store or get_vector_store()
    existing = collection_size(store)
    if existing and not force:
        print(f"Collection already has {existing} vectors; skipping (use --force to rebuild).")
        return existing
    if existing:
        print(f"--force: deleting {existing} existing vectors")
        store.reset_collection()

    print(f"Loading documents from {folder}/")
    chunks = split_documents(load_documents(folder))
    for start in range(0, len(chunks), EMBED_BATCH):
        store.add_documents(chunks[start : start + EMBED_BATCH])
        print(f"  embedded {min(start + EMBED_BATCH, len(chunks))}/{len(chunks)}")
    total = collection_size(store)
    print(f"Done. Collection '{store._collection.name}' now holds {total} vectors.")
    return total


# ---------------------------------------------------------------------------
# 4. LLM (Groq) & structured output
# ---------------------------------------------------------------------------


def get_llm() -> ChatGroq:
    """Groq chat model (temperature 0). Needs GROQ_API_KEY in the environment / .env."""
    if not os.getenv("GROQ_API_KEY"):
        raise SelfRagError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
    return ChatGroq(model=GROQ_MODEL, temperature=0)


def structured(llm: BaseChatModel, schema: type[BaseModel]) -> Any:
    """LLM that returns a validated `schema` instance.

    Uses Groq's native json_schema mode, falling back to forced tool calling
    if that call fails. Output is never parsed from free text.
    """
    primary = llm.with_structured_output(schema, method="json_schema")
    fallback = llm.with_structured_output(schema, method="function_calling")
    return primary.with_fallbacks([fallback])


# ---------------------------------------------------------------------------
# 5. State & schemas
# ---------------------------------------------------------------------------


class RetrievalDecision(BaseModel):
    """Does this question need the external documents?"""

    should_retrieve: bool = Field(
        description="True if the question needs facts from the document collection; "
        "False if it can be answered from general knowledge."
    )
    reason: str = Field(description="One short sentence explaining the decision.")


class RelevanceGrade(BaseModel):
    """ISREL: is this chunk relevant to the question?"""

    relevant: bool = Field(
        description="True if the passage contains information that helps answer the question."
    )


class GraphState(TypedDict, total=False):
    """Everything the graph passes between nodes."""

    question: str
    needs_retrieval: bool
    retrieval_reason: str
    docs: list[Document]  # retrieved chunks
    relevance: list[bool]  # ISREL verdict for each chunk in `docs`
    relevant_docs: list[Document]  # chunks that passed ISREL
    answers: list[str]  # one candidate answer per relevant chunk
    selected_answer: str
    final_answer: str


# ---------------------------------------------------------------------------
# 6. Nodes
# ---------------------------------------------------------------------------

DECIDE_PROMPT = """You decide whether a question needs a private document collection. The \
collection contains technical books on: database internals and modern databases, network \
programming (client/server, sockets), and software development practices.

Retrieve (should_retrieve=true) when the question is about those subjects, asks for specific \
facts, definitions or details likely found in such books, or refers to "the documents".
Do not retrieve (should_retrieve=false) for greetings, simple arithmetic, common general \
knowledge, or topics unrelated to the collection.

Question: {question}"""

DIRECT_PROMPT = """Answer the question concisely from your general knowledge.

Question: {question}"""

RELEVANCE_PROMPT = """Decide whether the passage contains information that helps answer the \
question. Mark it relevant only if it actually bears on the question, not merely if it shares \
keywords.

Question: {question}

Passage:
{passage}"""

GENERATE_PROMPT = """Answer the question using only the passage below. Do not add outside \
knowledge. If the passage does not contain the answer, say so.

Passage:
{context}

Question: {question}"""

NO_CONTEXT_ANSWER = (
    "I could not find relevant information about this in the documents, so I can't answer "
    "it reliably from them."
)


def format_source(doc: Document) -> str:
    """'file.pdf, p. 12' (PDF pages are stored 0-based, shown 1-based)."""
    name = Path(doc.metadata.get("source", "unknown")).name
    page = doc.metadata.get("page")
    return f"{name}, p. {page + 1}" if isinstance(page, int) else name


class Nodes:
    """Graph nodes bound to one LLM and one vector store."""

    def __init__(self, llm: BaseChatModel, store: Chroma) -> None:
        self.llm = llm
        self.store = store
        self.batch_cfg = {"max_concurrency": MAX_CONCURRENCY}
        self.decider = structured(llm, RetrievalDecision)
        self.relevance_grader = structured(llm, RelevanceGrade)

    def decide_retrieval(self, state: GraphState) -> GraphState:
        """Retrieve-or-not decision (Self-RAG's `Retrieve` token)."""
        decision = self.decider.invoke(DECIDE_PROMPT.format(question=state["question"]))
        return {
            "needs_retrieval": decision.should_retrieve,
            "retrieval_reason": decision.reason,
        }

    def generate_direct(self, state: GraphState) -> GraphState:
        """Answer from general knowledge, without touching the documents."""
        answer = self.llm.invoke(DIRECT_PROMPT.format(question=state["question"]))
        return {"final_answer": str(answer.content), "docs": []}

    def retrieve(self, state: GraphState) -> GraphState:
        """Top-k similarity search over the Chroma collection."""
        docs = self.store.similarity_search(state["question"], k=TOP_K)
        return {"docs": docs}

    def is_relevant(self, state: GraphState) -> GraphState:
        """ISREL: grade every retrieved chunk and keep only the relevant ones."""
        question, docs = state["question"], state["docs"]
        prompts = [
            RELEVANCE_PROMPT.format(question=question, passage=d.page_content) for d in docs
        ]
        grades = self.relevance_grader.batch(prompts, config=self.batch_cfg)
        verdicts = [g.relevant for g in grades]
        kept = [d for d, ok in zip(docs, verdicts) if ok]
        return {"relevance": verdicts, "relevant_docs": kept}

    def generate_from_context(self, state: GraphState) -> GraphState:
        """One candidate answer per relevant chunk, each using only that chunk."""
        prompts = [
            GENERATE_PROMPT.format(context=d.page_content, question=state["question"])
            for d in state["relevant_docs"]
        ]
        replies = self.llm.batch(prompts, config=self.batch_cfg)
        return {"answers": [str(r.content) for r in replies]}

    def select_answer(self, state: GraphState) -> GraphState:
        """Pick the answer to return (first candidate for now)."""
        answer = state["answers"][0]
        return {"selected_answer": answer, "final_answer": answer}

    def no_context(self, _state: GraphState) -> GraphState:
        """Honest fallback when no retrieved chunk is relevant."""
        return {"final_answer": NO_CONTEXT_ANSWER}


# ---------------------------------------------------------------------------
# 7. Routing
# ---------------------------------------------------------------------------


def route_after_decision(state: GraphState) -> Literal["retrieve", "generate_direct"]:
    """Retrieve only if the decision node said the documents are needed."""
    return "retrieve" if state["needs_retrieval"] else "generate_direct"


def route_after_relevance(state: GraphState) -> Literal["generate_from_context", "no_context"]:
    """Generate if at least one chunk is relevant, otherwise give the honest fallback."""
    return "generate_from_context" if state["relevant_docs"] else "no_context"


# ---------------------------------------------------------------------------
# 8. Graph
# ---------------------------------------------------------------------------


def build_graph(llm: BaseChatModel | None = None, store: Chroma | None = None) -> Any:
    """Assemble and compile the LangGraph state machine."""
    nodes = Nodes(llm or get_llm(), store or get_vector_store())
    graph = StateGraph(GraphState)
    graph.add_node("decide_retrieval", nodes.decide_retrieval)
    graph.add_node("generate_direct", nodes.generate_direct)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("is_relevant", nodes.is_relevant)
    graph.add_node("generate_from_context", nodes.generate_from_context)
    graph.add_node("select_answer", nodes.select_answer)
    graph.add_node("no_context", nodes.no_context)
    graph.add_edge(START, "decide_retrieval")
    graph.add_conditional_edges("decide_retrieval", route_after_decision)
    graph.add_edge("generate_direct", END)
    graph.add_edge("retrieve", "is_relevant")
    graph.add_conditional_edges("is_relevant", route_after_relevance)
    graph.add_edge("generate_from_context", "select_answer")
    graph.add_edge("select_answer", END)
    graph.add_edge("no_context", END)
    return graph.compile()


def ask(question: str, graph: Any | None = None) -> GraphState:
    """Run one question through the graph and return the final state."""
    return (graph or build_graph()).invoke({"question": question})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-RAG with Groq + Ollama")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="embed documents/ into Chroma")
    p_ingest.add_argument("--force", action="store_true", help="rebuild even if already ingested")
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    return parser


def print_result(result: GraphState) -> None:
    """Pretty-print the answer and its sources."""
    print(f"\nRetrieve? {'yes' if result['needs_retrieval'] else 'no'} ({result['retrieval_reason']})")
    if "relevance" in result:
        print(f"Relevant chunks: {sum(result['relevance'])}/{len(result['relevance'])}")
    print("\n=== Answer ===")
    print(result["final_answer"])
    sources = sorted({format_source(d) for d in result.get("relevant_docs", [])})
    if sources:
        print("\nSources: " + "; ".join(sources))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            ingest(force=args.force)
        elif args.command == "ask":
            print_result(ask(args.question))
    except SelfRagError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
