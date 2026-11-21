"""Self-RAG: retrieval that decides, grades and checks itself.

Run:
    uv run self_rag.py ingest [--force]
    uv run self_rag.py ask "your question"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypedDict

import groq
import ollama
from dotenv import load_dotenv
from groq import BadRequestError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

load_dotenv()
logger = logging.getLogger("self_rag")

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
MAX_CONCURRENCY = 2  # parallel Groq calls per batch (keeps us inside free-tier rate limits)
GROQ_MAX_RETRIES = 8  # client retries on 429 / transient errors (it honours Groq's retry-after)
REASONING_EFFORT = "low"  # gpt-oss reasoning depth; "low" cuts tokens (free tier is 8k tokens/min)
MAX_REVISIONS = 2  # max rounds of "rewrite using only what the context supports"
MAX_RETRIES = 2  # max times we rewrite the question and retrieve again


class SelfRagError(RuntimeError):
    """A user-facing error with a clear message (no traceback needed)."""


@contextmanager
def friendly_errors() -> Iterator[None]:
    """Turn Groq / Ollama failures into short SelfRagError messages.

    Rate limits (429) are retried with backoff by the Groq client (GROQ_MAX_RETRIES);
    only a limit that survives all those retries reaches this handler.
    """
    try:
        yield
    except SelfRagError:
        raise
    except groq.AuthenticationError as exc:
        raise SelfRagError("Groq rejected GROQ_API_KEY. Check the key in your .env file.") from exc
    except groq.RateLimitError as exc:
        raise SelfRagError(
            "Groq rate limit still exceeded after several retries (free tier: ~8k tokens/min). "
            "Wait a minute and try again."
        ) from exc
    except groq.APIConnectionError as exc:
        raise SelfRagError("Cannot reach the Groq API. Check your internet connection.") from exc
    except (ConnectionError, ollama.ResponseError) as exc:
        raise SelfRagError(
            f"Ollama stopped responding ({exc}). Make sure it is running and "
            f"'{EMBED_MODEL}' is installed."
        ) from exc


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
        host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        raise SelfRagError(
            f"Cannot reach Ollama at {host}. Start it (open the Ollama app or run "
            "`ollama serve`) and try again."
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
    with friendly_errors():
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
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_retries=GROQ_MAX_RETRIES,
        reasoning_effort=REASONING_EFFORT,
    )


def structured(llm: BaseChatModel, schema: type[BaseModel]) -> Any:
    """LLM that returns a validated `schema` instance.

    Uses Groq's native json_schema mode, falling back to forced tool calling
    only when the model itself fails to produce valid output. Rate-limit errors are
    deliberately NOT caught here: the client retries them with backoff, and falling
    back would only hide the real error.
    """
    primary = llm.with_structured_output(schema, method="json_schema")
    fallback = llm.with_structured_output(schema, method="function_calling")
    return primary.with_fallbacks(
        [fallback], exceptions_to_handle=(OutputParserException, ValidationError, BadRequestError)
    )


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


Support = Literal["fully_supported", "partially_supported", "no_support"]
SUPPORT_RANK: dict[str, int] = {"fully_supported": 2, "partially_supported": 1, "no_support": 0}


class SupportGrade(BaseModel):
    """ISSUP: is the answer grounded in the passage it was generated from?"""

    verdict: Support = Field(
        description="fully_supported: every claim is stated in the passage. "
        "partially_supported: some claims are, some are not. "
        "no_support: the answer is not backed by the passage."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer that the passage does not support (empty if none).",
    )


class UsefulnessGrade(BaseModel):
    """ISUSE: does the answer actually address the question?"""

    useful: bool = Field(
        description="True if the answer directly and substantively addresses the question. "
        "False if it dodges, only says the information was not found, or is off-topic."
    )
    reason: str = Field(description="One short sentence explaining the verdict.")


class RewrittenQuery(BaseModel):
    """A better search query for the vector store."""

    query: str = Field(description="The rewritten, self-contained retrieval query.")


class BestAnswer(TypedDict):
    """Best answer seen across attempts, kept for the limited-context fallback."""

    answer: str
    rank: int  # SUPPORT_RANK of the answer (-1 for a direct, unsupported-by-docs answer)
    sources: list[str]


class GraphState(TypedDict, total=False):
    """Everything the graph passes between nodes."""

    question: str
    needs_retrieval: bool
    retrieval_reason: str
    docs: list[Document]  # retrieved chunks
    relevance: list[bool]  # ISREL verdict for each chunk in `docs`
    relevant_docs: list[Document]  # chunks that passed ISREL
    answers: list[str]  # one candidate answer per relevant chunk
    support_verdicts: list[str]  # ISSUP verdict per candidate answer
    unsupported_claims: list[list[str]]  # unsupported claims per candidate answer
    revisions: int  # revision rounds used so far
    selected_index: int  # which candidate (and chunk) was chosen
    selected_answer: str
    support: str  # ISSUP verdict of the selected answer
    useful: bool  # ISUSE verdict
    usefulness_reason: str
    retries: int  # question rewrites used so far
    rewritten_question: str  # latest retrieval query (empty until a rewrite happens)
    best: BestAnswer  # best not-useful answer so far (for the fallback)
    limited_context: bool  # True if we gave up and returned the best effort
    sources: list[str]  # "file.pdf, p. N" for the chunk behind the final answer
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

SUPPORT_PROMPT = """Check whether the answer is grounded in the passage. Judge only against the \
passage, not against your own knowledge.

Passage:
{passage}

Question: {question}

Answer:
{answer}"""

REVISE_PROMPT = """Rewrite the answer so it contains only information stated in the passage. \
Remove or correct every unsupported claim. If the passage does not answer the question, say so \
plainly.

Passage:
{passage}

Question: {question}

Current answer:
{answer}

Unsupported claims to remove:
{claims}"""

USEFUL_PROMPT = """Decide whether the answer is useful for the question. It is useful only if it \
directly and substantively addresses what was asked. It is NOT useful if it dodges, is \
off-topic, or merely says the information could not be found.

Question: {question}

Answer:
{answer}"""

REWRITE_PROMPT = """The search over a document collection did not produce a good answer. Rewrite \
the question as a better retrieval query: expand abbreviations, add the key technical terms and \
synonyms a textbook would use, and make it self-contained. Do not answer it.

Original question: {question}
Previous query: {previous}
Why the last attempt failed: {feedback}"""

NO_CONTEXT_ANSWER = (
    "I could not find relevant information about this in the documents, so I can't answer "
    "it reliably from them."
)
LIMITED_CONTEXT_NOTE = (
    "\n\n_Note: limited context. The documents did not fully cover this question, "
    "so this answer may be incomplete or only partly relevant._"
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
        self.support_grader = structured(llm, SupportGrade)
        self.usefulness_grader = structured(llm, UsefulnessGrade)
        self.rewriter = structured(llm, RewrittenQuery)

    def decide_retrieval(self, state: GraphState) -> GraphState:
        """Retrieve-or-not decision (Self-RAG's `Retrieve` token). Also zeroes the counters."""
        decision = self.decider.invoke(DECIDE_PROMPT.format(question=state["question"]))
        logger.info("decide_retrieval: retrieve=%s (%s)", decision.should_retrieve, decision.reason)
        return {
            "needs_retrieval": decision.should_retrieve,
            "retrieval_reason": decision.reason,
            "retries": 0,
            "revisions": 0,
            "rewritten_question": "",
            "limited_context": False,
        }

    def generate_direct(self, state: GraphState) -> GraphState:
        """Answer from general knowledge, without touching the documents."""
        answer = str(self.llm.invoke(DIRECT_PROMPT.format(question=state["question"])).content)
        return {"final_answer": answer, "selected_answer": answer, "docs": [], "sources": []}

    def retrieve(self, state: GraphState) -> GraphState:
        """Top-k similarity search, using the rewritten query when there is one."""
        query = state.get("rewritten_question") or state["question"]
        logger.info("retrieve: query=%r", query)
        return {"docs": self.store.similarity_search(query, k=TOP_K)}

    def is_relevant(self, state: GraphState) -> GraphState:
        """ISREL: grade every retrieved chunk and keep only the relevant ones."""
        question, docs = state["question"], state["docs"]
        prompts = [
            RELEVANCE_PROMPT.format(question=question, passage=d.page_content) for d in docs
        ]
        grades = self.relevance_grader.batch(prompts, config=self.batch_cfg)
        verdicts = [g.relevant for g in grades]
        kept = [d for d, ok in zip(docs, verdicts) if ok]
        logger.info("is_relevant: %d/%d chunks relevant %s", len(kept), len(docs), verdicts)
        return {"relevance": verdicts, "relevant_docs": kept}

    def generate_from_context(self, state: GraphState) -> GraphState:
        """One candidate answer per relevant chunk, each using only that chunk."""
        prompts = [
            GENERATE_PROMPT.format(context=d.page_content, question=state["question"])
            for d in state["relevant_docs"]
        ]
        replies = self.llm.batch(prompts, config=self.batch_cfg)
        return {
            "answers": [str(r.content) for r in replies],
            "revisions": 0,
            # fresh attempt: drop verdicts left over from an earlier retrieval round
            "support_verdicts": ["no_support"] * len(replies),
            "unsupported_claims": [[] for _ in replies],
        }

    def is_supported(self, state: GraphState) -> GraphState:
        """ISSUP: grade each candidate against its own chunk (re-grading only weak ones)."""
        docs, answers = state["relevant_docs"], state["answers"]
        verdicts = list(state.get("support_verdicts", ["no_support"] * len(answers)))
        claims = [list(c) for c in state.get("unsupported_claims", [[]] * len(answers))]
        todo = [i for i, v in enumerate(verdicts) if v != "fully_supported"]
        prompts = [
            SUPPORT_PROMPT.format(
                passage=docs[i].page_content, question=state["question"], answer=answers[i]
            )
            for i in todo
        ]
        for i, grade in zip(todo, self.support_grader.batch(prompts, config=self.batch_cfg)):
            verdicts[i], claims[i] = grade.verdict, grade.unsupported_claims
        logger.info("is_supported: %s", verdicts)
        return {"support_verdicts": verdicts, "unsupported_claims": claims}

    def revise_answer(self, state: GraphState) -> GraphState:
        """Rewrite every not-fully-supported answer to keep only what its chunk supports."""
        docs, answers = state["relevant_docs"], list(state["answers"])
        todo = [i for i, v in enumerate(state["support_verdicts"]) if v != "fully_supported"]
        prompts = [
            REVISE_PROMPT.format(
                passage=docs[i].page_content,
                question=state["question"],
                answer=answers[i],
                claims="\n".join(f"- {c}" for c in state["unsupported_claims"][i]) or "- (any)",
            )
            for i in todo
        ]
        for i, reply in zip(todo, self.llm.batch(prompts, config=self.batch_cfg)):
            answers[i] = str(reply.content)
        logger.info("revise_answer: round %d, revised %d answer(s)", state["revisions"] + 1, len(todo))
        return {"answers": answers, "revisions": state["revisions"] + 1}

    def select_answer(self, state: GraphState) -> GraphState:
        """Pick the best-supported candidate (ties go to the better-ranked chunk)."""
        verdicts = state["support_verdicts"]
        best = max(range(len(verdicts)), key=lambda i: (SUPPORT_RANK[verdicts[i]], -i))
        answer = state["answers"][best]
        return {
            "selected_index": best,
            "selected_answer": answer,
            "support": verdicts[best],
            "final_answer": answer,
            "sources": [format_source(state["relevant_docs"][best])],
        }

    def is_useful(self, state: GraphState) -> GraphState:
        """ISUSE: does the selected answer address the question? Remember it if not."""
        answer = state["selected_answer"]
        grade = self.usefulness_grader.invoke(
            USEFUL_PROMPT.format(question=state["question"], answer=answer)
        )
        logger.info("is_useful: %s (%s)", grade.useful, grade.reason)
        update: GraphState = {"useful": grade.useful, "usefulness_reason": grade.reason}
        if not grade.useful:
            rank = SUPPORT_RANK.get(state.get("support", ""), -1)
            if "best" not in state or rank >= state["best"]["rank"]:
                update["best"] = {"answer": answer, "rank": rank, "sources": state["sources"]}
        return update

    def rewrite_question(self, state: GraphState) -> GraphState:
        """Turn the question into a better retrieval query, then retrieve again."""
        feedback = state.get("usefulness_reason") or "no retrieved passage was relevant"
        rewritten = self.rewriter.invoke(
            REWRITE_PROMPT.format(
                question=state["question"],
                previous=state.get("rewritten_question") or state["question"],
                feedback=feedback,
            )
        )
        logger.info("rewrite_question: retry %d -> %r", state["retries"] + 1, rewritten.query)
        return {
            "rewritten_question": rewritten.query,
            "retries": state["retries"] + 1,
            "needs_retrieval": True,  # a direct answer that failed escalates to the documents
        }

    def limited_context(self, state: GraphState) -> GraphState:
        """Retries exhausted: return the best answer found, honestly flagged."""
        best = state.get("best")
        logger.info("limited_context: retries exhausted, best answer found=%s", best is not None)
        if best is None:
            return {"final_answer": NO_CONTEXT_ANSWER, "sources": [], "limited_context": True}
        return {
            "final_answer": best["answer"] + LIMITED_CONTEXT_NOTE,
            "sources": best["sources"],
            "limited_context": True,
        }


# ---------------------------------------------------------------------------
# 7. Routing
# ---------------------------------------------------------------------------


def route_after_decision(state: GraphState) -> Literal["retrieve", "generate_direct"]:
    """Retrieve only if the decision node said the documents are needed."""
    return "retrieve" if state["needs_retrieval"] else "generate_direct"


def route_after_relevance(
    state: GraphState,
) -> Literal["generate_from_context", "rewrite_question", "limited_context"]:
    """Generate if any chunk is relevant; otherwise rewrite the query, or give up honestly."""
    if state["relevant_docs"]:
        return "generate_from_context"
    return "rewrite_question" if state["retries"] < MAX_RETRIES else "limited_context"


def route_after_support(state: GraphState) -> Literal["revise_answer", "select_answer"]:
    """Revise only while no candidate is fully supported and revision budget remains."""
    grounded = "fully_supported" in state["support_verdicts"]
    if not grounded and state["revisions"] < MAX_REVISIONS:
        return "revise_answer"
    return "select_answer"


def route_after_usefulness(state: GraphState) -> Literal["__end__", "rewrite_question", "limited_context"]:
    """Finish if useful; otherwise retry with a rewritten query, or give up honestly."""
    if state["useful"]:
        return END
    return "rewrite_question" if state["retries"] < MAX_RETRIES else "limited_context"


# ---------------------------------------------------------------------------
# 8. Graph
# ---------------------------------------------------------------------------


def build_graph(llm: BaseChatModel | None = None, store: Chroma | None = None) -> Any:
    """Assemble and compile the LangGraph state machine."""
    llm = llm or get_llm()
    store = store or get_vector_store()
    if collection_size(store) == 0:
        raise SelfRagError("The vector store is empty. Ingest your documents first (ingest).")
    nodes = Nodes(llm, store)
    graph = StateGraph(GraphState)
    graph.add_node("decide_retrieval", nodes.decide_retrieval)
    graph.add_node("generate_direct", nodes.generate_direct)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("is_relevant", nodes.is_relevant)
    graph.add_node("generate_from_context", nodes.generate_from_context)
    graph.add_node("is_supported", nodes.is_supported)
    graph.add_node("revise_answer", nodes.revise_answer)
    graph.add_node("select_answer", nodes.select_answer)
    graph.add_node("is_useful", nodes.is_useful)
    graph.add_node("rewrite_question", nodes.rewrite_question)
    graph.add_node("limited_context", nodes.limited_context)
    graph.add_edge(START, "decide_retrieval")
    graph.add_conditional_edges("decide_retrieval", route_after_decision)
    graph.add_edge("generate_direct", "is_useful")
    graph.add_edge("retrieve", "is_relevant")
    graph.add_conditional_edges("is_relevant", route_after_relevance)
    graph.add_edge("generate_from_context", "is_supported")
    graph.add_conditional_edges("is_supported", route_after_support)
    graph.add_edge("revise_answer", "is_supported")
    graph.add_edge("select_answer", "is_useful")
    graph.add_conditional_edges("is_useful", route_after_usefulness)
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("limited_context", END)
    return graph.compile()


def ask(question: str, graph: Any | None = None) -> GraphState:
    """Run one question through the graph and return the final state."""
    with friendly_errors():
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
    parser.add_argument("-v", "--verbose", action="store_true", help="log each graph step")
    return parser


def print_result(result: GraphState) -> None:
    """Pretty-print the answer and its sources."""
    print(f"\nRetrieve? {'yes' if result['needs_retrieval'] else 'no'} ({result['retrieval_reason']})")
    if "relevance" in result:
        print(f"Relevant chunks: {sum(result['relevance'])}/{len(result['relevance'])}")
    if "support" in result:
        print(f"Support: {result['support']} (revisions: {result['revisions']})")
    if "useful" in result:
        print(f"Useful: {result['useful']} ({result.get('usefulness_reason', '')})")
    if result["retries"]:
        print(f"Retries: {result['retries']}, last query: {result['rewritten_question']!r}")
    print("\n=== Answer ===")
    print(result["final_answer"])
    if result["sources"]:
        print("\nSources: " + "; ".join(result["sources"]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "groq", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
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
