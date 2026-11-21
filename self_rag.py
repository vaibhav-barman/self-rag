"""Self-RAG: retrieval that decides, grades and checks itself.

Run:
    uv run self_rag.py ingest [--force]
    uv run self_rag.py ask "your question"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-RAG with Groq + Ollama")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="embed documents/ into Chroma")
    p_ingest.add_argument("--force", action="store_true", help="rebuild even if already ingested")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            ingest(force=args.force)
    except SelfRagError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
