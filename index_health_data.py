"""
index_health_data.py
--------------------
One-time job: embeds all chunks into a persistent ChromaDB vector store
using LlamaIndex + a local HuggingFace embedding model.

Run this AFTER parse_health_data.py and build_chunks.py.
Re-run only if you re-export your Apple Health data.

Usage:
    python index_health_data.py
    python index_health_data.py --chunks ./chunks --db ./health_db
"""

import argparse
from pathlib import Path

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from build_chunks import load_chunks


# Embedding model — runs fully locally, no API key needed
# bge-small-en embeds almost all daily/weekly prose chunks with cosine
# similarity ~0.95+ regardless of date or content, which collapses
# retrieval onto a narrow cluster of chunks. bge-base-en-v1.5 produces
# meaningfully different embeddings for chunks with different dates/metrics.
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def build_index(chunks_dir: str = "./chunks", db_path: str = "./health_db") -> VectorStoreIndex:
    """
    Load all chunks, embed them, and persist to ChromaDB.
    Returns the VectorStoreIndex for immediate use.
    """

    # ── Load chunks ──────────────────────────────────────────────────────────
    print("Loading chunks…")
    chunks = load_chunks(chunks_dir)
    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_dir}. Run build_chunks.py first.")

    print(f"  {len(chunks):,} total chunks ({sum(1 for c in chunks if c['type']=='daily')} daily, "
          f"{sum(1 for c in chunks if c['type']=='sleep')} sleep, "
          f"{sum(1 for c in chunks if c['type']=='weekly')} weekly)")

    # ── Build LlamaIndex Documents ───────────────────────────────────────────
    documents = [
        Document(
            text=chunk["text"],
            metadata={
                "type": chunk["type"],
                "date": chunk["date"],
            },
            # Exclude metadata fields from embedding context (keep embedding pure text)
            excluded_embed_metadata_keys=["type", "date"],
            excluded_llm_metadata_keys=[],
        )
        for chunk in chunks
    ]

    # ── Embedding model (local) ──────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL_NAME} …")
    print("  (First run will download ~130MB; subsequent runs use cache)")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.llm = None  # Don't need LLM during indexing

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    print(f"\nConnecting to ChromaDB at {db_path} …")
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(
        "health_data",
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Index ────────────────────────────────────────────────────────────────
    print(f"\nEmbedding and indexing {len(documents):,} documents…")
    print("  (This may take a few minutes — runs only once)")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\n✓ Index built and persisted to {db_path}/")
    print("  You can now run app.py")
    return index


def load_index(db_path: str = "./health_db") -> VectorStoreIndex:
    """
    Load an existing index from ChromaDB (fast — no re-embedding).
    Call this from app.py instead of build_index().
    """
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection("health_data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="./chunks",    help="Directory of chunk JSON files")
    parser.add_argument("--db",     default="./health_db", help="ChromaDB output directory")
    args = parser.parse_args()

    if not Path(args.chunks).exists():
        print(f"ERROR: {args.chunks} not found. Run build_chunks.py first.")
        raise SystemExit(1)

    build_index(chunks_dir=args.chunks, db_path=args.db)
