"""
ingest_chroma.py — Chunk Markdown files and ingest embeddings into ChromaDB.
Uses BGE-base-en-v1.5 via Ollama for local embeddings.

Usage:
    # Ingest all markdown files from a directory
    python src/ingest_chroma.py --input data/markdown/ --topic cholesterol

    # Ingest a single file
    python src/ingest_chroma.py --input data/markdown/Cholesterol.md --topic cholesterol

    # Clear collection and re-ingest
    python src/ingest_chroma.py --input data/markdown/ --topic cholesterol --reset

Topics: cholesterol | thyroid | ckd | asthma | anemia
"""

import argparse
import os
import time
from pathlib import Path

import requests
import chromadb

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "healthcare_guidelines"

CHUNK_SIZE_TOKENS = 900      # approximate token window
CHUNK_OVERLAP_TOKENS = 150   # overlap to preserve context across boundaries
CHARS_PER_TOKEN = 4          # rough approximation (1 token ≈ 4 chars for English)

CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN      # ~3600 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~600 chars


# ──────────────────────────────────────────────
# Safe chunker (guaranteed forward progress)
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Split text into overlapping chunks with guaranteed forward progress.
    No infinite loop risk: always advances by at least (chunk_size - overlap).
    Tries to split on paragraph or sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    min_advance = chunk_size - overlap  # guaranteed minimum advance per iteration

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find a good split point (paragraph > sentence > word)
        split_pos = end

        # Look for paragraph break in the last 20% of window
        search_start = start + int(chunk_size * 0.8)
        para_pos = text.rfind("\n\n", search_start, end)
        if para_pos > search_start:
            split_pos = para_pos
        else:
            # Try sentence boundary
            for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                sent_pos = text.rfind(punct, search_start, end)
                if sent_pos > search_start:
                    split_pos = sent_pos + 1
                    break
            else:
                # Fall back to word boundary
                word_pos = text.rfind(" ", search_start, end)
                if word_pos > search_start:
                    split_pos = word_pos

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Guaranteed forward progress: advance by at least min_advance
        next_start = split_pos - overlap
        start = max(next_start, start + min_advance)

    return chunks


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────
def get_embedding(text: str, retries: int = 3) -> list[float]:
    """Get BGE embedding from local Ollama server with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  [embed] Retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(2)


# ──────────────────────────────────────────────
# ChromaDB helpers
# ──────────────────────────────────────────────
def get_collection(reset: bool = False):
    """Get or create ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[chroma] Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ──────────────────────────────────────────────
# Ingest a single markdown file
# ──────────────────────────────────────────────
def ingest_file(
    md_path: str,
    topic: str,
    collection,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS
) -> int:
    """
    Read a markdown file, chunk it, embed each chunk, and store in ChromaDB.
    Returns number of chunks ingested.
    """
    md_path = Path(md_path)
    source_name = md_path.stem  # filename without extension, e.g. "Cholesterol"

    print(f"\n[ingest] Processing: {md_path.name} (topic={topic})")

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print(f"  [ingest] SKIP: empty file")
        return 0

    chunks = chunk_text(text, chunk_size, overlap)
    print(f"  [ingest] {len(chunks)} chunks from {len(text)} chars")

    # Check for already-ingested chunks from this source
    existing = collection.get(where={"source": source_name})
    if existing["ids"]:
        print(f"  [ingest] Found {len(existing['ids'])} existing chunks for '{source_name}', skipping.")
        return 0

    ingested = 0
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{source_name}__chunk{idx}"
        embedding = get_embedding(chunk)

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": source_name,
                "chunk_index": idx,
                "topic": topic,
                "file_path": str(md_path),
                "char_count": len(chunk)
            }]
        )
        ingested += 1

        if (idx + 1) % 10 == 0:
            print(f"  [ingest] Progress: {idx + 1}/{len(chunks)} chunks embedded")

    print(f"  [ingest] Done: {ingested} chunks ingested for '{source_name}'")
    return ingested


# ──────────────────────────────────────────────
# Ingest directory
# ──────────────────────────────────────────────
def ingest_directory(input_dir: str, topic: str, collection, reset: bool = False) -> dict:
    """Ingest all markdown files from a directory."""
    input_dir = Path(input_dir)
    md_files = list(input_dir.glob("*.md"))

    if not md_files:
        print(f"[ingest] No .md files found in {input_dir}")
        return {}

    print(f"[ingest] Found {len(md_files)} markdown files")
    results = {}
    total_chunks = 0

    for md_file in sorted(md_files):
        count = ingest_file(str(md_file), topic, collection)
        results[md_file.name] = count
        total_chunks += count

    print(f"\n[ingest] ── Summary ──")
    print(f"  Files processed: {len(md_files)}")
    print(f"  Total chunks ingested: {total_chunks}")
    print(f"  Collection size: {collection.count()} total chunks")
    return results


# ──────────────────────────────────────────────
# Collection stats
# ──────────────────────────────────────────────
def print_collection_stats(collection):
    """Print breakdown of chunks by topic and source."""
    total = collection.count()
    print(f"\n[chroma] Collection '{COLLECTION_NAME}' — {total} total chunks")

    if total == 0:
        return

    all_meta = collection.get(include=["metadatas"])["metadatas"]

    # Count by topic
    topic_counts = {}
    source_counts = {}
    for meta in all_meta:
        t = meta.get("topic", "unknown")
        s = meta.get("source", "unknown")
        topic_counts[t] = topic_counts.get(t, 0) + 1
        source_counts[s] = source_counts.get(s, 0) + 1

    print("\n  By topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"    {topic:20s}: {count} chunks")

    print("\n  By source document:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source:40s}: {count} chunks")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ingest Markdown files into ChromaDB with BGE embeddings"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to .md file or directory containing .md files"
    )
    parser.add_argument(
        "--topic", required=True,
        choices=["cholesterol", "thyroid", "ckd", "asthma", "anemia", "general"],
        help="Disease topic label for metadata"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate the ChromaDB collection before ingesting"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print collection statistics and exit"
    )
    args = parser.parse_args()

    collection = get_collection(reset=args.reset)

    if args.stats:
        print_collection_stats(collection)
        return

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() == ".md":
        ingest_file(str(input_path), args.topic, collection)
    elif input_path.is_dir():
        ingest_directory(str(input_path), args.topic, collection)
    else:
        print(f"[ingest] ERROR: {args.input} is not a .md file or directory")
        exit(1)

    print_collection_stats(collection)


if __name__ == "__main__":
    main()
