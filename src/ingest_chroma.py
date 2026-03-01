# src/ingest_chroma.py
# Ingest markdown docs -> chunk safely (no infinite loop) -> embed via Ollama -> store in Chroma (persistent)

print("STEP 1: start script")

import os
from pathlib import Path
from typing import List, Dict, Any

print("STEP 2: basic imports ok")

import requests

print("STEP 3: requests import ok")

import chromadb

print("STEP 4: chromadb import ok")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_DIR = PROJECT_ROOT / "data" / "processed" / "guidelines_docling"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "cholesterol_guidelines"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get(
    "OLLAMA_EMBED_MODEL",
    "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest",
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
UPSERT_BATCH = 8  # keep small to reduce load


def chunk_text(text: str) -> List[str]:
    """
    Safe chunker: guarantees forward progress (prevents infinite loops).
    """
    text = text.replace("\r\n", "\n")

    chunk_size = CHUNK_SIZE
    overlap = CHUNK_OVERLAP

    # guard: overlap must be smaller than chunk_size
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    i, n = 0, len(text)

    while i < n:
        j = min(i + chunk_size, n)
        c = text[i:j].strip()
        if c:
            chunks.append(c)

        if j == n:
            break

        next_i = j - overlap
        # guard: ensure progress even if overlap logic would stall
        if next_i <= i:
            next_i = j
        i = next_i

    return chunks


def ollama_embed(text: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def main() -> None:
    print("STEP 5: entering main")

    if not MD_DIR.exists():
        raise FileNotFoundError(f"Missing markdown folder: {MD_DIR}")

    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {MD_DIR}")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("STEP 6: before PersistentClient")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    print("STEP 7: after PersistentClient")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("STEP 8: got/created collection")

    total = 0

    for md_path in md_files:
        print(f"Reading: {md_path.name}")

        text = md_path.read_text(encoding="utf-8", errors="ignore")
        print("STEP 9: file read ok")

        chunks = chunk_text(text)
        print(f"STEP 10: chunking ok ({len(chunks)} chunks)")

        ids = [f"{md_path.stem}-c{i}" for i in range(len(chunks))]
        metadatas: List[Dict[str, Any]] = [
            {"source": md_path.name, "chunk_index": i, "path": str(md_path)}
            for i in range(len(chunks))
        ]

        for start in range(0, len(chunks), UPSERT_BATCH):
            batch_chunks = chunks[start : start + UPSERT_BATCH]
            batch_ids = ids[start : start + UPSERT_BATCH]
            batch_meta = metadatas[start : start + UPSERT_BATCH]

            batch_embs = [ollama_embed(c) for c in batch_chunks]

            collection.upsert(
                ids=batch_ids,
                documents=batch_chunks,
                metadatas=batch_meta,
                embeddings=batch_embs,
            )

        total += len(chunks)
        print(f"✅ indexed {md_path.name}")

    print(f"\nDone. Total chunks stored: {total}")
    print(f"Chroma persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
