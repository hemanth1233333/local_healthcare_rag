# src/retrieve_chroma.py
import os
from pathlib import Path
import requests
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "cholesterol_guidelines"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get(
    "OLLAMA_EMBED_MODEL",
    "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest",
)

def ollama_embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]

def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    question = "How can I lower LDL cholesterol with diet?"
    q_emb = ollama_embed(question)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    for i in range(len(res["documents"][0])):
        doc = res["documents"][0][i]
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i]
        print("\n" + "-" * 80)
        print(f"Rank {i+1} | distance={dist:.4f} | {meta['source']}#chunk{meta['chunk_index']}")
        print(doc[:900])

if __name__ == "__main__":
    main()
