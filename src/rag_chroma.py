# src/rag_chroma.py
import os
from pathlib import Path
import requests
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "cholesterol_guidelines"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "gemma3:latest")  # set to your gemma name

TOP_K = 8
MAX_CONTEXT_CHARS = 5000

# If retrieval seems weak, refuse. You may tune this after you see typical distances.
# Start conservative: if best distance is too large, likely irrelevant.
DISTANCE_REFUSE_THRESHOLD = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "0.55"))

ENABLE_SELF_CHECK = True


def ollama_embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def ollama_generate(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
        timeout=240,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def build_context(docs, metas, max_chars=MAX_CONTEXT_CHARS) -> str:
    out = []
    used = 0
    for doc, meta in zip(docs, metas):
        tag = f"{meta['source']}#chunk{meta['chunk_index']}"
        block = f"[{tag}]\n{doc}\n"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


def strict_answer_prompt(question: str, context: str) -> str:
    return f"""You are a strict, grounded assistant.

RULES (must follow):
1) Use ONLY the CONTEXT below. Do not use outside knowledge.
2) If the CONTEXT does not explicitly contain the answer, output exactly:
NOT FOUND IN PROVIDED DOCUMENTS.
3) Do not guess. Do not infer dates, approvals, or "latest" claims unless stated in context.
4) When you provide an answer, include citations in the form [filename#chunkX] for every factual claim.
5) Do not invent citations. Only cite tags that appear in CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

OUTPUT:
"""


def self_check_prompt(question: str, context: str, draft: str) -> str:
    return f"""You are a verifier.

Check whether the DRAFT answer is fully supported by the CONTEXT.
If any claim is not directly supported, output exactly:
NOT FOUND IN PROVIDED DOCUMENTS.

Otherwise output the same DRAFT unchanged.

QUESTION:
{question}

CONTEXT:
{context}

DRAFT:
{draft}

VERIFIED OUTPUT:
"""


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)

    question = input("Ask a question: ").strip()
    if not question:
        print("No question provided.")
        return

    q_emb = ollama_embed(question)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None]])[0]

    # Print retrieval preview (helps debugging + trust)
    print("\nRetrieved sources:")
    for i in range(min(len(docs), 5)):
        tag = f"{metas[i]['source']}#chunk{metas[i]['chunk_index']}"
        print(f"  {i+1}. distance={dists[i]:.4f} | {tag}")

    # Retrieval guardrail: refuse if best match is weak
    best = dists[0] if dists and dists[0] is not None else None
    if best is not None and best > DISTANCE_REFUSE_THRESHOLD:
        print("\nNOT FOUND IN PROVIDED DOCUMENTS.")
        return

    context = build_context(docs, metas)

    prompt = strict_answer_prompt(question, context)
    draft = ollama_generate(prompt)

    # Generation guardrail: enforce exact refusal string if it tries to hedge
    if "NOT FOUND IN PROVIDED DOCUMENTS" in draft.upper():
        print("\nNOT FOUND IN PROVIDED DOCUMENTS.")
        return

    if ENABLE_SELF_CHECK:
        verified = ollama_generate(self_check_prompt(question, context, draft))
        if "NOT FOUND IN PROVIDED DOCUMENTS" in verified.upper():
            print("\nNOT FOUND IN PROVIDED DOCUMENTS.")
            return
        print("\n" + verified)
    else:
        print("\n" + draft)


if __name__ == "__main__":
    main()
