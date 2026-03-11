"""
rag_chroma.py — Core RAG pipeline for local healthcare QA system.
Uses ChromaDB for vector storage, BGE embeddings via Ollama, and Gemma for generation.
All computation is 100% local — no external API calls.
"""

import requests
import json
import chromadb
from chromadb.config import Settings


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"          # served via: ollama pull bge-base-en-v1.5
GENERATION_MODEL = "gemma:latest"          # served via: ollama pull gemma
CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "healthcare_guidelines"

DEFAULT_TOP_K = 8
DEFAULT_DISTANCE_THRESHOLD = 0.75
DEFAULT_MAX_CONTEXT_CHARS = 6000

NOT_FOUND_MSG = "NOT FOUND IN PROVIDED DOCUMENTS"

SYSTEM_PROMPT = """You are a grounded healthcare information assistant.

Rules you MUST follow (no exceptions):
1. Answer ONLY using the context blocks provided below.
2. You MUST cite EVERY sentence in your answer using the exact tag format [source#chunkN].
3. Every single factual claim needs a citation tag immediately after it.
4. If the context does not contain enough information, output exactly:
   NOT FOUND IN PROVIDED DOCUMENTS
5. Never write a sentence without a citation tag at the end of it.

Example of correct format:
LDL cholesterol builds up in artery walls and causes plaque [Cholesterol | MedlinePlus#chunk2]. 
Eating saturated fats raises LDL levels [How to Lower Cholesterol with Diet: MedlinePlus#chunk1].

Context blocks:
{context}
"""

VERIFIER_PROMPT = """You are a strict fact-checker for a medical RAG system.

Original question: {question}

Draft answer:
{answer}

Retrieved context:
{context}

Task: Check every factual claim in the draft answer against the retrieved context only.
- If ALL claims are fully supported by the context, reply with exactly: VERIFIED
- If ANY claim is not supported or is contradicted, reply with exactly:
  NOT FOUND IN PROVIDED DOCUMENTS

Reply with only one of those two options. No explanation."""


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Get BGE embedding from Ollama local server."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]


# ──────────────────────────────────────────────
# ChromaDB client
# ──────────────────────────────────────────────
def get_chroma_collection(collection_name: str = COLLECTION_NAME):
    """Return ChromaDB persistent collection (cosine similarity)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    collection_name: str = COLLECTION_NAME,
    topic_filter: str = None
) -> tuple[list[dict], float | None]:
    """
    Embed query and retrieve top-K chunks from ChromaDB.
    Returns (chunks, best_distance). chunks is empty if guardrail triggers.
    """
    collection = get_chroma_collection(collection_name)
    query_embedding = get_embedding(query)

    where_filter = None
    if topic_filter and topic_filter != "All Topics":
        where_filter = {"topic": topic_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where_filter
    )

    if not results["documents"][0]:
        return [], None

    best_distance = results["distances"][0][0]

    # Distance guardrail
    if best_distance > distance_threshold:
        return [], best_distance

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "topic": meta.get("topic", "general"),
            "distance": dist
        })

    return chunks, best_distance


# ──────────────────────────────────────────────
# Context formatting
# ──────────────────────────────────────────────
def format_context(chunks: list[dict], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    """Format retrieved chunks into tagged context blocks."""
    context_parts = []
    total_chars = 0
    for chunk in chunks:
        tag = f"[{chunk['source']}#chunk{chunk['chunk_index']}]"
        block = f"{tag}\n{chunk['text']}\n"
        if total_chars + len(block) > max_chars:
            break
        context_parts.append(block)
        total_chars += len(block)
    return "\n".join(context_parts)


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
def generate_answer(prompt: str, max_tokens: int = 512) -> str:
    """Call Gemma via Ollama for local generation."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1}
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ──────────────────────────────────────────────
# Self-check verifier
# ──────────────────────────────────────────────
def verify_answer(question: str, draft_answer: str, context: str) -> str:
    """Run second LLM pass to verify all claims are grounded in context."""
    if NOT_FOUND_MSG in draft_answer:
        return draft_answer

    verifier_input = VERIFIER_PROMPT.format(
        question=question,
        answer=draft_answer,
        context=context
    )
    verdict = generate_answer(verifier_input, max_tokens=64)

    if "VERIFIED" in verdict.upper():
        return draft_answer
    else:
        return NOT_FOUND_MSG


# ──────────────────────────────────────────────
# Full RAG pipeline
# ──────────────────────────────────────────────
def rag_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    use_self_check: bool = True,
    topic_filter: str = None,
    collection_name: str = COLLECTION_NAME
) -> dict:
    """
    Full RAG pipeline:
    1. Embed query
    2. Retrieve top-K chunks with distance guardrail
    3. Build grounded prompt
    4. Generate answer with Gemma
    5. (Optional) Self-check verification pass
    Returns dict with answer, chunks, distances, and metadata.
    """
    # Step 1 & 2: Retrieve
    chunks, best_distance = retrieve_chunks(
        query=question,
        top_k=top_k,
        distance_threshold=distance_threshold,
        collection_name=collection_name,
        topic_filter=topic_filter
    )

    # Guardrail triggered
    if not chunks:
        return {
            "answer": NOT_FOUND_MSG,
            "chunks": [],
            "best_distance": best_distance,
            "guardrail_triggered": True,
            "self_check_used": False,
            "context": ""
        }

    # Step 3: Format context
    context = format_context(chunks, max_context_chars)

    # Step 4: Generate
    full_prompt = SYSTEM_PROMPT.format(context=context) + f"\nQuestion: {question}\nAnswer:"
    draft_answer = generate_answer(full_prompt)

    # Step 5: Self-check
    final_answer = draft_answer
    self_check_used = False
    if use_self_check:
        final_answer = verify_answer(question, draft_answer, context)
        self_check_used = True

    return {
        "answer": final_answer,
        "chunks": chunks,
        "best_distance": best_distance,
        "guardrail_triggered": False,
        "self_check_used": self_check_used,
        "context": context
    }
