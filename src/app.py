# src/app.py
import os
from pathlib import Path
import requests
import chromadb
import gradio as gr


# -----------------------------
# Paths / defaults
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "cholesterol_guidelines"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get(
    "OLLAMA_EMBED_MODEL",
    "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest",
)
GEN_MODEL_DEFAULT = os.environ.get("OLLAMA_GEN_MODEL", "gemma:latest")


# -----------------------------
# Ollama helpers
# -----------------------------
def ollama_embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def ollama_generate(model: str, prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=240,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


# -----------------------------
# Prompt / guardrails
# -----------------------------
def build_context(docs, metas, max_chars: int) -> str:
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


# -----------------------------
# Chroma client (lazy init)
# -----------------------------
_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found at {CHROMA_DIR}. Run ingest_chroma.py first."
        )

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


# -----------------------------
# Main RAG function
# -----------------------------
def rag_answer(
    question: str,
    gen_model: str,
    top_k: int,
    distance_threshold: float,
    max_context_chars: int,
    enable_self_check: bool,
    show_context: bool,
):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", ""

    col = get_collection()

    # embed query (must be 768-d to match stored embeddings)
    q_emb = ollama_embed(question)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # Sources display
    src_lines = []
    for i in range(min(len(docs), 10)):
        tag = f"{metas[i]['source']}#chunk{metas[i]['chunk_index']}"
        src_lines.append(f"{i+1}. distance={dists[i]:.4f} | {tag}")
    sources_text = "\n".join(src_lines)

    # retrieval guardrail
    best = dists[0] if dists else None
    if best is not None and float(best) > float(distance_threshold):
        return "NOT FOUND IN PROVIDED DOCUMENTS.", sources_text, ""

    context = build_context(docs, metas, int(max_context_chars))

    draft = ollama_generate(gen_model, strict_answer_prompt(question, context))

    # enforce refusal
    if "NOT FOUND IN PROVIDED DOCUMENTS" in draft.upper():
        return "NOT FOUND IN PROVIDED DOCUMENTS.", sources_text, (context if show_context else "")

    if enable_self_check:
        verified = ollama_generate(gen_model, self_check_prompt(question, context, draft))
        if "NOT FOUND IN PROVIDED DOCUMENTS" in verified.upper():
            return "NOT FOUND IN PROVIDED DOCUMENTS.", sources_text, (context if show_context else "")
        return verified, sources_text, (context if show_context else "")

    return draft, sources_text, (context if show_context else "")


# -----------------------------
# Gradio UI
# -----------------------------
def build_ui():
    with gr.Blocks(title="Local Cholesterol RAG (Ollama + Chroma)") as demo:
        gr.Markdown("# 🧠 Local RAG (Ollama + Chroma)\nAsk questions grounded in your ingested documents.")

        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="e.g., How can I lower LDL cholesterol with diet?",
                lines=2,
            )

        with gr.Row():
            gen_model = gr.Textbox(
                label="Ollama generation model",
                value=GEN_MODEL_DEFAULT,
                placeholder="e.g., gemma2:2b or qwen2.5:0.5b",
            )
            top_k = gr.Slider(3, 15, value=8, step=1, label="Top-K retrieved chunks")
            distance_threshold = gr.Slider(
                0.1, 2.0, value=0.75, step=0.05, label="Distance threshold (refuse above this)"
            )

        with gr.Row():
            max_context_chars = gr.Slider(
                1500, 12000, value=5000, step=250, label="Max context characters"
            )
            enable_self_check = gr.Checkbox(value=True, label="Enable self-check (reduces hallucinations)")
            show_context = gr.Checkbox(value=False, label="Show retrieved context (debug)")

        ask_btn = gr.Button("Ask", variant="primary")

        answer = gr.Textbox(label="Answer", lines=10)
        sources = gr.Textbox(label="Retrieved sources (debug)", lines=8)
        context_box = gr.Textbox(label="Context (optional)", lines=12, visible=True)

        ask_btn.click(
            fn=rag_answer,
            inputs=[question, gen_model, top_k, distance_threshold, max_context_chars, enable_self_check, show_context],
            outputs=[answer, sources, context_box],
        )

        gr.Markdown(
            "Tip: If it refuses too often, increase the distance threshold a bit (e.g., 0.9–1.2)."
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=bool(int(os.environ.get("GRADIO_SHARE", "0"))),
    )
