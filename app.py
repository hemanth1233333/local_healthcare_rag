"""
app.py — Gradio UI for local healthcare RAG system.
Run: python app.py
Access: http://localhost:7860
"""

import gradio as gr
from src.rag_chroma import rag_query, get_chroma_collection, NOT_FOUND_MSG

TOPIC_OPTIONS = ["All Topics", "cholesterol", "thyroid", "ckd", "asthma", "anemia"]


def get_collection_info() -> str:
    """Return collection stats string for display."""
    try:
        collection = get_chroma_collection()
        total = collection.count()
        return f"✅ ChromaDB connected — {total} chunks indexed"
    except Exception as e:
        return f"❌ ChromaDB error: {e}"


def run_query(
    question: str,
    topic_filter: str,
    top_k: int,
    distance_threshold: float,
    max_context_chars: int,
    use_self_check: bool,
    show_context: bool
) -> tuple[str, str, str]:
    """
    Main query handler for Gradio interface.
    Returns (answer_text, sources_text, debug_text).
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        result = rag_query(
            question=question,
            top_k=int(top_k),
            distance_threshold=distance_threshold,
            max_context_chars=int(max_context_chars),
            use_self_check=use_self_check,
            topic_filter=topic_filter if topic_filter != "All Topics" else None
        )
    except Exception as e:
        return f"❌ Pipeline error: {str(e)}", "", ""

    answer = result["answer"]
    chunks = result["chunks"]
    best_dist = result["best_distance"]
    guardrail = result["guardrail_triggered"]

    # Format answer with privacy badge
    if NOT_FOUND_MSG in answer:
        answer_display = f"⚠️ {NOT_FOUND_MSG}\n\nThis query is outside the scope of the indexed documents."
    else:
        answer_display = answer

    # Format sources
    if not chunks:
        if guardrail:
            sources_display = f"🛡️ Distance guardrail triggered (best distance: {best_dist:.4f} > threshold {distance_threshold:.2f})\nNo sources retrieved."
        else:
            sources_display = "No sources retrieved."
    else:
        source_lines = [f"Retrieved {len(chunks)} chunks (best cosine distance: {best_dist:.4f})\n"]
        for i, chunk in enumerate(chunks, 1):
            source_lines.append(
                f"[{i}] {chunk['source']}#chunk{chunk['chunk_index']} "
                f"| topic: {chunk['topic']} "
                f"| distance: {chunk['distance']:.4f}"
            )
        sources_display = "\n".join(source_lines)

    # Debug / context display
    debug_display = ""
    if show_context and result["context"]:
        debug_display = f"── Retrieved Context ──\n\n{result['context']}"
    elif show_context:
        debug_display = "No context retrieved."

    debug_display += f"\n\n── Pipeline Metadata ──\n"
    debug_display += f"Self-check used: {result['self_check_used']}\n"
    debug_display += f"Guardrail triggered: {result['guardrail_triggered']}\n"
    debug_display += f"Chunks retrieved: {len(chunks)}\n"
    if best_dist is not None:
        debug_display += f"Best cosine distance: {best_dist:.4f}\n"

    return answer_display, sources_display, debug_display


# ──────────────────────────────────────────────
# Build Gradio interface
# ──────────────────────────────────────────────
with gr.Blocks(
    title="Strata Mind — Local Healthcare RAG",
    theme=gr.themes.Soft(),
    css=".privacy-badge { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 8px; }"
) as demo:

    gr.Markdown("""
    # 🏥 Strata Mind — Secure Local Healthcare RAG
    **100% local inference. No cloud calls. No data leaves your machine.**
    
    Ask questions about: Cholesterol | Thyroid | CKD | Asthma/COPD | Anemia
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your Health Question",
                placeholder="e.g. What foods should I avoid to lower LDL cholesterol?",
                lines=2
            )
            topic_filter = gr.Dropdown(
                choices=TOPIC_OPTIONS,
                value="All Topics",
                label="Filter by Topic (optional)"
            )

        with gr.Column(scale=1):
            db_status = gr.Textbox(
                label="Database Status",
                value=get_collection_info(),
                interactive=False
            )
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

    with gr.Accordion("⚙️ Advanced Settings", open=False):
        with gr.Row():
            top_k_slider = gr.Slider(
                minimum=3, maximum=15, value=8, step=1,
                label="Top-K chunks retrieved"
            )
            distance_slider = gr.Slider(
                minimum=0.3, maximum=1.0, value=0.75, step=0.05,
                label="Distance threshold (lower = stricter)"
            )
            context_slider = gr.Slider(
                minimum=2000, maximum=12000, value=6000, step=500,
                label="Max context characters"
            )
        with gr.Row():
            self_check_toggle = gr.Checkbox(value=True, label="Enable self-check verification pass")
            show_context_toggle = gr.Checkbox(value=False, label="Show retrieved context (debug)")

    submit_btn = gr.Button("🔍 Ask", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Answer (grounded, cited)",
                lines=10,
                interactive=False
            )
        with gr.Column(scale=1):
            sources_output = gr.Textbox(
                label="Retrieved Sources",
                lines=10,
                interactive=False
            )

    debug_output = gr.Textbox(
        label="Debug / Context",
        lines=8,
        interactive=False,
        visible=True
    )

    gr.Markdown("""
    ---
    🔒 **Privacy Guarantee**: All embeddings and generation run locally via Ollama.
    No query text, no retrieved content, and no answers are transmitted to any external server.
    """)

    # Wire up events
    submit_btn.click(
        fn=run_query,
        inputs=[
            question_input, topic_filter, top_k_slider,
            distance_slider, context_slider, self_check_toggle, show_context_toggle
        ],
        outputs=[answer_output, sources_output, debug_output]
    )

    question_input.submit(
        fn=run_query,
        inputs=[
            question_input, topic_filter, top_k_slider,
            distance_slider, context_slider, self_check_toggle, show_context_toggle
        ],
        outputs=[answer_output, sources_output, debug_output]
    )

    refresh_btn.click(
        fn=get_collection_info,
        outputs=db_status
    )

    # Example questions
    gr.Examples(
        examples=[
            ["What foods should I avoid to lower LDL cholesterol?", "cholesterol"],
            ["What are the symptoms of hypothyroidism?", "thyroid"],
            ["How is chronic kidney disease diagnosed?", "ckd"],
            ["What triggers an asthma attack?", "asthma"],
            ["What are the causes of iron deficiency anemia?", "anemia"],
            ["Can high cholesterol cause heart disease?", "cholesterol"],
        ],
        inputs=[question_input, topic_filter],
        label="Example Questions"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
