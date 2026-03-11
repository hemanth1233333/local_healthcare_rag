"""
evaluate.py — Checkpoint 1 evaluation: Precision@5, RAGAs faithfulness, citation coverage.
Compares RAG pipeline vs. vanilla Gemma baseline.

Usage:
    # Run full evaluation
    python eval/evaluate.py --eval-set eval/cholesterol_eval_set.json --output eval/results.json

    # Quick smoke test (5 questions)
    python eval/evaluate.py --eval-set eval/cholesterol_eval_set.json --limit 5
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import requests

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "gemma:latest"
NOT_FOUND_MSG = "NOT FOUND IN PROVIDED DOCUMENTS"


# ──────────────────────────────────────────────
# Imports from src
# ──────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag_chroma import rag_query, retrieve_chunks


# ──────────────────────────────────────────────
# Vanilla Gemma baseline (no retrieval)
# ──────────────────────────────────────────────
def vanilla_gemma_answer(question: str) -> str:
    """Answer using Gemma with no RAG context — baseline comparison."""
    prompt = f"""You are a helpful medical information assistant.
Answer the following health question based on your general knowledge.
Be factual and concise.

Question: {question}
Answer:"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 300, "temperature": 0.1}
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ──────────────────────────────────────────────
# Metric 1: Retrieval Precision@K
# ──────────────────────────────────────────────
def compute_precision_at_k(
    question: str,
    relevant_keywords: list[str],
    k: int = 5,
    topic_filter: str = None
) -> dict:
    """
    Manual Precision@K: retrieve top-K chunks and check how many contain
    at least one relevant keyword from the expected answer keywords.

    Returns dict with precision score and chunk details.
    """
    chunks, best_dist = retrieve_chunks(
        query=question,
        top_k=k,
        distance_threshold=1.0,  # no guardrail for eval
        topic_filter=topic_filter
    )

    if not chunks:
        return {"precision_at_k": 0.0, "relevant_count": 0, "total": 0, "chunks": []}

    eval_k = min(k, len(chunks))
    top_chunks = chunks[:eval_k]
    relevant_count = 0
    chunk_results = []

    for chunk in top_chunks:
        chunk_text_lower = chunk["text"].lower()
        is_relevant = any(kw.lower() in chunk_text_lower for kw in relevant_keywords)
        if is_relevant:
            relevant_count += 1
        chunk_results.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "distance": chunk["distance"],
            "is_relevant": is_relevant,
            "matched_keywords": [kw for kw in relevant_keywords if kw.lower() in chunk_text_lower]
        })

    precision = relevant_count / eval_k if eval_k > 0 else 0.0

    return {
        "precision_at_k": precision,
        "relevant_count": relevant_count,
        "total_retrieved": eval_k,
        "best_distance": best_dist,
        "chunks": chunk_results
    }


# ──────────────────────────────────────────────
# Metric 2: Citation Coverage
# ──────────────────────────────────────────────
def check_citation_coverage(answer: str) -> dict:
    """Check if answer contains at least one citation tag."""
    import re
    # Match [anything#chunkN] — handles spaces, colons, pipes, quotes in source names
    citations = re.findall(r'\[[^\]]+#chunk\d+\]', answer)
    has_citation = len(citations) > 0
    return {
        "has_citation": has_citation,
        "citation_count": len(citations),
        "citations": citations
    }


# ──────────────────────────────────────────────
# Metric 3: RAGAs-style Faithfulness (LLM-as-judge)
# ──────────────────────────────────────────────
def compute_faithfulness_llm(answer: str, context: str) -> dict:
    """
    LLM-as-judge faithfulness score (0.0 to 1.0).
    Prompts Gemma to rate how grounded the answer is in the context.
    Returns score and raw verdict.
    """
    if NOT_FOUND_MSG in answer or not answer.strip():
        return {"faithfulness_score": 1.0, "verdict": "NOT_FOUND_RESPONSE", "raw": ""}

    judge_prompt = f"""You are evaluating whether an AI answer is faithful to retrieved context.

Context:
{context[:3000]}

Answer to evaluate:
{answer[:1000]}

Rate the faithfulness of the answer on a scale from 0.0 to 1.0:
- 1.0 = Every claim is directly supported by the context
- 0.75 = Most claims supported, minor gaps
- 0.5 = About half the claims are supported
- 0.25 = Few claims are supported
- 0.0 = Answer contradicts or ignores the context

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": GENERATION_MODEL,
                "prompt": judge_prompt,
                "stream": False,
                "options": {"num_predict": 10, "temperature": 0.0}
            },
            timeout=60
        )
        raw = response.json()["response"].strip()
        # Parse the score
        import re
        match = re.search(r'\b(0\.\d+|1\.0|0|1)\b', raw)
        score = float(match.group()) if match else 0.5
        score = max(0.0, min(1.0, score))
    except Exception as e:
        print(f"  [faithfulness] Error: {e}")
        score = 0.0
        raw = str(e)

    return {"faithfulness_score": score, "verdict": "SCORED", "raw": raw}


# ──────────────────────────────────────────────
# Run evaluation on full eval set
# ──────────────────────────────────────────────
def run_evaluation(
    eval_set: list[dict],
    limit: int = None,
    k: int = 5,
    verbose: bool = True
) -> dict:
    """
    Run Precision@5, faithfulness, and citation coverage on eval set.
    Compare RAG vs vanilla Gemma.
    """
    if limit:
        eval_set = eval_set[:limit]

    results = []
    rag_faithfulness_scores = []
    vanilla_faithfulness_scores = []
    precision_scores = []
    citation_coverage_count = 0
    not_found_count = 0

    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(eval_set)} questions")
    print(f"{'='*60}\n")

    for i, item in enumerate(eval_set):
        question = item["question"]
        relevant_keywords = item.get("relevant_keywords", [])
        topic = item.get("topic", None)

        print(f"[{i+1}/{len(eval_set)}] {question[:80]}...")

        # ── RAG answer ──
        try:
            rag_result = rag_query(
                question=question,
                top_k=k,
                distance_threshold=0.75,
                use_self_check=False,  # disable for speed in eval
                topic_filter=topic
            )
            rag_answer = rag_result["answer"]
            rag_context = rag_result["context"]
            rag_chunks = rag_result["chunks"]
        except Exception as e:
            print(f"  [RAG ERROR] {e}")
            rag_answer = ""
            rag_context = ""
            rag_chunks = []

        # ── Vanilla Gemma answer ──
        try:
            vanilla_answer = vanilla_gemma_answer(question)
        except Exception as e:
            print(f"  [VANILLA ERROR] {e}")
            vanilla_answer = ""

        # ── Metrics ──
        precision_result = compute_precision_at_k(question, relevant_keywords, k, topic)
        citation_result = check_citation_coverage(rag_answer)
        rag_faith = compute_faithfulness_llm(rag_answer, rag_context)
        vanilla_faith = compute_faithfulness_llm(vanilla_answer, rag_context)  # same context for fair comparison

        is_not_found = NOT_FOUND_MSG in rag_answer

        precision_scores.append(precision_result["precision_at_k"])
        rag_faithfulness_scores.append(rag_faith["faithfulness_score"])
        vanilla_faithfulness_scores.append(vanilla_faith["faithfulness_score"])
        if citation_result["has_citation"]:
            citation_coverage_count += 1
        if is_not_found:
            not_found_count += 1

        result_item = {
            "question": question,
            "topic": topic,
            "rag_answer": rag_answer,
            "vanilla_answer": vanilla_answer,
            "precision_at_5": precision_result["precision_at_k"],
            "rag_faithfulness": rag_faith["faithfulness_score"],
            "vanilla_faithfulness": vanilla_faith["faithfulness_score"],
            "citation_coverage": citation_result["has_citation"],
            "citation_count": citation_result["citation_count"],
            "not_found_response": is_not_found,
            "chunks_retrieved": len(rag_chunks),
            "best_distance": rag_result.get("best_distance") if rag_result else None,
            "precision_detail": precision_result
        }
        results.append(result_item)

        if verbose:
            print(f"  Precision@{k}: {precision_result['precision_at_k']:.2f} | "
                  f"RAG Faith: {rag_faith['faithfulness_score']:.2f} | "
                  f"Vanilla Faith: {vanilla_faith['faithfulness_score']:.2f} | "
                  f"Citation: {'✓' if citation_result['has_citation'] else '✗'}")

        time.sleep(0.5)  # small delay to avoid overwhelming Ollama

    # ── Aggregate metrics ──
    n = len(eval_set)
    avg_precision = sum(precision_scores) / n if n > 0 else 0
    avg_rag_faith = sum(rag_faithfulness_scores) / n if n > 0 else 0
    avg_vanilla_faith = sum(vanilla_faithfulness_scores) / n if n > 0 else 0
    citation_coverage_pct = citation_coverage_count / n * 100 if n > 0 else 0

    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "n_questions": n,
        "k": k,
        "targets": {
            "precision_at_5_target": 0.70,
            "faithfulness_target": 0.75,
            "citation_coverage_target": 0.80
        },
        "results": {
            "avg_precision_at_5": round(avg_precision, 4),
            "avg_rag_faithfulness": round(avg_rag_faith, 4),
            "avg_vanilla_faithfulness": round(avg_vanilla_faith, 4),
            "faithfulness_improvement_over_vanilla": round(avg_rag_faith - avg_vanilla_faith, 4),
            "citation_coverage_pct": round(citation_coverage_pct, 1),
            "not_found_responses": not_found_count,
            "not_found_pct": round(not_found_count / n * 100, 1) if n > 0 else 0
        },
        "target_met": {
            "precision_at_5": avg_precision >= 0.70,
            "faithfulness": avg_rag_faith >= 0.75,
            "citation_coverage": citation_coverage_pct >= 80.0
        },
        "per_question_results": results
    }

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Questions evaluated:          {n}")
    print(f"")
    print(f"Precision@{k}:                {avg_precision:.4f}  (target ≥ 0.70) {'✅' if avg_precision >= 0.70 else '❌'}")
    print(f"RAG Faithfulness:             {avg_rag_faith:.4f}  (target ≥ 0.75) {'✅' if avg_rag_faith >= 0.75 else '❌'}")
    print(f"Vanilla Faithfulness:         {avg_vanilla_faith:.4f}")
    print(f"Faithfulness improvement:     +{avg_rag_faith - avg_vanilla_faith:.4f}")
    print(f"Citation Coverage:            {citation_coverage_pct:.1f}%   (target ≥ 80%)  {'✅' if citation_coverage_pct >= 80 else '❌'}")
    print(f"NOT FOUND responses:          {not_found_count}/{n} ({not_found_count/n*100:.1f}%)")
    print(f"{'='*60}\n")

    return summary


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate local healthcare RAG system")
    parser.add_argument("--eval-set", default="eval/cholesterol_eval_set.json",
                        help="Path to JSON eval set")
    parser.add_argument("--output", default="eval/results.json",
                        help="Output path for results JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions (for quick testing)")
    parser.add_argument("--k", type=int, default=5,
                        help="K for Precision@K (default: 5)")
    args = parser.parse_args()

    with open(args.eval_set, "r") as f:
        eval_set = json.load(f)

    summary = run_evaluation(eval_set, limit=args.limit, k=args.k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
