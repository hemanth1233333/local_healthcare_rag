"""
security_eval.py — Membership Inference Attack Evaluation for local RAG system.
Tests whether retrieval confidence signals reveal whether a document is in the index.

Attack: Probe with exact phrases from indexed documents (members) vs. phrases NOT in the
        index (non-members). If cosine distances are significantly lower for members,
        the system leaks document membership.

Defense: Local-only inference + distance guardrail + NOT FOUND fallback.

Usage:
    python security/security_eval.py --output security/security_results.json
    python security/security_eval.py --probe-set security/membership_probes.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag_chroma import retrieve_chunks, get_chroma_collection

DEFAULT_DISTANCE_THRESHOLD = 0.75


# ──────────────────────────────────────────────
# Load probe set
# ──────────────────────────────────────────────
def load_probes(probe_path: str) -> list[dict]:
    with open(probe_path) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Run single probe
# ──────────────────────────────────────────────
def run_probe(probe: dict, top_k: int = 5) -> dict:
    """
    Run a single membership inference probe.
    Returns distances and whether guardrail was triggered.
    """
    query = probe["query"]
    is_member = probe["is_member"]  # True = phrase IS in the index
    expected_source = probe.get("expected_source", None)

    chunks, best_dist = retrieve_chunks(
        query=query,
        top_k=top_k,
        distance_threshold=1.0,  # no guardrail — we want raw distances for analysis
    )

    if not chunks:
        return {
            "query": query,
            "is_member": is_member,
            "expected_source": expected_source,
            "best_distance": None,
            "guardrail_would_trigger": True,
            "top_sources": [],
            "leakage_detected": False,
            "description": probe.get("description", "")
        }

    distances = [c["distance"] for c in chunks]
    best = distances[0]
    top_sources = [f"{c['source']}#chunk{c['chunk_index']}" for c in chunks[:3]]

    # Guardrail simulation
    guardrail_would_trigger = best > DEFAULT_DISTANCE_THRESHOLD

    # Leakage detection: if member query returns very low distance (< 0.3),
    # the attacker can infer membership
    LEAKAGE_THRESHOLD = 0.30
    leakage_detected = is_member and best < LEAKAGE_THRESHOLD

    return {
        "query": query,
        "is_member": is_member,
        "expected_source": expected_source,
        "best_distance": round(best, 6),
        "all_distances": [round(d, 6) for d in distances[:5]],
        "guardrail_would_trigger": guardrail_would_trigger,
        "top_sources": top_sources,
        "leakage_detected": leakage_detected,
        "description": probe.get("description", "")
    }


# ──────────────────────────────────────────────
# Analyze results
# ──────────────────────────────────────────────
def analyze_results(probe_results: list[dict]) -> dict:
    """
    Statistical analysis of membership inference results.
    Key question: Are member distances significantly lower than non-member distances?
    """
    member_dists = [r["best_distance"] for r in probe_results
                    if r["is_member"] and r["best_distance"] is not None]
    non_member_dists = [r["best_distance"] for r in probe_results
                        if not r["is_member"] and r["best_distance"] is not None]

    def stats(dists):
        if not dists:
            return {}
        return {
            "count": len(dists),
            "mean": round(sum(dists) / len(dists), 4),
            "min": round(min(dists), 4),
            "max": round(max(dists), 4),
        }

    member_stats = stats(member_dists)
    non_member_stats = stats(non_member_dists)

    # Attack success: how often could attacker correctly identify members?
    # Assume attacker uses threshold: dist < 0.4 → "in index"
    attack_threshold = 0.40
    true_positives = sum(1 for d in member_dists if d < attack_threshold)
    false_positives = sum(1 for d in non_member_dists if d < attack_threshold)
    true_negatives = sum(1 for d in non_member_dists if d >= attack_threshold)
    false_negatives = sum(1 for d in member_dists if d >= attack_threshold)

    n_members = len(member_dists)
    n_non_members = len(non_member_dists)

    attack_accuracy = (
        (true_positives + true_negatives) / (n_members + n_non_members)
        if (n_members + n_non_members) > 0 else 0
    )
    attack_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    attack_recall = true_positives / n_members if n_members > 0 else 0

    # Leakage count
    leakage_count = sum(1 for r in probe_results if r.get("leakage_detected", False))

    # Guardrail effectiveness
    guardrail_blocked_non_member = sum(
        1 for r in probe_results
        if not r["is_member"] and r.get("guardrail_would_trigger", False)
    )

    distance_gap = (
        non_member_stats.get("mean", 0) - member_stats.get("mean", 0)
        if member_stats and non_member_stats else 0
    )

    return {
        "member_distance_stats": member_stats,
        "non_member_distance_stats": non_member_stats,
        "distance_gap_member_vs_non_member": round(distance_gap, 4),
        "attack_metrics": {
            "threshold_used": attack_threshold,
            "accuracy": round(attack_accuracy, 4),
            "precision": round(attack_precision, 4),
            "recall": round(attack_recall, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        },
        "leakage_detected_count": leakage_count,
        "guardrail_blocked_non_member_queries": guardrail_blocked_non_member,
        "interpretation": interpret_results(attack_accuracy, distance_gap, leakage_count)
    }


def interpret_results(accuracy: float, distance_gap: float, leakage_count: int) -> str:
    """Human-readable interpretation of membership inference results."""
    if accuracy > 0.8 and distance_gap > 0.3:
        return ("HIGH RISK: Attack achieves high accuracy. "
                "Distance signals significantly distinguish members from non-members. "
                "Consider additional defenses (distance noise, response obfuscation).")
    elif accuracy > 0.6 or distance_gap > 0.15:
        return ("MODERATE RISK: Attack has some success. "
                "Distance gap between members and non-members is detectable. "
                "Guardrail provides partial protection.")
    else:
        return ("LOW RISK: Attack cannot reliably distinguish members from non-members. "
                "Local-only inference and distance guardrail provide effective defense. "
                "MedlinePlus public documents limit PHI leakage risk.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def run_security_evaluation(probe_path: str, output_path: str):
    probes = load_probes(probe_path)
    print(f"\n{'='*60}")
    print(f"Security Evaluation — Membership Inference Attack")
    print(f"{'='*60}")
    print(f"Probes: {len(probes)} total "
          f"({sum(1 for p in probes if p['is_member'])} members, "
          f"{sum(1 for p in probes if not p['is_member'])} non-members)")

    results = []
    for i, probe in enumerate(probes):
        print(f"[{i+1}/{len(probes)}] {'MEMBER' if probe['is_member'] else 'NON-MEMBER'}: {probe['query'][:70]}...")
        result = run_probe(probe)
        results.append(result)
        dist_str = f"{result['best_distance']:.4f}" if result['best_distance'] is not None else "N/A"
        guard_str = "🛡️ BLOCKED" if result["guardrail_would_trigger"] else "retrieved"
        leak_str = " ⚠️ LEAKAGE" if result.get("leakage_detected") else ""
        print(f"         dist={dist_str} | guardrail={guard_str}{leak_str}")

    analysis = analyze_results(results)

    output = {
        "evaluation_date": datetime.now().isoformat(),
        "n_probes": len(probes),
        "n_members": sum(1 for p in probes if p["is_member"]),
        "n_non_members": sum(1 for p in probes if not p["is_member"]),
        "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
        "analysis": analysis,
        "probe_results": results
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("SECURITY EVALUATION SUMMARY")
    print(f"{'='*60}")
    a = analysis
    print(f"Member distances:     mean={a['member_distance_stats'].get('mean','N/A')}, "
          f"min={a['member_distance_stats'].get('min','N/A')}")
    print(f"Non-member distances: mean={a['non_member_distance_stats'].get('mean','N/A')}, "
          f"min={a['non_member_distance_stats'].get('min','N/A')}")
    print(f"Distance gap:         {a['distance_gap_member_vs_non_member']}")
    print(f"Attack accuracy:      {a['attack_metrics']['accuracy']}")
    print(f"Leakage events:       {a['leakage_detected_count']}")
    print(f"\n{a['interpretation']}")
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run membership inference security evaluation")
    parser.add_argument("--probe-set", default="security/membership_probes.json",
                        help="Path to membership probe JSON")
    parser.add_argument("--output", default="security/security_results.json",
                        help="Output path for results")
    args = parser.parse_args()
    run_security_evaluation(args.probe_set, args.output)


if __name__ == "__main__":
    main()
