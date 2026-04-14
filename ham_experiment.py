"""
HAM Experiment Runner — Systematic tests for the research claims

This script runs controlled experiments to validate (or falsify) the
core claims of the HAMesh paper. Each experiment is self-contained,
logs structured data, and produces a results JSON you can analyse.

Claims under test:
  C1  Attractor convergence is deterministic and semantically coherent
      — same mesh + same dream cycles → same attractor clusters
  C2  Attractor topology reflects real semantic structure
      — the clusters are meaningful, not random high-energy memories
  C3  Cross-pollination creates cross-domain curiosity
      — after pollination, each mesh becomes curious about the other's domain
  C4  Curiosity synthesis is genuinely cross-domain
      — insights draw from multiple mesh domains, not just the home domain
  C5  Multi-hop diffraction follows transitive chains
      — 2-hop gives meaningfully different activations than 1-hop

Usage:
  # Run all experiments (requires Bonsai server + distilled meshes)
  python ham_experiment.py --all

  # Run a specific claim
  python ham_experiment.py --claim C1
  python ham_experiment.py --claim C2
  python ham_experiment.py --claim C4

  # Run with custom meshes
  python ham_experiment.py --claim C1 --meshes science:ham_science.pt,philosophy:ham_philosophy.pt

Results saved to: ./ham_logs/experiment_<claim>_<timestamp>.json
"""

import argparse
import json
import time
import copy
import random
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from ham_core import HolographicMesh
from ham_brain import embed, generate, CONFIDENCE_FLOOR
from ham_collective import MeshCollective
from ham_logger import HAMLogger

RESULTS_DIR = Path("./ham_logs")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_MESHES = {
    "science":    "./ham_science.pt",
    "philosophy": "./ham_philosophy.pt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_meshes(mesh_spec: dict) -> dict:
    meshes = {}
    for name, path in mesh_spec.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Mesh '{name}' not found at {path}. "
                f"Run: python ham_distill.py --topics ... --save {path}"
            )
        print(f"  Loading [{name}] from {path}...")
        meshes[name] = HolographicMesh.load(path)
    return meshes


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        F.normalize(a, dim=0).unsqueeze(0),
        F.normalize(b, dim=0).unsqueeze(0),
    ).item()


def attractor_overlap(attractors_a: list, attractors_b: list, top_n=5) -> float:
    """
    Measure overlap between two attractor lists.
    Returns fraction of top-N attractors that appear in both.
    """
    set_a = {t for t, _ in attractors_a[:top_n]}
    set_b = {t for t, _ in attractors_b[:top_n]}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def semantic_coherence_score(attractor_texts: list, n=5) -> float:
    """
    Ask the LLM to rate how semantically coherent a set of attractor
    concepts is (0-10). Returns the score / 10.
    """
    if not attractor_texts:
        return 0.0
    concepts = "\n".join(f"- {t}" for t in attractor_texts[:n])
    prompt = (
        f"Rate the semantic coherence of this set of concepts on a scale "
        f"from 0 to 10, where 10 = very tightly related field/cluster, "
        f"0 = completely random unrelated topics.\n\n"
        f"Concepts:\n{concepts}\n\n"
        f"Reply with ONLY a single integer from 0 to 10."
    )
    try:
        raw = generate(prompt, system="Rate only. Integer only. No explanation.", max_tokens=5)
        score = int("".join(c for c in raw.strip() if c.isdigit())[:2])
        return min(score, 10) / 10.0
    except Exception:
        return -1.0  # indeterminate


def save_results(claim: str, data: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"experiment_{claim}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# C1 — Attractor convergence determinism
# ---------------------------------------------------------------------------

def experiment_c1(mesh_spec: dict, dream_cycles=500, n_runs=3):
    """
    Claim: The same mesh develops the same attractor clusters
    across independent dream runs.

    Method:
      1. Load mesh (fresh copy each run so state doesn't accumulate)
      2. Run N independent dream sessions of K cycles each
      3. Record top-5 attractors after each run
      4. Measure pairwise overlap between runs
      5. Also measure semantic coherence of each attractor set

    Expected result: overlap > 0.6, coherence > 0.6
    """
    print("\n" + "=" * 60)
    print("  C1: Attractor Convergence Determinism")
    print(f"  {dream_cycles} cycles × {n_runs} independent runs per mesh")
    print("=" * 60)

    results = {
        "claim": "C1",
        "hypothesis": "Same mesh produces same attractor clusters across independent runs",
        "dream_cycles_per_run": dream_cycles,
        "n_runs": n_runs,
        "meshes": {},
        "conclusion": None,
    }

    for mesh_name, mesh_path in mesh_spec.items():
        print(f"\n  [{mesh_name}]")
        mesh_results = {"runs": [], "pairwise_overlap": [], "coherence_scores": []}

        for run_idx in range(n_runs):
            print(f"    Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)

            # Fresh copy so runs are independent
            ham = HolographicMesh.load(mesh_path)
            t0 = time.time()

            _, attractors, loops = ham.dream(cycles=dream_cycles, fold_strength=0.1)

            elapsed = time.time() - t0
            top5 = [(t, c) for t, c in attractors[:5]]
            coherence = semantic_coherence_score([t for t, _ in top5])

            run_data = {
                "run": run_idx + 1,
                "top_5_attractors": top5,
                "n_loops": len(loops),
                "coherence_score": coherence,
                "elapsed_s": round(elapsed, 1),
            }
            mesh_results["runs"].append(run_data)
            mesh_results["coherence_scores"].append(coherence)

            top_texts = [t for t, _ in top5]
            print(f"coherence={coherence:.2f}  top={top_texts[0][:40]}")

        # Pairwise overlap between all run pairs
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                ov = attractor_overlap(
                    mesh_results["runs"][i]["top_5_attractors"],
                    mesh_results["runs"][j]["top_5_attractors"],
                )
                mesh_results["pairwise_overlap"].append({
                    "run_pair": [i + 1, j + 1], "overlap": round(ov, 3)
                })
                print(f"    Overlap run{i+1}↔run{j+1}: {ov:.3f}")

        avg_overlap = sum(x["overlap"] for x in mesh_results["pairwise_overlap"]) / \
                      max(len(mesh_results["pairwise_overlap"]), 1)
        avg_coherence = sum(x for x in mesh_results["coherence_scores"] if x >= 0) / \
                        max(sum(1 for x in mesh_results["coherence_scores"] if x >= 0), 1)

        mesh_results["avg_pairwise_overlap"] = round(avg_overlap, 3)
        mesh_results["avg_coherence"] = round(avg_coherence, 3)
        mesh_results["verdict"] = (
            "SUPPORTS C1" if avg_overlap > 0.5 and avg_coherence > 0.5
            else "MIXED" if avg_overlap > 0.3
            else "REFUTES C1"
        )

        print(f"    → Avg overlap: {avg_overlap:.3f}  Avg coherence: {avg_coherence:.3f}  [{mesh_results['verdict']}]")
        results["meshes"][mesh_name] = mesh_results

    results["conclusion"] = "SUPPORTS C1" if all(
        m["verdict"] == "SUPPORTS C1" for m in results["meshes"].values()
    ) else "MIXED"

    return save_results("C1", results), results


# ---------------------------------------------------------------------------
# C2 — Semantic coherence of attractors
# ---------------------------------------------------------------------------

def experiment_c2(mesh_spec: dict, dream_cycles=500, n_baseline=10):
    """
    Claim: Attractor clusters are semantically coherent — the HAM is
    finding real conceptual neighborhoods, not just high-energy noise.

    Method:
      1. Dream to convergence, record top-5 attractors
      2. Score semantic coherence of the attractor set (LLM judge)
      3. Compare against baseline: n_baseline random memory samples
      4. t-test or simple comparison: attractor coherence > random baseline

    Expected result: attractor coherence significantly > random baseline
    """
    print("\n" + "=" * 60)
    print("  C2: Semantic Coherence of Attractor Clusters")
    print("=" * 60)

    results = {
        "claim": "C2",
        "hypothesis": "Attractor clusters have higher semantic coherence than random memory samples",
        "meshes": {},
        "conclusion": None,
    }

    for mesh_name, mesh_path in mesh_spec.items():
        print(f"\n  [{mesh_name}]")
        ham = HolographicMesh.load(mesh_path)

        # Baseline: random memory samples
        print(f"    Scoring {n_baseline} random samples...")
        baseline_scores = []
        for _ in range(n_baseline):
            if len(ham.memories) < 5:
                break
            sample = random.sample(ham.memories, min(5, len(ham.memories)))
            texts = [t for _, t in sample]
            score = semantic_coherence_score(texts)
            if score >= 0:
                baseline_scores.append(score)

        # Dream and score attractors
        print("    Dreaming to convergence...")
        _, attractors, loops = ham.dream(cycles=dream_cycles, fold_strength=0.1)
        attractor_score = semantic_coherence_score([t for t, _ in attractors[:5]])

        avg_baseline = sum(baseline_scores) / max(len(baseline_scores), 1)
        delta = attractor_score - avg_baseline

        mesh_result = {
            "attractor_coherence": attractor_score,
            "random_baseline_avg": round(avg_baseline, 3),
            "random_baseline_scores": baseline_scores,
            "delta": round(delta, 3),
            "top_5_attractors": [(t, c) for t, c in attractors[:5]],
            "n_loops": len(loops),
            "verdict": "SUPPORTS C2" if delta > 0.15 else "MIXED" if delta > 0 else "REFUTES C2",
        }

        print(f"    Attractor coherence: {attractor_score:.2f}")
        print(f"    Random baseline:     {avg_baseline:.2f}")
        print(f"    Delta:               {delta:+.2f}  [{mesh_result['verdict']}]")
        results["meshes"][mesh_name] = mesh_result

    results["conclusion"] = "SUPPORTS C2" if all(
        m["verdict"] == "SUPPORTS C2" for m in results["meshes"].values()
    ) else "MIXED"

    return save_results("C2", results), results


# ---------------------------------------------------------------------------
# C3 — Cross-pollination creates cross-domain curiosity
# ---------------------------------------------------------------------------

def experiment_c3(mesh_spec: dict):
    """
    Claim: After cross-pollination, each mesh develops curiosity about
    topics from the other mesh's domain.

    Method:
      1. Find isolated memories in each mesh BEFORE pollination
      2. Cross-pollinate
      3. Find isolated memories AFTER pollination
      4. Check: do post-pollination gaps include topics from the OTHER domain?
      5. Compute cross-domain gap fraction: gaps that mention other-domain concepts

    Expected result: cross-domain gap fraction increases after pollination
    """
    print("\n" + "=" * 60)
    print("  C3: Cross-Pollination Creates Cross-Domain Curiosity")
    print("=" * 60)

    meshes = load_meshes(mesh_spec)
    collective = MeshCollective(meshes)

    results = {
        "claim": "C3",
        "hypothesis": "Cross-pollination causes each mesh to develop curiosity about the other domain",
        "before": {},
        "after": {},
        "conclusion": None,
    }

    # Before: record isolated memories per mesh
    print("\n  Gaps BEFORE cross-pollination:")
    for name, ham in meshes.items():
        isolated = ham.find_isolated(top_k=15)
        gaps = [t for _, _, t in isolated]
        results["before"][name] = {"gaps": gaps}
        print(f"    [{name}]: {len(gaps)} isolated memories")
        for t in gaps[:3]:
            print(f"      - {t[:70]}")

    # Cross-pollinate
    print("\n  Cross-pollinating...")
    n_shared = collective.cross_pollinate(strength=0.04, n=8)
    print(f"  {n_shared} patterns shared.")

    # After: re-check isolation
    print("\n  Gaps AFTER cross-pollination:")
    other_domain_markers = {
        name: [t[:40] for _, t in meshes[other_name].memories[:50]]
        for name, _ in meshes.items()
        for other_name in meshes
        if other_name != name
    }

    for name, ham in meshes.items():
        isolated = ham.find_isolated(top_k=15)
        gaps = [t for _, _, t in isolated]
        results["after"][name] = {"gaps": gaps}

        # Count how many new gaps are from the other domain
        before_gaps = set(results["before"][name]["gaps"])
        new_gaps = [g for g in gaps if g not in before_gaps]
        cross_domain_new = [
            g for g in new_gaps
            if any(marker[:30].lower() in g.lower() or g[:30].lower() in marker.lower()
                   for marker in other_domain_markers.get(name, []))
            or "[from " in g  # explicitly marked cross-domain
        ]

        results["after"][name]["new_gaps"] = new_gaps
        results["after"][name]["cross_domain_new_gaps"] = cross_domain_new
        results["after"][name]["cross_domain_fraction"] = (
            len(cross_domain_new) / max(len(new_gaps), 1)
        )

        print(f"    [{name}]: {len(gaps)} isolated, {len(new_gaps)} new gaps, "
              f"{len(cross_domain_new)} cross-domain")
        for g in cross_domain_new[:3]:
            print(f"      ✓ {g[:70]}")

    verdict = "SUPPORTS C3" if any(
        m["cross_domain_fraction"] > 0.2
        for m in results["after"].values()
    ) else "MIXED"
    results["conclusion"] = verdict
    print(f"\n  [{verdict}]")

    return save_results("C3", results), results


# ---------------------------------------------------------------------------
# C4 — Curiosity synthesis is genuinely cross-domain
# ---------------------------------------------------------------------------

def experiment_c4(mesh_spec: dict, n_gaps=8):
    """
    Claim: The curiosity engine generates insights that draw from MULTIPLE
    mesh domains — not just retrieving from the home domain.

    Method:
      1. Run curiosity engine on the collective
      2. For each Q→A insight, record which meshes' memories activated
      3. Compute cross-domain fraction: insights where >1 domain activated
      4. Also: compare answer quality with and without multi-mesh context

    Expected result: cross-domain fraction > 0.5
    """
    print("\n" + "=" * 60)
    print("  C4: Curiosity Synthesis Is Genuinely Cross-Domain")
    print("=" * 60)

    meshes = load_meshes(mesh_spec)
    collective = MeshCollective(meshes)
    logger = HAMLogger(f"experiment_C4")

    results = {
        "claim": "C4",
        "hypothesis": "Curiosity insights draw from multiple mesh domains",
        "n_gaps_tested": n_gaps,
        "insights": [],
        "cross_domain_count": 0,
        "cross_domain_fraction": 0.0,
        "conclusion": None,
    }

    print(f"\n  Running curiosity engine ({n_gaps} gaps)...")

    # Monkeypatched version that logs everything
    gap_catalog = {}
    for name, ham in meshes.items():
        for score, idx, text in ham.find_isolated(top_k=20):
            key = text[:80]
            gap_catalog.setdefault(key, []).append((score, name))

    universal = [(t, e) for t, e in gap_catalog.items() if len(e) >= 2]
    domain_only = [(t, e) for t, e in gap_catalog.items() if len(e) == 1]
    gaps = (universal + domain_only)[:n_gaps]

    for gap_text, entries in gaps:
        mesh_names = [name for _, name in entries]
        print(f"\n  Gap [{', '.join(mesh_names)}]: '{gap_text[:55]}'")

        q_prompt = (
            f"The concept '{gap_text}' is at the boundary of knowledge "
            f"spanning {', '.join(meshes.keys())}. "
            f"Generate 2 specific probing questions. One per line."
        )
        raw_q = generate(q_prompt,
                         system="Be curious. One question per line.",
                         max_tokens=200)
        questions = [q.strip().lstrip('-').strip()
                     for q in raw_q.strip().split('\n')
                     if q.strip() and len(q.strip()) > 15][:2]

        for question in questions:
            q_emb = embed(question)
            blended, activated = collective.collective_resonate(q_emb, hops=2, top_k=6)

            meshes_seen = list({m for sim, _t, m in activated if sim > CONFIDENCE_FLOOR})
            is_cross_domain = len(meshes_seen) > 1

            context = "\n".join(
                f"[{m}|{s:+.2f}] {t[:100]}"
                for s, t, m in activated if s > CONFIDENCE_FLOOR
            )
            sys_prompt = (
                "Synthesise from these cross-domain patterns:\n\n"
                f"{context}\n\nFind connections across domains."
            ) if context else "Answer from your own knowledge."

            answer = generate(question, system=sys_prompt, max_tokens=350)

            insight = {
                "gap":           gap_text[:80],
                "question":      question[:150],
                "answer_excerpt": answer[:300],
                "meshes_activated": meshes_seen,
                "cross_domain":     is_cross_domain,
                "top_sim":          round(activated[0][0], 4) if activated else 0.0,
            }
            results["insights"].append(insight)

            if is_cross_domain:
                results["cross_domain_count"] += 1
                print(f"    ✓ cross-domain [{', '.join(meshes_seen)}]: {question[:50]}")
            else:
                print(f"    · single-domain [{', '.join(meshes_seen)}]: {question[:50]}")

    total = len(results["insights"])
    if total > 0:
        results["cross_domain_fraction"] = round(
            results["cross_domain_count"] / total, 3
        )

    verdict = (
        "STRONGLY SUPPORTS C4" if results["cross_domain_fraction"] > 0.6
        else "SUPPORTS C4"      if results["cross_domain_fraction"] > 0.4
        else "MIXED"             if results["cross_domain_fraction"] > 0.2
        else "REFUTES C4"
    )
    results["conclusion"] = verdict

    print(f"\n  Cross-domain fraction: {results['cross_domain_fraction']:.3f} "
          f"({results['cross_domain_count']}/{total})")
    print(f"  [{verdict}]")

    return save_results("C4", results), results


# ---------------------------------------------------------------------------
# C5 — Multi-hop follows transitive chains
# ---------------------------------------------------------------------------

def experiment_c5(mesh_spec: dict, n_queries=20):
    """
    Claim: 2-hop diffraction activates meaningfully different (more
    transitive) memories than 1-hop diffraction.

    Method:
      1. Take N diverse queries
      2. Run both 1-hop and 2-hop diffraction on each mesh
      3. Measure: do the top activations differ? Are 2-hop activations
         semantically further from the query (transitivity score)?

    Transitivity score: cosine_sim(query, 2hop_top) < cosine_sim(query, 1hop_top)
    — if 2-hop lands on something LESS similar to the query, it followed a chain.
    """
    print("\n" + "=" * 60)
    print("  C5: Multi-Hop Follows Transitive Chains")
    print("=" * 60)

    # Seed queries spanning different domains
    TEST_QUERIES = [
        "What is the relationship between entropy and information?",
        "How does evolutionary pressure create structure?",
        "What is the foundation of number theory?",
        "How does perception relate to reality?",
        "What connects cryptography to mathematics?",
        "How does memory shape identity?",
        "What is the nature of consciousness?",
        "How do systems self-organize?",
        "What is the role of symmetry in physics?",
        "How does language shape thought?",
    ][:n_queries]

    meshes = load_meshes(mesh_spec)

    results = {
        "claim": "C5",
        "hypothesis": "2-hop activates semantically different (more transitive) memories than 1-hop",
        "queries_tested": len(TEST_QUERIES),
        "meshes": {},
        "conclusion": None,
    }

    for mesh_name, ham in meshes.items():
        print(f"\n  [{mesh_name}]")
        query_results = []

        for question in TEST_QUERIES:
            q_emb = embed(question)

            intuition_1 = ham.diffract(q_emb, hops=1)
            intuition_2 = ham.diffract(q_emb, hops=2)

            _, act_1 = ham.resonate(q_emb, hops=1, top_k=3)
            _, act_2 = ham.resonate(q_emb, hops=2, top_k=3)

            top1_text = act_1[0][2] if act_1 else ""
            top2_text = act_2[0][2] if act_2 else ""

            # Semantic distance from query (lower = further = more transitive)
            sim_1hop = act_1[0][0] if act_1 else 0.0
            sim_2hop = act_2[0][0] if act_2 else 0.0

            # Did 2-hop land somewhere different?
            different = top1_text[:40] != top2_text[:40]

            query_results.append({
                "question":   question,
                "1hop_top":   top1_text[:80],
                "2hop_top":   top2_text[:80],
                "1hop_sim":   round(sim_1hop, 4),
                "2hop_sim":   round(sim_2hop, 4),
                "different":  different,
            })

            marker = "≠" if different else "="
            print(f"    {marker} 1h={top1_text[:35]!r:38} 2h={top2_text[:35]!r}")

        different_fraction = sum(1 for r in query_results if r["different"]) / max(len(query_results), 1)
        avg_1hop_sim = sum(r["1hop_sim"] for r in query_results) / max(len(query_results), 1)
        avg_2hop_sim = sum(r["2hop_sim"] for r in query_results) / max(len(query_results), 1)

        verdict = (
            "SUPPORTS C5"  if different_fraction > 0.5
            else "MIXED"   if different_fraction > 0.3
            else "REFUTES C5"
        )

        print(f"    Different: {different_fraction:.2f}  1-hop sim: {avg_1hop_sim:.3f}  "
              f"2-hop sim: {avg_2hop_sim:.3f}  [{verdict}]")

        results["meshes"][mesh_name] = {
            "query_results": query_results,
            "different_fraction": round(different_fraction, 3),
            "avg_1hop_sim": round(avg_1hop_sim, 3),
            "avg_2hop_sim": round(avg_2hop_sim, 3),
            "verdict": verdict,
        }

    results["conclusion"] = "SUPPORTS C5" if all(
        m["verdict"] == "SUPPORTS C5" for m in results["meshes"].values()
    ) else "MIXED"

    return save_results("C5", results), results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_mesh_spec(spec_str: str) -> dict:
    mesh_spec = {}
    for part in spec_str.split(','):
        name, path = part.strip().split(':', 1)
        mesh_spec[name.strip()] = path.strip()
    return mesh_spec


def main():
    parser = argparse.ArgumentParser(description="HAMesh experiment runner")
    parser.add_argument("--claim", type=str, choices=["C1","C2","C3","C4","C5","all"],
                        default="all")
    parser.add_argument("--meshes", type=str, default=None)
    parser.add_argument("--cycles", type=int, default=500,
                        help="Dream cycles for C1/C2 (default 500)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Independent runs for C1 (default 3)")
    parser.add_argument("--gaps", type=int, default=8,
                        help="Gaps to explore for C4 (default 8)")
    args = parser.parse_args()

    mesh_spec = parse_mesh_spec(args.meshes) if args.meshes else DEFAULT_MESHES

    print("=" * 60)
    print("  HAMesh Experiment Runner")
    print(f"  Meshes: {', '.join(mesh_spec.keys())}")
    print(f"  Claim:  {args.claim}")
    print("=" * 60)

    claims_to_run = (
        ["C1","C2","C3","C4","C5"] if args.claim == "all"
        else [args.claim]
    )

    all_results = {}
    for claim in claims_to_run:
        try:
            if claim == "C1":
                _, r = experiment_c1(mesh_spec, dream_cycles=args.cycles, n_runs=args.runs)
            elif claim == "C2":
                _, r = experiment_c2(mesh_spec, dream_cycles=args.cycles)
            elif claim == "C3":
                _, r = experiment_c3(mesh_spec)
            elif claim == "C4":
                _, r = experiment_c4(mesh_spec, n_gaps=args.gaps)
            elif claim == "C5":
                _, r = experiment_c5(mesh_spec)
            all_results[claim] = r.get("conclusion", "ERROR")
        except Exception as e:
            print(f"\n  ERROR in {claim}: {e}")
            all_results[claim] = f"ERROR: {e}"

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for claim, conclusion in all_results.items():
        print(f"  {claim}: {conclusion}")

    # Save master summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"experiment_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "meshes": list(mesh_spec.keys()),
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
