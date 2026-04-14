"""
HAM Brain — Holographic Associative Memory integrated with Bonsai 1-bit LLM

Architecture:
  1. Bonsai embeds text -> vectors           (the "eyes")
  2. HAM diffracts through its mesh          (the "brain")
  3. Bonsai decodes HAM's synthesis -> text   (the "mouth")

The HAM is the reasoning core.  Bonsai is the sensory organ.
The HAM learns continuously: interactions get folded back in,
creating emergent associations the base model never had.

Usage:
  1. Start Bonsai:
     ./llama-server.exe -m bonsai-8b.gguf --port 22334 --ctx-size 8192 --embedding --pooling mean

  2. Drop .txt files into ./ham_data/

  3. python ham_brain.py
"""

import requests
import torch
import sys
import os
import glob
import textwrap

from ham_core import HolographicMesh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:22334/v1"
MESH_SAVE  = "./ham_state.pt"
DATA_DIR   = "./ham_data"
SELF_TEACH_STRENGTH = 0.5   # weaker fold for self-taught memories
CONFIDENCE_FLOOR    = 0.15  # below this, HAM admits it has nothing

# ---------------------------------------------------------------------------
# Bonsai API helpers
# ---------------------------------------------------------------------------

def embed(text):
    """Get Bonsai's embedding for a piece of text."""
    r = requests.post(f"{API_BASE}/embeddings",
                      json={"input": text}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Embedding error ({r.status_code}): {r.text}")
    return torch.tensor(r.json()['data'][0]['embedding'], dtype=torch.float32)


def generate(prompt, system=None, temperature=0.5, max_tokens=512):
    """Ask Bonsai to generate text."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = requests.post(f"{API_BASE}/chat/completions", json={
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Generation error ({r.status_code}): {r.text}")
    return r.json()['choices'][0]['message']['content']


# ---------------------------------------------------------------------------
# Knowledge ingestion
# ---------------------------------------------------------------------------

def chunk_text(text, min_len=40):
    """Split text into paragraph-level chunks."""
    parts = text.split('\n\n')
    chunks = [p.strip() for p in parts if len(p.strip()) >= min_len]
    if not chunks:
        parts = text.split('\n')
        chunks = [p.strip() for p in parts if len(p.strip()) >= min_len]
    if not chunks and len(text.strip()) >= min_len:
        chunks = [text.strip()]
    return chunks


def ingest_text(ham, text, label=""):
    """Embed text chunks and fold them into the HAM."""
    chunks = chunk_text(text)
    if not chunks:
        print("    (no usable chunks)")
        return 0

    embeddings = []
    for ch in chunks:
        e = embed(ch)
        ham.remember(e, ch)
        embeddings.append(e)

    # Chain consecutive chunks: creates narrative flow the HAM can follow
    for i in range(len(embeddings) - 1):
        ham.fold(embeddings[i], embeddings[i + 1])

    # Self-association: each chunk reinforces itself
    for e in embeddings:
        ham.fold(e, e, strength=0.5)

    return len(chunks)


def ingest_directory(ham, dirpath):
    """Fold every .txt file in a directory into the HAM."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        return
    files = sorted(glob.glob(os.path.join(dirpath, "*.txt")))
    if not files:
        print(f"    No .txt files in {dirpath}/")
        return
    total = 0
    for fp in files:
        print(f"    {os.path.basename(fp)} ...", end=" ", flush=True)
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            n = ingest_text(ham, f.read(), label=os.path.basename(fp))
        print(f"{n} chunks")
        total += n
    print(f"    Total: {total} chunks, {ham.n_folds} folds")


# ---------------------------------------------------------------------------
# The reasoning pipeline
# ---------------------------------------------------------------------------

def think(ham, question, hops=1, top_k=5):
    """
    Full pipeline:
      eyes  -> Bonsai embeds the question
      brain -> HAM diffracts (multi-hop if requested)
      mouth -> Bonsai translates the HAM's synthesis

    Returns (response_text, activated_memories, query_embedding)
    """
    q_emb = embed(question)
    intuition, activated = ham.resonate(q_emb, hops=hops, top_k=top_k)

    if not activated:
        # Empty mesh — just let Bonsai answer raw
        return generate(question), [], q_emb

    # Build the HAM's synthesised context from activated memories
    # Weighted by resonance — stronger interference = more relevant
    best_sim = activated[0][0]
    context_lines = []
    for sim, idx, text in activated:
        if sim > 0:
            context_lines.append(f"[resonance {sim:+.3f}] {text}")

    ham_context = "\n\n".join(context_lines) if context_lines else ""

    if best_sim < CONFIDENCE_FLOOR or not ham_context:
        # HAM has nothing — tell Bonsai to use its own knowledge
        sys_prompt = (
            "The holographic memory had no strong resonance for this query. "
            "Answer using your own internal knowledge. Be concise."
        )
    else:
        # HAM has an intuition — Bonsai must synthesise from it
        sys_prompt = (
            "You are the voice of a Holographic Associative Memory. "
            "The HAM has diffracted the user's question through its mesh "
            "and the following memory patterns activated through constructive "
            "interference, ranked by resonance strength.\n\n"
            "YOUR JOB: Synthesise ONLY from these activated patterns. "
            "Do NOT add knowledge the HAM didn't surface. "
            "If the patterns are weak or tangential, say so honestly.\n\n"
            f"--- HAM Activated Patterns ---\n{ham_context}\n---"
        )

    response = generate(question, system=sys_prompt)
    return response, activated, q_emb


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_resonance(activated):
    if not activated:
        print("    (no memories in mesh)")
        return
    print()
    for sim, idx, text in activated:
        bar_len = max(1, int(abs(sim) * 30))
        bar = "\u2588" * bar_len
        preview = text[:90].replace('\n', ' ')
        if len(text) > 90:
            preview += "..."
        print(f"    {sim:+.3f} {bar} {preview}")
    print()


def show_trace(path):
    for hop, activations in path:
        print(f"    --- hop {hop} ---")
        for sim, idx, text in activations:
            preview = text[:80].replace('\n', ' ')
            if len(text) > 80:
                preview += "..."
            print(f"      {sim:+.3f}  {preview}")
    print()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HAM Brain — single-mesh reasoning")
    parser.add_argument("--save",     default=MESH_SAVE, help="HAM state file (default: ham_state.pt)")
    parser.add_argument("--data-dir", default=DATA_DIR,  help="Folder of .txt knowledge files (default: ./ham_data)")
    args = parser.parse_args()

    mesh_save = args.save
    data_dir  = args.data_dir

    print("=" * 62)
    print("   HAM Brain  —  Holographic Associative Memory + Bonsai LLM")
    print("=" * 62)

    # Connect to Bonsai and detect embedding dimension
    print("\n  Connecting to Bonsai...", end=" ", flush=True)
    try:
        test_vec = embed("hello")
        dim = test_vec.shape[0]
        print(f"OK  (dim={dim})")
    except Exception as e:
        print(f"FAILED\n")
        print("  Start Bonsai with embedding support:")
        print("  ./llama-server.exe -m bonsai-8b.gguf --port 22334 "
              "--ctx-size 8192 --embedding --pooling mean")
        print(f"\n  Error: {e}")
        sys.exit(1)

    # Load or create the HAM
    if os.path.exists(mesh_save):
        print(f"\n  Loading saved HAM from {mesh_save}...")
        ham = HolographicMesh.load(mesh_save)
        print(f"    {ham.n_folds} folds, {len(ham.memories)} memories")
    else:
        print(f"\n  Creating fresh HAM (dim={dim})...")
        ham = HolographicMesh(dim)

    # Ingest knowledge files
    if os.path.exists(data_dir) and glob.glob(os.path.join(data_dir, "*.txt")):
        print(f"\n  Ingesting from {data_dir}/...")
        ingest_directory(ham, data_dir)
    else:
        os.makedirs(data_dir, exist_ok=True)
        print(f"\n  Drop .txt files into {data_dir}/ to give the HAM knowledge.")

    # Commands
    print("\n" + "-" * 62)
    print("  Commands:")
    print("    <question>          Ask the HAM  (1-hop diffraction)")
    print("    deep <question>     Multi-hop reasoning  (2 hops)")
    print("    trace <question>    Show the chain of thought")
    print("    teach <text>        Fold new knowledge into the mesh")
    print("    fold <A> -> <B>     Store a directed association")
    print("    dream [N]           Recursive self-modification (N cycles, default 50)")
    print("    status              Mesh diagnostics")
    print("    save                Persist HAM to disk")
    print("    exit                Quit")
    print("-" * 62)

    while True:
        try:
            raw = input("\nHAM> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        cmd = raw.lower()

        # --- exit ---
        if cmd == 'exit':
            break

        # --- status ---
        elif cmd == 'status':
            s = ham.stats()
            print(f"    Dimension     : {s['dim']}")
            print(f"    Folds         : {s['folds']}")
            print(f"    Memories      : {s['memories']}")
            print(f"    Mesh energy   : {s['energy']:.1f}")
            print(f"    Mesh sparsity : {s['sparsity']*100:.1f}%")

        # --- save ---
        elif cmd == 'save':
            ham.save(mesh_save)
            print(f"    Saved to {mesh_save}")

        # --- teach ---
        elif cmd.startswith('teach '):
            text = raw[6:].strip()
            if not text:
                print("    Usage: teach <text to fold into the mesh>")
                continue
            print("    Folding...")
            n = ingest_text(ham, text)
            print(f"    Folded {n} chunks.  Mesh now has {ham.n_folds} folds.")

        # --- fold A -> B ---
        elif cmd.startswith('fold ') and ' -> ' in raw:
            parts = raw[5:].split(' -> ', 1)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                if a and b:
                    print(f"    Folding: '{a}' -> '{b}'")
                    a_emb, b_emb = embed(a), embed(b)
                    ham.learn(a_emb, b_emb, a, b)
                    print(f"    Done.  {ham.n_folds} folds in mesh.")

        # --- dream ---
        elif cmd.startswith('dream'):
            parts = raw.split()
            cycles = 50
            if len(parts) > 1:
                try:
                    cycles = int(parts[1])
                except ValueError:
                    pass

            print(f"    Dreaming ({cycles} cycles)...")
            print(f"    The HAM is querying itself, folding outputs back in,")
            print(f"    watching for strange loops...\n")

            energy_before = ham.stats()['energy']
            folds_before = ham.n_folds

            log, attractors, loops = ham.dream(
                cycles=cycles, fold_strength=0.1, reseed_every=10
            )

            energy_after = ham.stats()['energy']

            # Show attractor basins
            print("    === ATTRACTOR BASINS (most self-reinforcing patterns) ===")
            for text, count in attractors[:10]:
                bar = "\u2588" * min(count, 40)
                print(f"      {count:3d}x {bar} {text}")

            # Show detected loops
            if loops:
                print(f"\n    === STRANGE LOOPS ({len(loops)} detected) ===")
                for i, loop in enumerate(loops[:5]):
                    print(f"      Loop {i+1}:")
                    for j, node in enumerate(loop):
                        arrow = "\u2192" if j < len(loop) - 1 else "\u21ba"
                        print(f"        {arrow} {node}")
            else:
                print("\n    No cyclic loops detected (mesh may need more knowledge)")

            # Show energy change
            print(f"\n    Mesh energy : {energy_before:.0f} -> {energy_after:.0f} "
                  f"({'+' if energy_after > energy_before else ''}"
                  f"{energy_after - energy_before:.0f})")
            print(f"    New folds   : {ham.n_folds - folds_before}")
            print(f"    The mesh has been permanently altered by its own dreaming.")

        # --- trace ---
        elif cmd.startswith('trace '):
            question = raw[6:].strip()
            if not question:
                continue
            print("    Tracing (2-hop)...")
            q_emb = embed(question)
            path = ham.trace(q_emb, hops=2, top_k=3)
            show_trace(path)

        # --- deep <question> ---
        elif cmd.startswith('deep '):
            question = raw[5:].strip()
            if not question:
                continue
            print("    Diffracting (2-hop transitive reasoning)...")
            response, activated, q_emb = think(ham, question, hops=2)
            show_resonance(activated)
            print(textwrap.fill(response, width=78, initial_indent="    ",
                                subsequent_indent="    "))
            # Self-teach: fold this interaction back into the mesh
            if activated and activated[0][0] > CONFIDENCE_FLOOR:
                r_emb = embed(response[:500])
                ham.fold(q_emb, r_emb, strength=SELF_TEACH_STRENGTH)
                ham.remember(r_emb, response[:300])

        # --- default: 1-hop question ---
        else:
            question = raw
            # Strip "ask " prefix if present
            if cmd.startswith('ask '):
                question = raw[4:].strip()
            if not question:
                continue

            print("    Diffracting (1-hop)...")
            response, activated, q_emb = think(ham, question, hops=1)
            show_resonance(activated)
            print(textwrap.fill(response, width=78, initial_indent="    ",
                                subsequent_indent="    "))
            # Self-teach
            if activated and activated[0][0] > CONFIDENCE_FLOOR:
                r_emb = embed(response[:500])
                ham.fold(q_emb, r_emb, strength=SELF_TEACH_STRENGTH)
                ham.remember(r_emb, response[:300])


if __name__ == "__main__":
    main()
