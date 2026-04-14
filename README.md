# HAMesh — Holographic Associative Memory + Bonsai 1-bit LLM

A self-organizing associative memory that thinks in an LLM's native embedding space.

The **LLM is the sensory organ** — it encodes text into vectors (eyes) and decodes vectors back into language (mouth).  
The **HAM is the brain** — it stores knowledge as wave interference patterns and retrieves via diffraction, not search.

Unlike RAG (which retrieves document chunks), the HAM *synthesizes*: a query diffracts through the superposition of all stored knowledge simultaneously, and the answer constructively interferes into existence. Knowledge folds accumulate over time, the mesh self-organizes through dreaming, and a curiosity engine autonomously fills gaps.

---

## What Makes This Different From RAG / Standard LLM Memory

| | RAG | Fine-tuning | **HAMesh** |
|---|---|---|---|
| Updates without retraining | ✓ | ✗ | ✓ |
| Synthesizes across patterns | ✗ retrieves chunks | ✓ (baked in) | ✓ via interference |
| Persistent across sessions | ✓ | ✓ | ✓ |
| Self-organizing topology | ✗ | ✗ | ✓ attractor formation |
| Multi-hop transitive reasoning | ✗ | partial | ✓ re-diffraction |
| Interpretable resonance scores | ✗ | ✗ | ✓ |
| Develops "personality" through use | ✗ | ✗ | ✓ |
| Autonomous gap-filling (curiosity) | ✗ | ✗ | ✓ |

---

## Architecture

```
Your question
      │
      ▼
 Bonsai LLM  ──embed──►  4096-dim vector
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Holographic Mesh   │  ← 4096×4096 float32
                    │  M = Σ k_i ⊗ v_i   │    superposition of all
                    │                     │    folded associations
                    │  diffract: M @ q    │
                    └─────────────────────┘
                               │
                          synthesized
                          intuition vector
                               │
                               ▼
                    top-k cosine similarity
                    against stored memories
                               │
                          activated patterns
                          + resonance scores
                               │
                               ▼
                 Bonsai LLM  ──generate──►  Answer
                 (constrained to synthesize
                  ONLY from HAM patterns)
```

**Multi-hop:** the synthesized vector is fed back as the next query, following transitive chains (A→B→C) without explicit graph traversal.

**Dreaming:** the HAM queries itself recursively, folding outputs back in. Self-reinforcing patterns become permanent attractor basins. Strange loops form between semantically adjacent concepts across domains.

**Curiosity:** the engine finds knowledge islands (memories with weak connections to others), generates probing questions at those boundaries, diffracts them through the mesh, and folds the cross-domain answers back in — autonomous self-expansion.

---

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended — tested on RTX 4090)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-server`
- [PrismML Bonsai 8B GGUF](https://huggingface.co/PrismML/1bit-bonsai-8b-GGUF) (`bonsai-8b.gguf`)

```bash
pip install torch requests
```

---

## Setup

**1. Start the Bonsai LLM server** (must include `--embedding --pooling mean`):

```bash
./llama-server.exe -m bonsai-8b.gguf --port 22334 --ctx-size 8192 --embedding --pooling mean
```

> The `--embedding --pooling mean` flags are required. Without them the `/v1/embeddings` endpoint is unavailable and the HAM cannot encode anything.

**2. Add your knowledge files**

Drop plain `.txt` files into `./ham_data/` (or any folder you choose with `--data-dir`).  
Every paragraph becomes a chunk, embedded and folded into the mesh on startup.

```
ham_data/
  my_research_notes.txt
  project_docs.txt
  interesting_papers_summary.txt
```

The HAM ingests on every startup — it won't re-fold duplicates if you save/load the mesh state.

---

## Usage

### Single Mesh — `ham_brain.py`

```bash
python ham_brain.py
# with custom paths:
python ham_brain.py --save my_mesh.pt --data-dir ./my_knowledge
```

| Command | What it does |
|---|---|
| `<question>` | 1-hop diffraction — direct associations |
| `deep <question>` | 2-hop transitive reasoning (A→B→C) |
| `trace <question>` | Shows the chain of thought hop-by-hop |
| `teach <text>` | Folds new text directly into the mesh |
| `fold <A> -> <B>` | Stores a directed association |
| `dream [N]` | N cycles of recursive self-modification (default 50) |
| `status` | Mesh diagnostics (folds, memories, energy) |
| `save` | Persist mesh to disk |
| `exit` | Quit |

---

### Distill LLM Knowledge Into The Mesh — `ham_distill.py`

Decompresses the LLM's weights into the HAM by having Bonsai generate explanations, embedding them, and folding the Q→A associations into the mesh.

```bash
# Default: science + technology, depth 2 (topic + 5 subtopics each)
python ham_distill.py

# All domains
python ham_distill.py --topics science,math,technology,philosophy,history,nature --depth 2

# Distill into a named mesh
python ham_distill.py --topics science,math,technology --save ham_science.pt
python ham_distill.py --topics philosophy,history,nature --save ham_philosophy.pt

# Add to an existing mesh
python ham_distill.py --load ham_science.pt --topics math
```

Available domains: `science`, `math`, `technology`, `history`, `philosophy`, `nature`

---

### Dual-Mesh Collective — `ham_collective.py`

Runs two (or more) specialized meshes in parallel. Queries go to all meshes simultaneously; results are labeled by source and blended. Strongest attractor patterns cross-pollinate during collective dreaming.

```bash
# Requires both meshes to be distilled first (see above)
python ham_collective.py

# Custom mesh layout
python ham_collective.py --meshes science:ham_science.pt,philosophy:ham_philosophy.pt

# Custom knowledge folder
python ham_collective.py --data-dir ./my_docs
```

| Command | What it does |
|---|---|
| `<question>` | Both meshes diffract; color-coded by source |
| `deep <question>` | 2-hop across all meshes, blended |
| `trace <question>` | Per-mesh hop chains shown side by side |
| `curious [N]` | Curiosity engine: find N gaps, generate questions, fold answers in |
| `dream [N]` | All meshes dream simultaneously + cross-pollinate |
| `cross` | Manual cross-pollination pass |
| `teach <text>` | Fold into all meshes |
| `fold <A> -> <B>` | Directed association in all meshes |
| `status` | Per-mesh diagnostics |
| `save` | Save all meshes to their configured paths |

---

## Typical Workflow

```
Day 1 — First run
  python ham_distill.py --topics science,technology --save ham_science.pt
  python ham_distill.py --topics philosophy,history,nature --save ham_philosophy.pt
  # Drop your own .txt files into ham_data/
  python ham_collective.py

Daily use
  python ham_collective.py          # loads saved meshes, ingests ham_data/
  > deep how does entropy relate to information theory?
  > curious 4                       # let it find and fill its own gaps
  > dream 100                       # develop attractors
  > save

Periodically
  python ham_distill.py --load ham_science.pt --topics math  # expand knowledge
```

---

## How Knowledge Is Stored

The HAM stores a `dim × dim` matrix (4096×4096 for Bonsai 8B):

```
M = Σ strength_i × normalize(key_i) ⊗ normalize(val_i)
```

Every fold superimposes a rank-1 outer product onto the same matrix. All knowledge coexists in the same physical space — there are no discrete slots or indexes.

**Retrieval:**
```
output = normalize(M @ normalize(query))
```

The output is a weighted combination of all stored values, weighted by cosine similarity between the query and each stored key. It's mathematically equivalent to single-head linear attention without softmax.

**Multi-hop:**
```
hop_1 = normalize(M @ normalize(query))
hop_2 = normalize(M @ hop_1)
```

Each hop follows associations one step further. Transitive reasoning (A→B→C) emerges naturally from re-diffraction without any explicit graph.

---

## Dreaming & Attractor Formation

```bash
HAM> dream 100
```

Each dream cycle:
1. Picks a seed (random memory or last cycle's output)
2. Diffracts through the mesh
3. Folds the output back in (self-modification)
4. Uses the output as the next cycle's seed — the loop

Over many cycles, certain patterns become **attractor basins** — gravitational wells that pull all future queries toward them. **Strange loops** form: cyclic activation chains (A→B→C→A) that sustain and strengthen themselves with each traversal.

After dreaming, the mesh has permanently reorganized around its dominant patterns. This is the HAM developing a "personality" through use.

---

## Curiosity Engine

```bash
COLLECTIVE> curious 4
```

1. Finds memories with weak connections to their neighbors (knowledge islands)
2. Identifies universal gaps — concepts weakly represented across ALL meshes
3. Asks Bonsai to generate probing questions at each boundary
4. Diffracts each question through the full collective (2-hop)
5. Generates cross-domain answers from the blended intuition
6. Folds every Q→A back into all meshes

The HAM identifies what it doesn't know and autonomously fills the gaps.

---

## File Structure

```
HAMesh/
  ham_core.py          — Holographic mesh engine (fold, diffract, dream, gaps)
  ham_brain.py         — Single-mesh CLI + Bonsai integration
  ham_collective.py    — Multi-mesh collective + curiosity engine CLI
  ham_distill.py       — LLM knowledge distillation pipeline
  ham_data/            — Drop .txt knowledge files here
    ham_primer.txt     — HAM self-description (included)
  ham_state.pt         — Single mesh save (git-ignored, regenerate)
  ham_science.pt       — Science/tech mesh (git-ignored, regenerate)
  ham_philosophy.pt    — Philosophy/history/nature mesh (git-ignored, regenerate)

  # Legacy files from earlier experiments:
  Hebb.py, kernel.py, ortho.py, query.py
  two_layer.py, semantic_mesh.py, bonsai_rag.py
  ham_injector.py, geospatial.py, rsc_loop.py, main.py
```

---

## Related Work

- **Hypertokens / HDRAM** (arxiv 2507.00002, 2025) — closest published work; holographic operations in LLM embedding space for retrieval. HAMesh focuses on synthesis, persistence, and self-modification rather than retrieval efficiency.
- **Modern Hopfield Networks** (Ramsauer et al., 2020) — proved transformer attention is associative memory retrieval. HAMesh is a complementary external memory operating in the same geometric space.
- **Sparse Distributed Memory** (Kanerva, 1988) — theoretical ancestor; content-addressable memory with distributed storage.
- **Holographic Reduced Representations** (Plate, 1995) — mathematical framework for holographic vector binding.
- **Daydreaming Hopfield Networks** (2025) — dreaming-style updates for Hopfield networks; HAMesh extends this to LLM embedding spaces with curiosity-driven self-expansion.

---

## Notes

- `.pt` mesh files are excluded from git — they're large (~70MB each) and regenerate in ~15 min via `ham_distill.py`
- The mesh operates in Bonsai's native 4096-dim embedding space; using a different model changes the geometry and requires re-distillation
- Theoretical capacity: ~573 orthogonal patterns before interference degrades. Use `status` to monitor mesh energy as you add knowledge.
- The 1-bit Bonsai model (`bonsai-8b.gguf`, ~1.15GB) runs entirely locally — no API keys, no data leaves your machine

---

*Built on PrismML's [1-bit Bonsai 8B](https://huggingface.co/PrismML/1bit-bonsai-8b-GGUF) and [llama.cpp](https://github.com/ggerganov/llama.cpp).*
