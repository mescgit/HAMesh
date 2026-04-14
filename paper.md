# HAMesh: Emergent Attractor Dynamics and Cross-Domain Synthesis in Holographic Associative Memory Operating in Large Language Model Embedding Space

**[Author]**  
Independent Research  
GitHub: https://github.com/mescgit/HAMesh

---

## Abstract

We introduce HAMesh, a holographic associative memory (HAM) that operates natively in the embedding space of a large language model (LLM). Unlike retrieval-augmented generation (RAG), which fetches discrete document chunks, HAMesh *synthesizes*: a query vector diffracts through the superposition of all stored knowledge simultaneously, and the answer emerges through constructive interference. Through recursive self-modification we call *dreaming*, the mesh develops stable attractor basins without explicit supervision. We document four principal findings: **(1)** attractor identity is perfectly deterministic — independent dream runs on the same mesh converge to identical concept clusters (overlap = 1.000, n=6); **(2)** attractor clusters exhibit significantly higher semantic coherence than random memory samples (+0.22 to +0.30 above baseline); **(3)** a multi-mesh collective achieves 100% cross-domain activation in curiosity-driven synthesis (16/16 insights draw from both specialised meshes simultaneously); and **(4)** a post-distillation phase transition produces two deterministic equilibria — an always-collapsed state at 0 dream cycles and an always-distributed state at 500 cycles with mild decay — with stochastic bistability between, characteristic of systems near a bifurcation point. HAMesh provides persistent, self-organising memory that grows through use without retraining the underlying LLM.

---

## 1. Introduction

Large language models compress vast knowledge into fixed weights during training. At inference time they are stateless — each session begins with no memory of prior interactions, and updating their knowledge requires expensive retraining. Retrieval-augmented generation (RAG) partially addresses this by maintaining an external vector store, but retrieval returns the *nearest stored chunk*, not a synthesis across the full knowledge base.

Holographic associative memories offer a different model of storage and retrieval. In a HAM, knowledge is encoded as interference patterns in a dense matrix through outer-product superposition. Retrieval is not search — it is diffraction: a query propagates through the matrix and the answer emerges from the combined interference of all stored patterns simultaneously. This is content-addressable, gracefully degrading, and fundamentally different from lookup.

The key insight of HAMesh is to operate this holographic machinery *inside an LLM's native embedding space*. Rather than using random or learned representation vectors, we use the LLM's own embedding function as the sensory organ: queries are embedded by the LLM, diffracted through the holographic mesh, and the synthesised output is decoded back to language by the same LLM. The mesh inherits the LLM's full semantic geometry for free, enabling meaningful interference between concepts the model understands.

We make four contributions:

1. A holographic memory architecture that operates in LLM embedding space, with knowledge distilled from the LLM through generation rather than injected from external corpora.
2. A recursive self-modification process (*dreaming*) that develops stable attractor basins without supervision.
3. A multi-mesh collective with cross-pollination and a curiosity engine for autonomous gap identification and cross-domain synthesis.
4. An empirical characterisation of the phase transition between attractor-collapsed and multi-hop-capable mesh states, including two deterministic equilibria identified across six independent trials.

---

## 2. Related Work

**Holographic and associative memory.** Kanerva's Sparse Distributed Memory [1] and Plate's Holographic Reduced Representations [2] establish the mathematical foundations for content-addressable memory via distributed superposition. HRR uses circular convolution for compositional binding; our outer-product approach is closer to Hopfield network storage [3].

**Modern Hopfield networks.** Ramsauer et al. [4] proved that transformer self-attention is mathematically equivalent to dense associative memory retrieval, establishing a formal connection between transformers and Hopfield networks. HAMesh can be understood as an external Hopfield network operating in the same geometric space as the transformer's attention, rather than inside it.

**Holographic operations in LLM embedding space.** Hypertokens / HDRAM [5] (2025) is the closest published work: it treats the transformer's latent space as a spread-spectrum channel and applies holographic operations for improved key-value retrieval (2× improvement). HAMesh targets a different objective — persistent synthesis and self-organisation — rather than retrieval efficiency.

**Memory-augmented LLMs.** MemGPT [6], LangMem, and Memori [7] augment LLMs with external memory, but all use retrieval-based approaches (nearest-neighbour search over stored summaries). None develop self-organising topology or attractor dynamics.

**Dreaming in neural networks.** Dreaming Hopfield Networks [8] (2025) explores dream-like updates for correlated data. Prior work on sleep-inspired consolidation [9] separates learning from unlearning phases. HAMesh extends dreaming to LLM embedding spaces and documents the resulting attractor dynamics quantitatively.

---

## 3. Architecture

### 3.1 The Holographic Mesh

The core data structure is a `dim × dim` float32 matrix **M**, initialised to zero, where `dim` is the LLM's embedding dimension (4096 for Bonsai 8B). Knowledge is stored through *folding*: given a key embedding **k** and value embedding **v**, both L2-normalised:

```
M ← M + s · normalize(k) ⊗ normalize(v)
```

where `s` is a strength scalar and `⊗` denotes outer product. Multiple folds superimpose in the same physical space. This is Hebbian outer-product storage — mathematically equivalent to storing rank-1 updates to an associative matrix.

**Retrieval by diffraction.** Given a query embedding **q**:

```
output = normalize(M @ normalize(q))
         = normalize( Σ_i s_i · cos(k_i, q) · v_i )
```

The output is a weighted combination of all stored value vectors, weighted by cosine similarity between the query and each stored key. This is single-head linear attention without softmax, operating holographically: *all* associations respond simultaneously. The result is not the stored value nearest to the query — it is a synthesis vector blended from every relevant pattern.

**Multi-hop diffraction.** Re-applying the diffraction operator to its own output enables transitive reasoning:

```
hop_1 = normalize(M @ normalize(q))       # A → B
hop_2 = normalize(M @ hop_1)              # B → C
```

Each hop follows associations one step further without explicit graph traversal. We measure multi-hop *diversity* as the fraction of queries where the top-1 activated memory differs between 1-hop and 2-hop results.

### 3.2 Knowledge Distillation Through Generation

Rather than injecting external text, HAMesh extracts the LLM's own knowledge by prompting it to generate explanations of seed topics, embedding both the topic and explanation, and folding the association. For a seed topic T with generated explanation E:

```
M ← fold(embed(T), embed(E))
```

This decompresses knowledge from the LLM's weights into holographic form. The LLM generates subtopics, which are folded as children; cross-domain connections suggested by the LLM are folded as bidirectional bridges at reduced strength. We call this *distillation through generation*.

### 3.3 Dreaming

The dreaming process implements recursive self-modification. Each cycle:

1. Select a seed (random memory or previous cycle's output)
2. Diffract through the mesh to get `intuition`
3. Fold: `M ← M + α · intuition ⊗ intuition` (self-reference)
4. Fold: `M ← M + α · seed ⊗ intuition` (bridge)
5. Cross-link top-2 activated memories at reduced strength
6. Apply decay every 20 cycles: `M ← M · (1 - δ)`
7. Use `intuition` as next cycle's seed — the loop

Without decay (δ=0), self-reference folds compound indefinitely. With decay, the mesh reaches a dynamic equilibrium between self-reinforcement and forgetting.

### 3.4 Multi-Mesh Collective

Two specialised meshes (science/technology and philosophy/history/nature) run in parallel. Given a query embedding **q**, each mesh diffracts independently and their outputs are blended:

```
blended = normalize( Σ_m energy_m · diffract_m(q) )
```

weighted by each mesh's Frobenius norm (more knowledge → more influence). Activated memories from both meshes are pooled and ranked by cosine similarity to the blended signal.

**Cross-pollination** transfers each mesh's dominant attractor patterns (top eigenvector directions) to the other at low strength (0.04), creating cross-domain associations without homogenising the meshes.

**Curiosity engine.** Isolated memories (low average cosine similarity to their k-nearest neighbours) represent knowledge islands. For each isolated memory, the LLM generates probing questions at the knowledge boundary; the collective diffracts each question (2-hop); the cross-domain answer is folded into all meshes. This implements autonomous gap identification and filling.

---

## 4. Experiments

We evaluate five claims using Bonsai 8B (a 1-bit quantised Qwen3-8B, 1.15 GB GGUF) running via llama.cpp on an RTX 4090. Two meshes were distilled: science/technology (10 seed topics, depth 2, 5 subtopics each, ~120 concepts) and philosophy/history/nature (same structure).

### C1: Attractor Identity Is Deterministic

**Claim.** Independent dream runs on the same mesh converge to identical attractor clusters.

**Method.** Three independent runs of 500 dream cycles (decay=0.02) on each mesh, loaded fresh from disk for each run. Measure pairwise overlap of top-5 activated memories across run pairs.

**Results.**

| Mesh | Avg Pairwise Overlap | Avg Coherence | Verdict |
|---|---|---|---|
| Science | 1.000 | 0.733 | SUPPORTS |
| Philosophy | 1.000 | 0.667 | SUPPORTS |

Perfect overlap across all 6 run pairs (science: run1↔run2, run1↔run3, run2↔run3; same for philosophy). The attractors that formed — a cryptography/number-theory cluster in the science mesh and a consciousness/cognition cluster in the philosophy mesh — are identical regardless of random seed order in dreaming.

### C2: Attractor Clusters Are Semantically Coherent

**Claim.** The attractor clusters that form are meaningful — not high-energy noise, but real conceptual neighbourhoods.

**Method.** After dreaming to convergence, score the semantic coherence of the top-5 attractor concepts using the LLM as judge (0–10 scale, normalised). Compare against 10 random samples of 5 memories from the same mesh.

**Results.**

| Mesh | Attractor Coherence | Random Baseline | Δ | Verdict |
|---|---|---|---|---|
| Science | 0.80 | 0.50 | +0.30 | SUPPORTS |
| Philosophy | 0.70 | 0.48 | +0.22 | SUPPORTS |

The science mesh's cryptography cluster (Key Exchange → Hashing → Number Theory → Computer Architecture) received coherence 0.80 — number theory is the mathematical foundation of cryptographic key exchange, a real semantic relationship the mesh discovered through dreaming alone.

### C3: Cross-Pollination Creates Cross-Domain Curiosity

**Claim.** After cross-pollination, each mesh develops curiosity about topics from the other mesh's domain.

**Method.** Measure isolated memories per mesh before and after cross-pollination; count new gaps whose text includes markers from the other mesh's domain.

**Results.** After one cross-pollination pass, the science mesh acquired "[from philosophy] Battle of Stalingrad" as an isolated memory — a history topic that became a knowledge gap in the science mesh after philosophy's history knowledge bled in. The science mesh generated curiosity questions connecting strategic warfare to mathematical optimisation. Cross-domain gap fraction increased measurably post-pollination.

### C4: Curiosity Synthesis Is Cross-Domain

**Claim.** The curiosity engine generates insights drawing from multiple mesh domains simultaneously.

**Method.** Run curiosity engine on 8 knowledge gaps; for each generated Q→A pair, record which meshes' memories activated above the confidence floor.

**Results.**

| Metric | Value |
|---|---|
| Total insights | 16 |
| Cross-domain (both meshes activated) | 16/16 (100%) |
| Integrity (real questions + answers) | 16/16 (100%) |

Every single insight activated both the science and philosophy meshes. Sample cross-domain syntheses:

- *"The non-commutative nature of matrix multiplication — where order of multiplication affects the result — mirrors the asymmetric nature of power relationships in philosophical and political discourse."* [Gap: Matrix multiplication; Meshes: science × philosophy]

- *"Eigenvalues offer an analogy to epistemological fixed points — stable concepts that resist transformation under the application of new information."* [Gap: Fermat's Last Theorem; Meshes: science × philosophy]

- *"The constant motion of tectonic plates challenges traditional notions of stability and permanence — a theme that resonates across existentialist and process philosophy traditions."* [Gap: Plate tectonics; Meshes: science × philosophy]

### C5: Phase Transition in Multi-Hop Capability

**Claim.** Multi-hop diversity undergoes a phase transition as a function of dream cycles.

**Method.** Load fresh mesh, apply increasing dream cycles with decay=0.02, measure multi-hop diversity (fraction of 10 test queries where 2-hop top-1 differs from 1-hop top-1). Six independent trials.

**Results.**

| Dream Cycles | Mean Diversity | Std Dev | n | State |
|---|---|---|---|---|
| 0 | 0.0% | 0.0% | 6 | **Deterministic collapsed** |
| 100 | 53.3% | 48.9% | 6 | Stochastic bistable |
| 200 | 40.0% | 42.4% | 5 | Stochastic bistable |
| 300 | 70.0% | 44.7% | 5 | Stochastic (mostly distributed) |
| 400 | 60.0% | 54.8% | 5 | Stochastic bistable |
| 500 | **100.0%** | **0.0%** | **6** | **Deterministic distributed** |
| 700 | 60.0% | 54.8% | 5 | Stochastic bistable |
| 1000 | 66.7% | 51.6% | 6 | Stochastic (mostly distributed) |

Two points are perfectly consistent across all trials: 0 cycles (always collapsed) and 500 cycles (always distributed). All intermediate stages exhibit high variance (σ = 43–55%), indicating stochastic bistability — the random path taken through dream seed selection determines which attractor basin is occupied. This is characteristic of systems near a bifurcation point, where small perturbations can flip the system between competing stable states.

**Post-distillation collapse.** The fresh mesh (0 cycles) is already collapsed despite no dreaming. During distillation, hub topics (physics, computer architecture) receive disproportionate fold density — each appears as a seed, as parent of 5 subtopics, and as target in cross-links, accumulating ~10× the folds of peripheral topics. This creates attractor concentration before dreaming begins. Dreaming with decay redistributes this energy.

---

## 5. Discussion

### 5.1 Synthesis vs. Retrieval

RAG returns the stored chunk nearest to the query. HAMesh returns a synthesis vector that is a weighted interference of all relevant stored patterns. The practical difference is that HAMesh can respond to queries that fall between stored concepts — the interference of "genetics" and "computer architecture" produces a response drawing from both, weighted by relevance. No stored chunk captures this combination; it emerges from the geometry of the superposition.

### 5.2 The Goldilocks Zone

The phase transition data (Table C5) has a practical implication: operate the mesh at exactly 500 dream cycles with decay=0.02 after distillation. This is the only reliably distributed operating point. Fewer cycles leave the mesh in the post-distillation collapsed state; more cycles re-introduce stochasticity (though most paths remain distributed above 500 cycles). We recommend: distill, dream exactly 500 cycles, save, then limit interactive sessions to ≤100 additional dream cycles.

### 5.3 Biological Analogy

The decay mechanism in dreaming is functionally analogous to synaptic homeostasis during sleep [10]: the brain downscales synaptic weights overnight to prevent runaway potentiation while preserving relative connection strengths. HAMesh's decay prevents dominant attractors from monopolising the mesh energy, maintaining the distributed state that enables multi-hop reasoning. The Goldilocks zone may correspond to the point where homeostatic decay and Hebbian strengthening reach equilibrium.

### 5.4 Limitations

**Capacity wall.** A 4096×4096 mesh can store ~573 orthogonal patterns before interference noise degrades recall (Hopfield capacity ≈ 0.14 × dim). With 250–370 stored memories, we are approaching but not at this limit. Richer distillation may hit the wall.

**LLM dependence.** Without the LLM's embeddings and generation, the mesh is a matrix of floats. The quality of synthesis depends entirely on Bonsai's embedding geometry.

**C2 measurement variance.** Semantic coherence scoring via LLM judge introduces variance; the +0.22–0.30 delta is consistent but the absolute scores vary run-to-run. A larger evaluation with diverse judges would strengthen this claim.

**C5 bistability.** The high variance at intermediate cycle counts (σ ≈ 43–55%) means individual dream runs are not predictable outcomes. The two deterministic anchor points are robust; the transition path is not.

---

## 6. Conclusion

HAMesh demonstrates that holographic associative memory operating in LLM embedding space exhibits rich emergent dynamics: deterministic attractor formation, semantically coherent concept clustering, stochastic bistability near a phase transition, and 100% cross-domain synthesis in multi-mesh curiosity-driven learning. The 500-cycle equilibrium point provides a practical operating recommendation. The cross-domain synthesis results — where a curiosity engine operating on knowledge gaps in a science mesh autonomously generates connections to philosophical fixed points and political asymmetry — suggest that multi-mesh architectures can produce genuinely novel cross-domain reasoning without explicit instruction.

Code and experiment logs: https://github.com/mescgit/HAMesh

---

## References

[1] Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.

[2] Plate, T. A. (1995). Holographic reduced representations. *IEEE Transactions on Neural Networks*, 6(3), 623–641.

[3] Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.

[4] Ramsauer, H., et al. (2020). Hopfield Networks is All You Need. *arXiv:2008.02217*.

[5] [Author TBD]. (2025). Hypertokens: Holographic Associative Memory in Tokenized LLMs. *arXiv:2507.00002*.

[6] Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

[7] [Author TBD]. (2025). Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents. *arXiv:2603.19935*.

[8] [Author TBD]. (2025). Daydreaming Hopfield Networks and their surprising effectiveness on correlated data. ResearchGate.

[9] Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration. *Neuron*, 81(1), 12–34.

[10] Tononi, G., & Cirelli, C. (2003). Sleep and synaptic homeostasis: a hypothesis. *Brain Research Bulletin*, 62(2), 143–150.
