# Gate 4 results — graph/PPR re-ranker (MuSiQue, no LLM)

**Verdict: the cheap graph route does NOT clear the 28% bar — it makes the re-ranker worse,
and it explains why.** Adding personalized-PageRank + graph-reachability features over the
embedding kNN graph *lowers* bridge recovery (23% → 20%) and net recovery (62% → 55%) versus
Gate 2's plain 8 embedding features. The reason is structural and is the real payoff of this
gate: the kNN graph is built from the *same* embeddings, so a bridge — far from the context in
embedding space — is also far in the graph. **Graph reachability over an embedding-kNN graph is
not an independent signal; it is a noisier restatement of cosine similarity.** This is,
empirically, why HopRAG builds LLM-generated pseudo-query bridge edges rather than embedding-kNN
edges — Gate 4 rediscovers that design rationale from a controlled experiment.

Setup: extends Gate 2. Two heads trained on the SAME states — `emb8` (Gate 2's 8 embedding
features) and `emb8+graph` (+ scale-invariant graph features: PPR percentile from the explored
set, graph-hop distance, one-hop-frontier flag, degree). MuSiQue, bge-small, full val (6,404
states, 1,864 bridges). k=15 kNN graph.

## Recovery@K by policy

| | bridge K=5/10 | non-bridge K=5/10 | net K=5/10 |
|---|---|---|---|
| `dense` | 0 / 0 | 87.0 / 100 | 61.7 / 70.9 |
| `emb8` (Gate 2) | 16.0 / **22.6** | 70.7 / 78.0 | 54.8 / **61.9** |
| `emb8+graph` (scale-invariant) | 10.8 / **19.6** | 48.1 / 69.7 | 37.3 / **55.1** |
| `oracle` | 100 / 100 | 100 / 100 | 100 / 100 |

Adding graph features **hurts every metric**. Gate 3's bar (>28% bridge recovery) is not
approached — it moves the wrong way.

## Ruling out the mundane explanation

A first run used raw PPR magnitude as a feature and did even worse (bridge 18.6%, net 47.8%).
That raised a confound: PPR magnitude depends on corpus size, and the train graph (33k paragraphs)
differs from the val graph (21k), so a train/val feature-distribution shift could explain the
damage. So this run **drops raw PPR magnitude and keeps only scale-invariant graph features**
(PPR percentile, hop distance, one-hop flag, normalized degree). It still degrades the re-ranker
(bridge 23%→20%, net 62%→55%). **The scale confound is ruled out; the degradation is real.**

## Why graph reachability fails here (the substantive finding)

The candidate graph is `kNN(embeddings)`. Its edges connect embedding-*similar* nodes. A bridge
is, by definition, embedding-*dissimilar* to the accumulated context — so it is many hops away in
the very graph meant to reach it, and PPR from the explored set assigns it little mass. The graph
therefore recapitulates the similarity geometry that already fails, and its features add variance
without adding a new signal. To get an independent bridge signal the *edges* must come from
somewhere other than the same embeddings — entity links, hyperlinks, or LLM-generated
pseudo-queries (HopRAG), or a *trained* model that learns a different representation
(cross-encoder / GNN). A generic cross-encoder does not qualify: it scores relevance, and a
bridge is low-relevance by construction — it would miss bridges by the same logic as dense.

## Where this leaves STP (gates 0→4)

| gate | question | result |
|---|---|---|
| 0 | Do bridges exist? | Yes (~40–73% of questions). |
| 1 | Bridge Score under oracle? | Solves it, but ≈ 1-step lookahead; myopic EIG fails. |
| 2 | Learned BS on real data? | Recovers 23% of missed bridges; uniform re-rank net-negative. |
| 3 | Selective triggering? | No — needs >28% bridge recovery to pay off. |
| 4 | Cheap stronger re-ranker (graph/PPR)? | **No — embedding-kNN reachability degrades it; the 28% bar needs non-embedding edges (LLM) or a trained GNN/cross-encoder (GPU).** |

**Bottom line for the CPU / no-LLM budget: the investigation is complete and self-terminating.**
STP's mechanism is real (Gate 1/2) but cannot be made net-positive cheaply: the path to the >28%
bridge recovery that would flip the deployment (Gate 3) requires the expensive tier —
LLM-generated bridge edges (HopRAG-style) or a trained neural re-ranker on a GPU. That is a
precise, evidence-backed stopping point, not a vibe: we know the mechanism works, the exact bar
to clear, why the cheap fixes fail, and what the next (costly) investment must be.

## Honest caveats

1. **One graph construction.** k=15 symmetric kNN over bge-small embeddings. A different k, a
   weighted graph, or a different base encoder might shift magnitudes — but the structural
   argument (edges inherit the embedding geometry) is construction-independent.
2. **Explicit graph features, not a trained GNN.** A message-passing GNN *learns* representations
   and could in principle escape the inherited-geometry trap; that is the GPU-tier test this gate
   deliberately does not run. The claim here is narrow: cheap, hand-computed graph reachability
   over an embedding-kNN graph does not help.
3. Same base caveats as Gate 2 (teacher-forced states, single encoder/dataset, dev-split corpus).

## Reproduce

```bash
python gate4.py --train_limit 4000 --val_limit 2417 --model BAAI/bge-small-en-v1.5
```
Raw numbers: [`results_gate4.json`](results_gate4.json). Method: [`README.md`](README.md).
