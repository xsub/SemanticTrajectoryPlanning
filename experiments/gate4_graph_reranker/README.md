# Gate 4 — Does a graph/PPR-augmented re-ranker clear Gate 3's 28% bar?

The "stronger re-ranker" test. Gate 3 showed selective triggering needs the Bridge Score to
recover >28% of bridges (vs Gate 2's 23%) to pay off. This gate tries the cheap, CPU-feasible
route to a stronger signal.

> **Result: no — and it explains why.** Adding personalized-PageRank + graph-reachability
> features over the embedding kNN graph *lowers* bridge recovery (23% → 20%) and net recovery
> (62% → 55%). Root cause: the kNN graph is built from the same embeddings, so a bridge is far
> in the graph too — graph reachability is a noisier restatement of cosine, not a new signal.
> This is empirically *why HopRAG uses LLM-generated bridge edges* rather than embedding-kNN
> edges. Clearing 28% needs non-embedding edges (LLM) or a trained GNN/cross-encoder (GPU).
> Full numbers + the scale-confound control: [`RESULTS.md`](RESULTS.md).

## Why this experiment exists

[Gate 3](../gate3_selective_trigger/RESULTS.md) quantified the bottleneck: to make selective
triggering net-positive, the re-ranker must recover **>28%** of bridges. The obvious cheap
attempt is to make the Bridge Score faithful to POSITIONING §5 — where it is defined as a
*graph-reachability* quantity, not a similarity — by adding graph/PPR features.

## Why graph/PPR, not a cross-encoder?

A cross-encoder scores query–document **relevance**, and a bridge is low-relevance to the
accumulated context by construction — so a cross-encoder would miss bridges by the same logic as
dense retrieval. The principled stronger signal is **reachability from the explored set over a
graph** (the actual Bridge Score functional). A *trained* GNN is the GPU-tier version of the same
idea; this gate computes the graph features explicitly on CPU.

## Method

Extends Gate 2. Builds a symmetric k-NN graph over the corpus embeddings and adds, per candidate:

    ppr        personalized-PageRank percentile, seeded at the explored/retrieved set
    graph_hop  min hops from the explored set on the kNN graph (capped)
    onehop     is the candidate in the kNN of any explored node
    degree     global kNN degree (normalized)

Two heads are trained on the **same** states and compared head-to-head: `emb8` (Gate 2's 8
embedding features) and `emb8+graph`. Metric = bridge / non-bridge / net recovery@K, teacher-forced.

A first version also used raw PPR *magnitude*; because that depends on corpus size (train graph
33k vs val graph 21k) it introduced a train/val distribution shift, so the reported run uses
**scale-invariant graph features only** — and the degradation persists, ruling that confound out.

## How to read it

| outcome | meaning |
|---|---|
| `emb8+graph` bridge ≥ 28% and > `emb8` | graph route works → selective triggering viable |
| `emb8+graph` > `emb8` but < 28% | helps, not enough |
| `emb8+graph` ≤ `emb8` | graph reachability isn't an independent bridge signal (observed) |

## Run it

```bash
../gate0_bridge_jump/.venv/bin/python gate4.py --train_limit 4000 --val_limit 2417 \
    --model BAAI/bge-small-en-v1.5 --out results_gate4.json
```

## Limitations

One graph construction (k=15 kNN over bge-small); explicit hand-computed graph features, **not** a
trained GNN (a message-passing GNN learns representations and could escape the inherited-geometry
trap — that is the GPU-tier test this gate does not run). The claim is narrow and honest: cheap
graph reachability over an embedding-kNN graph does not clear the bar. Same base caveats as Gate 2.
