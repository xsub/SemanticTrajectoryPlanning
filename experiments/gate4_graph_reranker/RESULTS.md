# Gate 4 results — graph/PPR re-ranker (v2, TRUE corpus + noise control)

> **Rewritten after the [arc verification](../../docs/ARC_VERIFICATION.md)** (defects D1 —
> gold-only corpus; D3 — an unreproducible "scale-confound rebuttal", now deleted; D6 —
> mislabeled graph sizes; plus the missing noise-feature control, now implemented). The run
> below uses the true corpora (kNN graphs over 33,160 train / 21,100 val paragraphs) and adds
> an `emb8+noise` control head (same GBM, 4 pure-noise columns) that the earlier version
> lacked.

**Verdict (corrected, and humbler than v1): graph/PPR features do not help — and the noise
control shows the earlier "graph features actively mislead" story was wrong.** On the true
corpus, bridge recovery is statistically flat across heads (emb8 4.6%, emb8+graph 5.2%,
emb8+noise 6.0% — all negligible), while net recovery collapses for *both* augmented heads
(emb8 29.9% → graph 13.7%, noise 9.6%). Since **pure noise degrades the head even more than
graph features do**, the degradation is a *generic added-feature fragility* of this GBM on
this task — not evidence that graph reachability is a uniquely misleading signal. The v1
"inherited geometry" causal narrative is therefore **withdrawn as unproven**: it remains a
plausible design consideration (the kNN graph is built from the same embeddings whose
geometry defines the problem), but this experiment cannot distinguish it from ordinary
feature-noise fragility.

Setup: extends Gate 2 (true corpora). Symmetric k=15 kNN graph over corpus embeddings;
scale-invariant graph features (PPR percentile seeded at the explored set, hop distance,
one-hop flag, degree). Three heads trained on identical states/targets: `emb8`,
`emb8+graph`, `emb8+noise`. 6,404 val states, 3,098 bridges (48%).

## Recovery@K by policy (TRUE corpus)

| | bridge K=5/10 | non-bridge K=5/10 | net K=5/10 |
|---|---|---|---|
| `dense` | 0 / 0 | 83.8 / 100 | 43.2 / 51.6 |
| `emb8` (Gate 2) | 1.5 / 4.6 | 50.8 / 53.6 | 26.9 / **29.9** |
| `emb8+graph` | 2.7 / 5.2 | 5.7 / 21.7 | 4.3 / **13.7** |
| `emb8+noise` (control) | 2.6 / 6.0 | 5.4 / 12.9 | 4.0 / **9.6** |
| `oracle` | 100 / 100 | 100 / 100 | 100 / 100 |

Three readings:

1. **No bridge-recovery gain.** 4.6 → 5.2 with graph features (noise "gains" to 6.0) — all
   three heads are equally unable to find bridges in a realistic pool; differences are noise
   around a floor.
2. **The noise control is decisive against the v1 story.** If graph features *specifically*
   misled the model, `emb8+noise` should sit near `emb8` while `emb8+graph` drops. Instead
   noise drops *further* (9.6 vs 13.7 net). The honest conclusion: this GBM head, trained on
   652k rows with 8 informative features, loses most of its non-bridge precision when any 4
   uninformative columns are appended — a capacity/regularization artifact, not signal
   sabotage.
3. **Nothing here approaches usefulness.** Gate 3's corrected arithmetic needs a
   fundamentally stronger BS (fired-subset gain ~0.9 needed vs ~0.06 achieved); cheap graph
   features move nothing.

## What was deleted from v1 (for the record)

- The "scale-confound rebuttal" quoting bridge 18.6% / net 47.8% from a "raw-PPR first run":
  those numbers came from a pre-commit version of the code and had no committed artifact
  (verification D3). No conclusion rested on them; they are gone.
- The claim "graph reachability is a noisier restatement of cosine — this is why HopRAG uses
  LLM edges". Retained only as a **hypothesis** below, since the noise control shows this
  experiment cannot substantiate it.

## The (unproven) design hypothesis, stated honestly

It remains *plausible* that reachability over an embedding-kNN graph cannot carry independent
bridge signal, because its edges are constructed from the very similarity that defines a
bridge as distant — and it is consistent with HopRAG's design choice of building edges from
**LLM-generated pseudo-queries rather than raw passage-embedding kNN** (note: HopRAG's edge
*matching* is hybrid — keyword overlap + embedding similarity of pseudo-queries — so LLM
semantics relocate rather than eliminate embeddings; verification D9). But testing that
hypothesis requires a model that can actually exploit graph structure (a trained GNN) or
edges from a non-embedding source — both GPU/LLM-tier experiments this arc does not run.

## Where this leaves STP (gates 0→4, final, post-fix)

| gate | question | corrected result |
|---|---|---|
| 0 | Do bridges exist? | **Yes** — 37–73% of questions (dataset-dependent); 48% of hops on the true MuSiQue pool. |
| 1 | Oracle mechanism? | Reachability-seeking beats myopia under budget (frontier-stepping, not literal bridge-crossing); BS ≈ 1-step lookahead. |
| 2 | Cheap learned BS? | **No** — 4.6% bridge recovery on the true corpus with ruinous collateral (the earlier 23% was the D1 artifact). |
| 3 | Selective triggering? | **No** — empirical break-even precision ≈0.94, unreachable. |
| 4 | Cheap graph/PPR upgrade? | **No** — no bridge gain; net degradation shown by the noise control to be generic feature fragility, not a graph-specific effect. |

**Final bottom line: on the honest corpus, the bridge problem is large (≈half of all gold
hops) and the oracle mechanism is real, but every cheap (CPU, no-LLM, embedding-feature)
instantiation of a Bridge Score failed.** What would clear the bar is exactly what this arc
cannot afford to test: LLM-derived edge semantics, or a trained neural re-ranker /
bridge-state detector. That is a precise, evidence-backed stopping point — the negative
results are the contribution.

## Honest caveats

1. One graph construction (k=15, bge-small); one GBM capacity setting — the feature-fragility
   finding itself suggests results are sensitive to head capacity/regularization.
2. First-hop PPR is seeded from dense top-5 (no explored set yet), so first-hop graph features
   are cosine-seeded by construction.
3. Same base caveats as Gate 2 (teacher-forced, single encoder/dataset, dev-split corpus).
4. recovery@K is necessary-but-not-sufficient for end-to-end answer-F1 gains; no gate in this
   arc measures answer quality.

## Reproduce

```bash
python gate4.py --train_limit 4000 --val_limit 2417 --model BAAI/bge-small-en-v1.5
```
Raw numbers: [`results_gate4.json`](results_gate4.json) (includes corpus sizes + the
`emb8+noise` policy). Method: [`README.md`](README.md).
