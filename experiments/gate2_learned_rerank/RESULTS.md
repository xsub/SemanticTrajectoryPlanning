# Gate 2 results — learned re-ranker on real data (MuSiQue, no LLM)

**Verdict: the mechanism transfers, but no naive deployment pays off.** A cheap learned
Bridge Score recovers **23% of the bridge hops that dense retrieval misses by construction**
(from 0%) — so the signal that Gate 1 showed under oracle conditions is really present in
embedding features on real data. **But** applying it as a re-ranker is **net-negative**: it
demotes enough easy (non-bridge) golds that overall recovery drops from 71% (dense) to 62%
(uniform BS) — and a budget-split hybrid only claws back to 69%, still below dense. The honest
conclusion: STP's mechanism is real and non-trivial, but it only helps end-to-end under a
**selective trigger** (invoke BS only when dense is stuck), which this experiment did not
build — and which is hard, because at a bridge the dense retriever is *confidently wrong*, so a
naive confidence gate may not fire.

Setup: MuSiQue, encoder bge-small-en-v1.5, teacher-forced next-hop recovery (same framing as
Gate 0). Train two gradient-boosted heads on 4,000 train questions (652k candidate rows);
evaluate on the full validation split — 6,404 hop-states, of which **1,864 are bridge hops**
(gold not in dense top-10) and 4,540 are non-bridge. All embedding-only; no LLM anywhere.

## Recovery@K by policy

| | bridge K=5/10 | non-bridge K=5/10 | **overall (net) K=5/10** |
|---|---|---|---|
| `dense` (baseline) | 0 / 0 | 87.0 / 100 | 61.7 / **70.9** |
| `bs` (learned: next-gold) | 16.0 / **22.6** | 70.7 / 78.0 | 54.8 / 61.9 |
| `lookahead` (learned: on-path) | 10.1 / 14.8 | 63.5 / 70.4 | 48.0 / 54.2 |
| `hybrid` (dense ∪ bs, split budget) | 12.0 / 16.0 | 81.8 / 91.1 | 61.5 / **69.2** |
| `oracle` (upper bound) | 100 / 100 | 100 / 100 | 100 / 100 |

(`dense` = 0% on bridges by construction — a bridge is *defined* as a dense top-K miss. Random
re-rank would recover ≈ K/N ≈ 0.05% of them, so 23% is a real signal, not chance.)

## What the numbers say

1. **The mechanism transfers to real data.** Learned BS lifts bridge recovery from 0% to 23%
   at K=10 (16% at K=5). The oracle sits at 100%, so ~77% of bridges remain unrecovered — the
   8-feature GBM head is a weak model; a stronger re-ranker (cross-encoder, GNN, richer
   features) is obvious headroom. But the signal is unambiguously there.
2. **`bs` > `lookahead` (23% vs 15%).** The sharp "is this the next gold" target beats the
   broad "is this anywhere on the remaining path" target — the broader label dilutes the
   immediate-next-hop signal. This is a *nuance on Gate 1*: under oracle scoring BS ≈ lookahead,
   but as **learned targets** the tighter objective wins. It does not overturn Gate 1 (different
   operationalization), but it is a real, honest difference worth reporting.
3. **No naive deployment beats dense net.** Uniform BS re-rank: 62% vs dense 71% — the bridge
   gain (+23% on 29% of hops) is smaller than the collateral loss on the 71% of easy hops it
   perturbs. The budget-split `hybrid` protects the easy hops much better (non-bridge 91%) and
   nearly breaks even (69.2%), but still does not beat plain dense. **Bridge recovery is not
   free**, and a global re-rank spends more than it earns.

## The honest deployment problem (the real open question)

To be net-positive, BS must be applied **only where dense fails** — an Adaptive-RAG-style
selective trigger. The obstacle this experiment exposes: at a bridge, dense does not *look*
uncertain — it confidently retrieves an in-cluster distractor with high similarity. So a simple
"dense top-1 confidence is low → escalate to BS" gate may not separate bridge states from easy
states. A working selective policy likely needs a *learned bridge-state detector* (predict "am
I stuck?" from the state, not the candidate) — which is the natural Gate 3 and is left as
future work here.

## Where this leaves STP (across all three gates)

| gate | question | result |
|---|---|---|
| 0 | Do bridges exist in real data? | **Yes** — ~40–73% of questions have a bridge hop (q+last, K=10). |
| 1 | Does a Bridge Score solve it (oracle)? | **Yes, but ≈ 1-step lookahead**; myopic info-gain fails. |
| 2 | Does a *learned* BS help on real data? | **Recovers 23% of missed bridges, but net-negative under naive deployment**; needs a selective trigger (unbuilt). |

Net: the contribution is real but **narrow and conditional** — "a learned bridge re-ranker
recovers a fifth of otherwise-missed bridge hops; making that a net win requires selective
triggering." An honest findings-track result, not a paradigm.

## Honest caveats

1. **Weak re-ranker.** 8 hand-features + a GBM; oracle shows 77% headroom. A stronger model
   could change the net calculus — the negative net result is *for this cheap head*, not a
   proof that no learned BS can win.
2. **Teacher-forced states.** States use the true gold prefix (isolates scoring from trajectory
   error). Full rollout would compound errors and is a separate measurement.
3. **Single encoder, single dataset, corpus = dev split** (~21k). Gate 0's encoder-sensitivity
   threat applies: a stronger retriever would leave fewer bridges to recover.
4. **Selective trigger not built** — and shown to be non-trivial (dense is confidently wrong at
   bridges).

## Reproduce

```bash
python gate2.py --train_limit 4000 --val_limit 2417 --model BAAI/bge-small-en-v1.5
```
Raw numbers: [`results_gate2.json`](results_gate2.json). Method: [`README.md`](README.md).
