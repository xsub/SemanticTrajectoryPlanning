# Gate 1 results — synthetic bridge simulator

**Verdict: the mechanism SURVIVES, but narrows.** Under oracle conditions on a controlled
synthetic graph: myopic information gain (what InfoGain-RAG / IGP / BALD-style rankers use)
**provably fails** to cross bridges, so a bridge-aware signal is genuinely necessary — the
program does **not** die. But an explicit **Bridge Score is statistically
indistinguishable from a 1-step lookahead EIG**. So STP's defensible claim is *not* "a novel
bridge signal"; it is "a cheap, computable reachability surrogate for shallow planning, in
the regime where myopic info-gain fails." That is a real but modest, findings-track claim.

Setup: chain of C=6 Gaussian-blob clusters on a line, one bridge node per gap, one gold node
per cluster, answer = last cluster's gold. Local kNN-frontier retrieval tuned so a cluster's
bridge is barely reachable but the next cluster's gold is not until the bridge is crossed.
Everything is **oracle** (policies know node roles / exact reachability). 200 random instances
per cell. Metric = **success** (collected all gold = crossed every bridge to the answer).

## Success rate vs `bridge_info` (how much answer-info a stepping stone carries)

| bridge_info | cosine | query | curiosity | **eig** | bs | **eig+lookahead** | **eig+bs** |
|---|---|---|---|---|---|---|---|
| **0.0** (pure stepping stone) | 0% | 0% | 9% | **0%** | 0% | **79%** | **78%** |
| 0.05 | 0% | 0% | 9% | 42% | 0% | 100% | 100% |
| 0.10 | 0% | 0% | 9% | 42% | 0% | 100% | 100% |
| 0.25 | 0% | 0% | 9% | 42% | 0% | 100% | 100% |
| 0.50 | 0% | 0% | 9% | 42% | 0% | 100% | 100% |
| 1.00 (bridge = informative) | 0% | 0% | 9% | 40% | 1% | 100% | 100% |

(`random` ≈ 0% throughout, omitted.)

Three things this shows:

1. **Similarity retrieval never crosses bridges** (cosine/query = 0% everywhere) — the gate-0
   phenomenon reproduced in miniature: a bridge is the lowest-similarity frontier node, so a
   greedy-similarity retriever never selects it within budget.
2. **Myopic information gain is insufficient.** At `bridge_info=0` it is 0%; even when bridges
   carry substantial answer-info (0.05–1.0) it only reaches ~40% — because with a fixed budget
   it spends steps on in-cluster gold and still under-crosses. This is the load-bearing result
   for STP's critique of myopic info-gain rankers.
3. **You need a non-myopic signal, and two give it:** an explicit Bridge Score (reachable-gold
   gain) *or* 1-step lookahead EIG. They are **equal** (78% vs 79% at `bridge_info=0`; 100% vs
   100% above it). Neither `eig` alone (collects, can't cross) nor `bs` alone (crosses, doesn't
   collect) works — you need collect **and** cross.

## Robustness — chain length (bridge_info = 0.0, 150 instances)

| C (clusters) | cosine | eig | curiosity | **eig+lookahead** | **eig+bs** |
|---|---|---|---|---|---|
| 4 | 0% | 0% | 7% | 89% | 91% |
| 6 | 0% | 0% | 11% | 80% | 81% |
| 8 | 0% | 0% | 8% | 64% | 65% |
| 10 | 0% | 0% | 8% | 54% | 51% |

`eig+bs` and `eig+lookahead` track each other at every length (both decay as the chain grows —
more bridges to cross under a fixed 3·C budget); `eig`/`cosine` stay at 0% throughout. The
BS ≈ lookahead equivalence is not an artifact of one configuration.

Reproduce: `for C in 4 6 8 10; do python gate1.py --C $C --bridge_infos 0.0 --trials 150; done`

## What this means for STP

- **Answers the go/no-go directly:** myopic EIG-greedy does *not* match EIG+BS (0% vs 78%), so
  the bridge mechanism is not redundant → **not a dead program.**
- **But it sharpens the honest claim:** EIG+BS ≈ EIG+lookahead, so the contribution is a
  *cheap reachability surrogate for shallow planning*, not a fundamentally new signal. The
  paper should say exactly that and add lookahead-EIG as a first-class baseline in the real
  benchmark — if a learned BS cannot beat a shallow-lookahead retriever at matched cost, it is
  only a speed/΅cost win, not an accuracy win.

## Honest caveats (why this is an existence proof, not a result)

1. **Everything is oracle.** Both EIG and BS use ground-truth node roles / reachability. Real
   systems must *estimate* these; estimation noise could blur the clean separation. This sim
   shows the mechanism is *possible*, not that it is *estimable* — that is the real benchmark's
   job.
2. **BS here gets an especially strong oracle** (it literally knows which unlocked nodes are
   gold). Part of the BS ≈ lookahead equivalence is that both receive oracle signal; the
   downstream question is *learned* BS vs *learned* lookahead.
3. **Clean geometry.** Separated blobs with a single bridge per gap; real embedding spaces are
   messier and bridges are partially reachable, which would soften the 0% similarity baselines.
4. **Tuned knn/budget.** The frontier width and step budget are set so bridges are
   barely-reachable; very different settings change absolute difficulty (the C-sweep varies one
   axis; a knn/budget sweep is worthwhile future work).

## Reproduce

```bash
python gate1.py --trials 200 --out results_gate1.json
```
Raw numbers: [`results_gate1.json`](results_gate1.json). Model + definitions:
[`README.md`](README.md).
