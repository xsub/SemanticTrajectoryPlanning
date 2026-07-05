# Gate 1 results — synthetic simulator (v2, corrected after adversarial verification)

> **This file was rewritten after the [arc verification](../../docs/ARC_VERIFICATION.md)
> (findings D4, D5) and two new committed controls.** The numeric headline table is unchanged
> and reproduces exactly; what changed is the *causal story*: the original prose claimed the
> separation comes from "crossing instrumental bridges", and the no-bridge control proves it
> does not. The corrected mechanism is below.

**Verdict (corrected): at matched budget, a reachability-seeking signal (Bridge Score or
1-step lookahead EIG) dominates myopic information gain in clustered geometry — but the
mechanism is *frontier-stepping*, not bridge-crossing, and the myopic failure is budget
starvation, not structural impossibility.** BS remains statistically indistinguishable from
1-step lookahead everywhere, so the honest claim is unchanged: BS is a cheap surrogate for
shallow planning, not a novel signal.

Setup: chain of C=6 Gaussian-blob clusters, one designated bridge node per gap, one gold node
per cluster, answer = last cluster's gold; local kNN-frontier retrieval, budget 3·C=18; oracle
signals; 200 random instances per cell. Metric = success (collected all gold).

## Headline table (unchanged, reproduces from `results_gate1.json`)

| bridge_info | cosine | query | curiosity | **eig** | bs | **eig+lookahead** | **eig+bs** |
|---|---|---|---|---|---|---|---|
| **0.0** | 0% | 0% | 9% | **0%** | 0% | **79%** | **78%** |
| 0.05–0.5 | 0% | 0% | 9% | 42% | 0% | 100% | 100% |
| 1.00 | 0% | 0% | 9% | 40% | 1% | 100% | 100% |

`eig+bs` vs `eig+lookahead`: 78% vs 79% at n=200 is a paired z≈0.24 — statistically
indistinguishable, and this equivalence is robust across chain lengths C=4..10.

## The two committed controls (added after verification)

### Control 1 — no bridges at all (`--no_bridges`, D5)

Remove every bridge node from the instance. If the separation came from "crossing a bridge you
must cross", `eig+bs` should collapse toward `eig`.

| | eig | eig+bs | eig+lookahead |
|---|---|---|---|
| with bridges (original) | 0% | 78% | 79% |
| **no bridges at all** | 0% | **69%** | **70%** |

It does not collapse. **The bridges contribute only ~9 points of the ~78-point separation.**
Root cause: with `knn = n+3 = 33 ≥ cluster size 30`, each cluster's frontier already reaches
several plain nodes of the next cluster directly — the designated bridge node is *not* a cut
vertex. What BS/lookahead actually reward is **stepping onto the next cluster's frontier**
(any first-contact node), which unlocks that cluster's gold. The corrected mechanism
statement: *reachability-seeking beats myopia in clustered geometry because it values
zero-information first-steps into unexplored clusters; a literal "bridge" is one instance of
such a step, not the load-bearing object.*

### Control 2 — budget sweep (D4)

The original prose said myopic EIG "provably fails". Wrong kind of claim: at `bridge_info=0`
the whole frontier ties at info=0 once a cluster's gold is collected, so `eig` degenerates to
a random walk and *starves at budget 18*. Give the identical policy more budget:

| budget | eig | eig+bs | eig+lookahead |
|---|---|---|---|
| 18 (=3·C, matched) | 0% | 78% | 79% |
| 100 (≈5.5×) | **74%** | 100% | 100% |

So the correct claim is **budget-efficiency, not impossibility**: at matched budget myopic EIG
collects in-cluster gold but cannot afford the random-walk detour to the next cluster; a
reachability term buys the crossing within budget. (This is still the STP-relevant regime —
real retrieval runs under tight budgets — but "provably fails" is withdrawn.)

## What survives, precisely

1. **At matched budget, similarity and myopic info-gain policies score 0%** while
   reachability-seeking policies score ~78% — a real, large, correctly-powered separation.
2. **BS ≈ 1-step lookahead** (paired z≈0.24; robust across C) — BS is a cheap surrogate for
   shallow planning, not a distinct signal. Unchanged.
3. The failure of myopic EIG is **starvation under budget**, and the success mechanism is
   **frontier-stepping**, with designated bridges contributing only marginally (69→78).

## Honest caveats (expanded after verification)

1. **Everything is oracle** — existence proof, not evidence of learnability (Gate 2's job).
2. **`eig+bs` receives a reachability oracle that plain `eig` is denied**, so the comparison
   conflates "myopic vs non-myopic" with "no oracle vs reachability oracle". There is no
   reachability-without-gold-identity middle baseline; the learned-data analogue is Gate 2.
3. **The similarity baselines are structurally weak by design**: `query` scores similarity to
   the fixed seed only (can never leave cluster 0), and `cosine` feeds on an over-dense
   in-cluster frontier. Their 0% is a consequence of the clustered geometry + budget, and
   should be read as "similarity has no drive to leave a cluster", not as a strong-baseline
   defeat.
4. **Clean geometry, one configuration family** (blobs on a line; knn/budget tuned so
   next-cluster gold is initially unreachable). The C-sweep varies one axis; knn/sep/sigma
   sweeps are future work.

## Reproduce

```bash
python gate1.py --trials 200 --out results_gate1.json                    # headline
python gate1.py --trials 200 --bridge_infos 0.0 --no_bridges \
    --out results_gate1_nobridges.json                                   # control 1 (D5)
python gate1.py --trials 200 --bridge_infos 0.0 --budget 100 \
    --out results_gate1_budget100.json                                   # control 2 (D4)
```
Raw numbers: [`results_gate1.json`](results_gate1.json),
[`results_gate1_nobridges.json`](results_gate1_nobridges.json),
[`results_gate1_budget100.json`](results_gate1_budget100.json).
