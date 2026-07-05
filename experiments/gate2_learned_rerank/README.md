# Gate 2 — Can a *learned* Bridge Score recover bridges on real data? (no LLM)

The real-data, oracle-free test that follows Gate 0 (bridges exist) and Gate 1 (a Bridge Score
solves them under oracle conditions, but ≈ lookahead).

> **Result (v2, TRUE corpus — corrected after [ARC_VERIFICATION](../../docs/ARC_VERIFICATION.md)
> D1): the cheap learned Bridge Score does NOT transfer.** On the full ~21k distractor corpus it
> recovers only **4.6%** of bridge hops (the previously-reported 23% was an artifact of a
> gold-only ~2.6k pool) while collapsing easy-hop recovery (100% → 54%); every deployment
> variant is deeply net-negative (bs 30% vs dense 52%). Bridges are **48% of hops** on the
> honest pool — the problem grew, the cheap solution shrank. This is a negative result for
> embedding-feature Bridge Scores, not for the oracle mechanism (Gate 1). Full numbers + the
> before/after comparison: [`RESULTS.md`](RESULTS.md).

## Why this experiment exists

[Gate 1](../gate1_synthetic_bridge/RESULTS.md) showed the mechanism works under oracle
conditions on a synthetic graph. The obvious objection: real signals are noisy and must be
*learned*. Gate 2 drops the oracle — it trains a cheap re-ranker on real multi-hop data and asks
whether it recovers the bridge hops that plain dense retrieval structurally misses.

## Measurement (teacher-forced next-hop recovery — same framing as Gate 0)

For each question with ordered gold chain `g₁…gₘ` and each hop `t`, the state is the true
prefix `C_<t = question + gold(g₁…g_{t-1})`, and the target is `g_t`, ranked against the whole
corpus (already-collected gold excluded). A hop is a **bridge** iff `g_t` is not in the dense
top-K of `C_<t` — so **dense recovers 0% of bridges by construction**, and the question is what
a learned score recovers. Teacher-forcing isolates *scoring quality* from trajectory
error-compounding.

## Policies (all embedding-only, no LLM)

| policy | score | role |
|---|---|---|
| `dense` | cos(candidate, state query) | baseline; 0% on bridges by definition |
| `bs` | learned head: P(candidate = next gold `g_t`) | the Bridge Score |
| `lookahead` | learned head: P(candidate on remaining path `g_t…gₘ`) | broader reachability target |
| `hybrid` | dense top-⌈K/2⌉ ∪ bs top-⌈K/2⌉ | budget-split selective deployment |
| `oracle` | candidate == `g_t` | upper bound / sanity |

**Features** (per candidate, cheap): cos to state, to question, to last gold; max/mean/min cos
to retrieved; novelty; relevance-minus-redundancy. **Heads:** gradient-boosted trees (sklearn),
trained on the train split with sampled hard + random negatives.

## How to read it

- `bs` bridge recovery ≫ 0 → the mechanism transfers to real data.
- `bs` overall (net) vs `dense` → whether re-ranking is a net win or a net loss.
- `hybrid` overall → whether a budget-split selective deployment recovers the loss.
- `bs` vs `lookahead` → whether the sharp next-gold target beats the broad on-path target
  (a nuance on Gate 1's oracle equivalence).

## Run it

```bash
# reuses the Gate 0 venv (sentence-transformers + sklearn); a few minutes on CPU
../gate0_bridge_jump/.venv/bin/python gate2.py --train_limit 4000 --val_limit 2417 \
    --model BAAI/bge-small-en-v1.5 --out results_gate2.json
```

## Limitations

Cheap 8-feature GBM head (oracle shows ~77% headroom — a stronger model could change the net
result), teacher-forced states, single encoder / dataset, corpus = dev split (~21k), and no
selective trigger (shown to be non-trivial). This measures whether the *signal exists and is
learnable*, not whether a production system built on it wins — see [`RESULTS.md`](RESULTS.md)
§ deployment problem.
