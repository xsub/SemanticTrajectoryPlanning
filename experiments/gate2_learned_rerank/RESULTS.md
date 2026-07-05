# Gate 2 results — learned re-ranker on real data (v2, TRUE corpus)

> **This file was rewritten after the [arc verification](../../docs/ARC_VERIFICATION.md)
> found defect D1**: the original run silently evaluated on a gold-only pool (~2.6k
> paragraphs) instead of the documented full corpus (~21k with distractors), and reported
> "learned BS recovers 23% of bridges — the mechanism transfers". **On the true corpus that
> conclusion does not survive.** The numbers below are from the corrected run
> (corpus_train=33,160, corpus_val=21,100, recorded in the JSON).

**Verdict (corrected): the cheap learned Bridge Score does NOT meaningfully transfer to the
real retrieval pool.** On the full distractor corpus it recovers only **4.6%** of bridge hops
(vs the 23% artifact on the easy gold-only pool), while demolishing easy-hop recovery
(dense 100% → bs 54% on non-bridges). Every deployment variant is deeply net-negative. The
mechanism remains oracle-proven (Gate 1), but this 8-feature embedding head is not a viable
instantiation of it. The one finding that *strengthened*: bridges are **48% of all hops**
(3,098/6,404) on the honest corpus — the problem is even bigger than Gate 0 suggested; the
cheap solution is smaller.

Setup: MuSiQue, bge-small, teacher-forced next-hop recovery; heads trained on 4,000 train
questions over the full 33k train corpus, evaluated on the full validation split (6,404
hop-states) ranking against the full 21k val corpus. No LLM anywhere.

## Recovery@K by policy (TRUE corpus)

| | bridge K=5/10 | non-bridge K=5/10 | **overall (net) K=5/10** |
|---|---|---|---|
| `dense` (baseline) | 0 / 0 | 83.8 / 100 | 43.2 / **51.6** |
| `bs` (learned: next-gold) | 1.5 / **4.6** | 50.8 / 53.6 | 26.9 / 29.9 |
| `lookahead` (learned: on-path) | 3.3 / 5.7 | 37.2 / 39.1 | 20.8 / 23.0 |
| `hybrid` (dense ∪ bs, split budget) | 0.7 / 1.5 | 74.0 / 84.6 | 38.5 / 44.4 |
| `oracle` (upper bound) | 100 / 100 | 100 / 100 | 100 / 100 |

(Random re-rank would recover ≈ K/N ≈ 10/21,100 ≈ **0.05%** of bridges, so 4.6% is still
~90× chance — the features carry *some* signal — but it is practically negligible and comes
at a ruinous cost elsewhere.)

## What changed vs the gold-only artifact run — and why it matters

| | gold-only pool (~2.6k, buggy) | **true corpus (~21k)** |
|---|---|---|
| bridge hops | 1,864 (29%) | **3,098 (48%)** |
| bs bridge recovery @10 | 22.6% | **4.6%** |
| bs non-bridge @10 | 78.0% | **53.6%** |
| net: bs vs dense | 61.9 vs 70.9 | **29.9 vs 51.6** |
| bs vs lookahead on bridges | 22.6 vs 14.8 ("sharp target wins") | 4.6 vs 5.7 (**both weak; claim withdrawn**) |

Interpretation: on the easy pool, the only competitors for a gold were *other questions'
golds*, so 8 cosine-derived features could pick out "next-gold-ness". On the real corpus the
candidate sea is dominated by same-topic distractors that look exactly like what the features
measure, and the learnable signal collapses. **The earlier "sharp next-gold target beats the
broad on-path target" finding is also withdrawn** — on the true corpus the two targets are
comparably weak on bridges (4.6 vs 5.7), and neither is usable.

## Where this leaves STP (honest, post-fix)

1. **The problem is real and large**: 48% of gold hops are not dense-retrievable from the
   accumulated context (Gate 0's phenomenon, confirmed at ranking granularity).
2. **The oracle mechanism is real** (Gate 1): reachability-seeking beats myopia under budget.
3. **The cheap learnable instantiation fails**: embedding-derived features do not encode
   "unlocks the next hop" on a realistic pool. 4.6% recovery with -46pt collateral is not a
   mechanism transfer; it is a negative result.
4. Any viable Bridge Score must therefore come from a **richer signal source** — LLM-generated
   edge semantics (HopRAG-style), entity/hyperlink structure, or a trained neural re-ranker —
   i.e. the expensive tier. And per Gate 3's corrected arithmetic, it must clear a far higher
   bar than previously stated.

## Honest caveats

1. Weak 8-feature GBM head — this is a negative result *for this cheap head*, not an
   impossibility proof. (The oracle headroom is now 95 points, underscoring how much signal
   the features fail to capture.)
2. Teacher-forced states; single encoder; single dataset; corpus = dev-split pool (~21k), still
   far smaller than production Wikipedia.
3. `hybrid`'s non-bridge value (84.6 < dense's 100) shows even a half-budget diversion to a
   weak signal is costly at K=10.

## Reproduce

```bash
python gate2.py --train_limit 4000 --val_limit 2417 --model BAAI/bge-small-en-v1.5
```
Raw numbers: [`results_gate2.json`](results_gate2.json) (includes `corpus_train`/`corpus_val`
for auditability). Method: [`README.md`](README.md).
