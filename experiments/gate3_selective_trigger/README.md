# Gate 3 — Can bridge states be detected and the Bridge Score triggered selectively?

The deployment test after Gate 2 showed a uniform Bridge Score re-rank is net-negative.

> **Result (v2, TRUE corpus — corrected after [ARC_VERIFICATION](../../docs/ARC_VERIFICATION.md)
> D1+D2): no — the selective-triggering door is closed, decisively.** The detector improved
> (ROC-AUC 0.736; detection is not the bottleneck), but the corrected, *empirical* fired-subset
> arithmetic requires trigger precision ≈**0.94** (gain when right R≈0.06, loss when wrong
> L≈0.86) while the detector achieves ≈0.7. No threshold beats dense (51.4% vs 51.6%); the
> optimum is "never fire". The v1 ">28% recovery" target was an artifact of population-average
> arithmetic (D2) and is obsolete. Full sweep with per-τ R/L: [`RESULTS.md`](RESULTS.md).

## Why this experiment exists

[Gate 2](../gate2_learned_rerank/RESULTS.md): a learned Bridge Score recovers 23% of missed
bridges but a uniform re-rank is net-negative — it demotes easy hops. The obvious fix is
selective: run dense normally, escalate to BS only when the retriever is *stuck on a bridge*.
Gate 3 tests whether that is possible, and whether it pays off.

## Two questions

- **Q1 — detectability.** Can a bridge state be flagged cheaply (dense-only, before paying for
  BS), given that at a bridge the dense retriever is *confidently wrong*?
- **Q2 — payoff.** Does gating BS on that detector beat plain dense retrieval net?

## Method

Reuses Gate 2's per-candidate Bridge Score head. Adds a per-STATE detector: a gradient-boosted
classifier on cheap dense-only features —

    hop depth, #retrieved, dense top-1 sim, top-5 mean, top1−top5 gap, top1−top10 gap,
    entropy of the top-20 similarity distribution, retrieved-set tightness, state-vs-last sim

trained on the train split to predict `is_bridge` (next gold not in dense top-K), AUC reported
on validation. The **selective policy** at threshold τ: escalate to the BS re-ranker where
P(bridge) ≥ τ, else keep dense. Sweep τ and read the net-recovery frontier.

## How to read it

| signal | meaning |
|---|---|
| detector AUC ≈ 0.5 | bridge states not detectable → selective path blocked |
| best selective net > dense net | selective triggering wins → deployment path exists |
| best selective net ≤ dense net | detector/​re-ranker too weak to pay off even when targeted |

The observed outcome is the third row, and `RESULTS.md` derives the exact bridge-recovery target
(>28%) that would flip it.

## Run it

```bash
../gate0_bridge_jump/.venv/bin/python gate3.py --train_limit 4000 --val_limit 2417 \
    --model BAAI/bge-small-en-v1.5 --out results_gate3.json
```

## Limitations

The detector is deliberately cheap (dense-only, to gate before spending on BS); a costlier
detector might reach higher precision but loses the "cheap gate" premise. The 28% break-even
target uses Gate 2's point estimates and is first-order. Same base caveats as Gate 2 (weak
8-feature head, teacher-forced states, single encoder, dev-split corpus).
