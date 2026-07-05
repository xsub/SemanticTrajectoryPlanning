# Gate 3 results — selective triggering (v2, TRUE corpus)

> **Rewritten after the [arc verification](../../docs/ARC_VERIFICATION.md)** (defects D1 —
> gold-only corpus, and D2 — break-even computed from population averages instead of the
> fired subset). Both fixes are applied: the run below uses the full 33k/21k corpora, and the
> sweep now reports the **empirical** fired-subset gain/loss (`R_fired`/`L_fired`) per
> threshold, from which the break-even is derived.

**Verdict (corrected): the selective-triggering door is closed for this head — decisively.**
The bridge-state detector actually got *better* on the true corpus (ROC-AUC **0.736**), but
the corrected, empirical break-even now requires trigger precision **≈ 0.94** (fired-subset
gain R ≈ 0.05–0.10 vs loss L ≈ 0.64–0.92), while the detector delivers ≈ 0.66–0.75. No
threshold beats plain dense retrieval (best selective 51.4% vs dense 51.6%); the optimum
remains "never fire". The original ">28% bridge recovery" target — already flagged by the
verification as understated (~44%) — is itself obsolete: with the true-corpus loss profile,
a *selectively-triggered* cheap BS would need to recover the large majority of bridges it
fires on to pay for its collateral, which is out of reach of any embedding-feature head
(Gate 2: 4.6%).

Setup: extends Gate 2 (same data/head, true corpora). Detector = cheap dense-only state
classifier (hop depth, top-1/top-5 similarity shape, entropy, retrieved-set tightness),
trained on train, evaluated on val: 6,404 states, **3,098 bridges (48%)**.

## Q1 — Are bridge states detectable?

**Yes, moderately: ROC-AUC = 0.736** (up from 0.702 on the gold-only pool — with a realistic
candidate sea, stuck-states look more distinctive). Depth and the dense-similarity
distribution carry real signal. Detection is not the bottleneck.

## Q2 — Does gating BS on the detector beat dense net?

**No.** Escalate to BS where P(bridge) ≥ τ (net recovery@K=10; `R_fired`/`L_fired` are the
empirical per-escalation gain on true bridges / loss on false fires):

| τ | fires on | precision | R_fired | L_fired | net |
|---|---|---|---|---|---|
| 0.05 | 82% | 0.57 | 0.05 | 0.64 | 30.9% |
| 0.25 | 60% | 0.65 | 0.05 | 0.84 | 35.7% |
| 0.45 | 39% | 0.66 | 0.06 | 0.91 | 41.0% |
| 0.65 | 18% | 0.68 | 0.07 | 0.92 | 47.1% |
| 0.85 | 4% | 0.67 | 0.08 | 0.91 | 50.6% |
| 0.95 | 1% | 0.75 | 0.10 | 0.86 | **51.4%** |
| — | | | | | dense = **51.6%** |

Net rises monotonically as you fire less and never crosses dense. The empirical break-even,
`precision* = L/(R+L)` on the fired subset, is **≈ 0.94** — every escalation must be almost
certainly a bridge to pay off, because the gain when right is tiny (R≈0.06) and the loss when
wrong is enormous (L≈0.86).

## Why the door is closed (corrected arithmetic)

The verification (D2) showed the original break-even wrongly used the population-average loss
(0.22), giving a flattering ">28% recovery" target. The empirical fired-subset numbers are
far harsher: the detector preferentially fires on hard, bridge-*like* states, where applying
a weak BS re-ranker destroys dense's otherwise-perfect easy-hop recovery (L up to 0.92, i.e.
dense would have recovered 92% of those false-fired states and BS recovers almost none). With
R this small, no realistic detector precision rescues the policy; the only fix is a
fundamentally stronger BS (large R), which Gate 2 shows embedding features cannot provide.

## Where this leaves STP (gates 0→3, post-fix)

| gate | question | corrected result |
|---|---|---|
| 0 | Do bridges exist? | Yes — and on the true corpus they are **48% of hops**. |
| 1 | Oracle mechanism? | Frontier-stepping reachability beats myopia under budget; BS ≈ lookahead. |
| 2 | Cheap learned BS? | **Fails on the true corpus** (4.6% recovery, ruinous collateral). |
| 3 | Selective triggering? | **Closed**: empirical break-even precision ≈0.94 vs achievable ≈0.7. |

## Honest caveats

1. Break-even numbers are specific to this (weak) BS head; a fundamentally stronger BS would
   change both R and L. The 0.94 is the bar *for rescuing Gate 2's head*, not a universal law.
2. Detector is deliberately cheap (dense-only features, so it can gate before paying for BS).
3. Same base caveats as Gate 2 (teacher-forced, single encoder/dataset, dev-split corpus).

## Reproduce

```bash
python gate3.py --train_limit 4000 --val_limit 2417 --model BAAI/bge-small-en-v1.5
```
Raw numbers: [`results_gate3.json`](results_gate3.json) (sweep includes per-τ
`R_fired`/`L_fired`). Method: [`README.md`](README.md).
