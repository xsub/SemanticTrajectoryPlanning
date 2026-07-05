# Gate 3 results — selective triggering (MuSiQue, no LLM)

**Verdict: selective triggering does NOT rescue the deployment — the bottleneck is re-ranker
strength, now quantified.** Bridge states are only *weakly* detectable (ROC-AUC 0.70), and even
at the best operating point, escalating the Bridge Score selectively **cannot beat plain dense
retrieval net** — the optimum is to never trigger it. The reason is arithmetic: at the
achievable detector precision (~0.44), the gain on true bridges (23% recovery) is smaller than
the loss on false-fired easy hops (22%), so every escalation is net-negative on average. This
turns Gate 2's open question into a concrete target: for selective triggering to pay off at this
precision, the re-ranker must recover **>28%** of bridges (vs the current 23%).

Setup: extends Gate 2 (same data / BS head). Adds a per-STATE bridge detector — a cheap,
dense-only classifier (hop depth, #retrieved, dense top-1 / top-5 / gaps / entropy of the top-20
similarity distribution, retrieved-set tightness) trained on the train split, AUC on validation.
6,404 val hop-states, 1,864 bridges (29%).

## Q1 — Are bridge states detectable?

**Weakly. ROC-AUC = 0.702.** Better than chance, so hop depth and the shape of the dense
similarity distribution do carry signal — but not sharply separable. This confirms the Gate 2
prediction: because at a bridge the dense retriever is *confidently wrong* (it returns an
in-cluster distractor at high similarity), a bridge state does not look obviously uncertain, so
a cheap detector cannot cleanly flag it.

## Q2 — Does gating BS on the detector beat dense net?

**No.** Escalate to BS where P(bridge) ≥ τ, else keep dense (net recovery@K=10):

| τ | fires on | precision | recall | **net** |
|---|---|---|---|---|
| 0.05 | 64% | 0.40 | 88% | 63.2% |
| 0.25 | 30% | 0.44 | 44% | 68.0% |
| 0.45 | 12% | 0.44 | 19% | 70.0% |
| 0.65 | 4% | 0.43 | 5% | 70.7% |
| 0.95 | 0% | — | 0% | **70.9%** |
| — | | | | dense = **70.9%** |

The net is **monotone in τ** — it rises as you fire *less*, peaking exactly where you fire on
nothing. There is no operating point that beats dense; the best selective policy is the trivial
one (never escalate).

## Why — the break-even arithmetic

Per escalation, expected net change vs dense is
  `E[Δ] = precision · R_bridge − (1 − precision) · L_nonbridge`,
where `R_bridge` = BS bridge-recovery (0.23, a true bridge dense missed, now recovered) and
`L_nonbridge` = the damage when BS is applied to an easy hop (dense 1.00 → BS 0.78, so 0.22).
Break-even needs `precision > L / (R + L) = 0.22 / 0.45 = 0.489`. The detector tops out at
**~0.44** in the useful range (and only reaches higher precision at ~0% recall), so every fire
is, on average, slightly net-negative. Hence "never fire" wins.

**The lever this exposes:** hold precision at the achievable ~0.44 and ask how strong the
re-ranker must be. Break-even then needs `R_bridge > (1−0.44)/0.44 · 0.22 ≈ 0.28`. So a
re-ranker that recovers **>28% of bridges** (vs today's 23%) would flip selective triggering
net-positive — and a stronger re-ranker also *shrinks* `L_nonbridge`, lowering the bar further.
That is the target for Gate 3b (a graph/PPR-augmented re-ranker).

## Where this leaves STP (gates 0→3)

| gate | question | result |
|---|---|---|
| 0 | Do bridges exist in real data? | Yes (~40–73% of questions). |
| 1 | Does a Bridge Score solve it (oracle)? | Yes, but ≈ 1-step lookahead; myopic EIG fails. |
| 2 | Does a *learned* BS help on real data? | Recovers 23% of missed bridges, but net-negative uniformly. |
| 3 | Can selective triggering make it net-positive? | **No — detector too soft (AUC 0.70) and BS too weak (23%); needs >28% bridge recovery.** |

Honest reading: the deployment path is **not closed but not open** — it hinges entirely on a
stronger re-ranker clearing the 28% bar. That is a falsifiable next experiment, not a vibe.

## Honest caveats

1. **Detector is cheap by design** (dense-only, so it can gate *before* paying for BS). A more
   expensive detector (e.g. one that peeks at BS's own disagreement with dense) might reach
   higher precision — but then the "cheap gate" premise is gone.
2. **Break-even uses Gate 2's point estimates** (R=0.23, L=0.22); a stronger re-ranker changes
   both, so the 28% target is a *first-order* threshold, not exact.
3. Same base caveats as Gate 2 (weak 8-feature head, teacher-forced states, single encoder,
   corpus = dev split).

## Reproduce

```bash
python gate3.py --train_limit 4000 --val_limit 2417 --model BAAI/bge-small-en-v1.5
```
Raw numbers: [`results_gate3.json`](results_gate3.json). Method: [`README.md`](README.md).
