# Gate 0 — Do "bridge jumps" actually exist in real multi-hop data?

A cheap, LLM-free measurement that decides whether Semantic Trajectory Planning's
surviving contribution has any room to exist **before** building the method.

> **Result (MuSiQue + 2WikiMultiHopQA, 2,417 q each): PASS, decisively.** Under the
> realistic `q+last` query at K=10, questions with a bridge hop: **MuSiQue ~73%, 2Wiki ~40%**
> (conservative bge-small encoder). Robust across 2 encoders and 4 query formulations, so
> it is *not* a context-encoding artifact; MuSiQue > 2Wiki exactly as expected (MuSiQue is
> adversarially filtered against shortcuts). Full numbers + interpretation:
> [`RESULTS.md`](RESULTS.md).

## Why this experiment exists

The [verification pass](../../docs/VERIFICATION.md) reduced STP's only defensible
contribution to a narrow bet: retrieval should reward **bridge** passages — gold
evidence that is *not* a nearest neighbour of the context accumulated so far, and so
is missed by a similarity retriever.

That bet only pays off if such passages are common in the target data. If almost every
gold hop is already a top-K neighbour of the accumulated context, then plain iterative
TopK retrieval already finds the stepping stones, and there is nothing for a bridge-aware
method to win. **This script measures that fraction directly** — no LLM, no GPU, no
training, no method. It is the gate that should be passed before the synthetic pilot or
any implementation work.

## What it measures

For each question with an **ordered gold evidence chain** `g₁, …, gₘ` over a corpus:

```
context C_<i = original_question + gold_text(g₁ … g_{i-1})
hop i is a BRIDGE-JUMP at K   ⇔   g_i ∉ TopK( corpus | cos(·, enc(C_<i)) )
```

where already-found gold `{g₁ … g_{i-1}}` is excluded from the candidate ranking, and
TopK is taken over the **whole split corpus** (every unique paragraph, gold *and*
distractor) — not just the question's ~20-paragraph pool. This is the honest
"retrieval over a corpus" setting and matches the bridge-jump labeling proposed in
`POSITIONING.md §6`.

Reported per encoder and per `K ∈ {5,10,20}`:

- **hop-rate (all)** — fraction of all gold hops that are bridge-jumps
- **hop-rate (≥2)** — same, restricted to hops 2+ (the meaningful ones; hop 1 only asks
  "is the first gold a top neighbour of the bare question")
- **item-rate (≥1 bridge)** — fraction of questions with at least one bridge hop.
  **This is the headline**: STP can only compete on these items.
- breakdown by hop depth.

### Two design choices that keep the gate honest

1. **Real retrieval encoders, not a static/toy embedder.** A weak encoder inflates the
   bridge rate (gold missed for encoder reasons, not real semantic distance), which would
   make the phenomenon look real when it isn't. We use `bge-small-en-v1.5` and
   `all-MiniLM-L6-v2` — two genuine retrieval encoders — and require the signal to hold
   across both.
2. **Hop 1 as an internal control.** Because hop 1 only sees the bare question, its
   bridge-rate measures baseline encoder competence. If hop-1 ≈ 0% while hop-≥2 is high,
   the gap cannot be blamed on the encoder — it is genuine distance between the
   accumulated context and the next stepping stone. That contrast *is* the bridge
   phenomenon.

Note this is a **lower bound** on real-world prevalence: the MuSiQue dev corpus (~21k
paragraphs) is tiny next to a real Wikipedia index; a larger distractor sea can only make
the next hop harder to retrieve, not easier.

## How to read the result

| item-rate (≥1 bridge) | verdict |
|---|---|
| **< ~5%** | phenomenon absent in this data → STP has no room → **stop** |
| **~15–30%** | real phenomenon → the synthetic pilot + method are worth building |
| **> ~30%** | strongly present |

Passing this gate does **not** validate STP — it only confirms the problem exists. The
next gate (the synthetic simulator in `POSITIONING.md §6`) tests whether a bridge-aware
method actually solves it.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full MuSiQue dev, both encoders (a few minutes on CPU)
python gate0.py --dataset musique --out results_musique.json

# quick smoke test
python gate0.py --dataset musique --limit 100
```

Results are printed and written to `results_*.json`; the headline table is in
[`RESULTS.md`](RESULTS.md).

## Datasets

- **MuSiQue** (`dgslibisey/MuSiQue`) — primary. Built bottom-up by composing single-hop
  questions and adversarially filtered against disconnected reasoning, so its gold chains
  genuinely require chaining. Ordered chain via `question_decomposition →
  paragraph_support_idx`.
- **2WikiMultiHopQA** — best-effort second dataset (`--dataset 2wiki`); ordered chain via
  `evidences` / `supporting_facts`.

## Limitations

- Accumulated context uses the original question + gold passage texts only (not the
  sub-question reasoning an iterative RAG system would also carry). This is the
  conservative, retrieval-faithful formulation; adding sub-question text could lower the
  rate and is a worthwhile variant.
- The corpus is the union of dev-split paragraphs, not full Wikipedia — a lower bound
  (see above).
- Bridge-jump is defined by a *baseline encoder's* TopK miss, so the fraction is
  encoder-relative; we report two encoders to bound that.
