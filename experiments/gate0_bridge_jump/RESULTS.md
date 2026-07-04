# Gate 0 results — MuSiQue dev (2,417 questions)

**Verdict: PASS, decisively.** Bridge hops — gold evidence not retrievable by similarity
from the context so far — are pervasive in MuSiQue and are **not** an artifact of how the
growing context is turned into a query. The premise behind STP's surviving contribution
holds: there is large headroom for a bridge-aware retriever. This confirms the *problem*
exists; it does not yet validate STP's *method* (that is Gate 1, the synthetic simulator).

Setup: full MuSiQue validation split, corpus = 21,100 unique paragraphs (gold +
distractors), TopK over the whole corpus, already-found gold excluded. Two retrieval
encoders, four query formulations. Metric = **% of questions with ≥1 bridge hop**.

## Headline — % questions with ≥1 bridge hop

| query mode | bge-small K=5/10/20 | MiniLM K=5/10/20 |
|---|---|---|
| `concat` (Q + all prev golds) | 80.5 / 73.6 / 66.2 | 84.4 / 77.7 / 68.3 |
| `question` (Q only) | 84.9 / 77.8 / 71.1 | 87.4 / 81.6 / 74.1 |
| `last` (prev gold only) | 87.5 / 82.8 / 77.8 | 89.6 / 86.3 / 80.7 |
| **`q+last` (Q + prev gold)** — realistic, conservative | **80.2 / 72.7 / 64.1** | **83.7 / 76.3 / 66.2** |

The honest headline is the **most realistic iterative-RAG query, `q+last`: ~73–76% of
questions contain a bridge hop at K=10** (~64–66% even at K=20). That is far above the
~15–30% "worth building" band and orders of magnitude above the ~5% "stop" line.

## The artifact check (the reason this result is trustworthy)

The worry was that a growing context, encoded into one dense vector and **truncated** by
the encoder, would fail to point at the next hop for mechanical reasons — inflating
"bridges" that aren't real. If that were the cause, the truncation-free modes would be
*lower*. They are not:

- `q+last` (no long concatenation) ≈ `concat`: **72.7% vs 73.6%** (bge, K=10).
- `last` (query = only the immediately preceding stepping stone, zero truncation) is
  **higher**: **82.8%** (bge, K=10).

So the effect is **robust across query formulations and grows when you query from the
most recent stepping stone** — exactly the signature of a genuine bridge (the previous
gold is semantically far from the next), not a context-encoding artifact. **Artifact
hypothesis refuted.**

## Depth gradient — the corpus-size-robust core signal

% bridge hops by hop depth, K=10 (hop 1 sees only the bare question, so it is the
encoder's baseline miss rate — an internal control):

| mode | h1 | h2 | h3 | h4+ |
|---|---|---|---|---|
| `last` (bge) | 20% | 73% | 87% | 80% |
| `q+last` (bge) | 20% | 57% | 75% | 65% |
| `last` (MiniLM) | 25% | 76% | 88% | 85% |
| `q+last` (MiniLM) | 25% | 62% | 66% | 68% |

Hop 1 misses ~20–25% (raw retrieval difficulty over 21k paragraphs); hops 2+ miss
**3–4× more**. A world where "retrieval is just hard" would be flat across depth. The
steep gradient is the bridge phenomenon: later hops need context the encoder cannot use.
Because both encoders agree and the gradient survives every query mode, this is the part
of the result that does **not** depend on the absolute corpus size.

## Honest caveats

1. **Absolute % scales with corpus size.** At a 973-paragraph corpus (50-question smoke)
   the K=10 `q+last` rate was ~27%; at 21,100 it is ~73%. Real Wikipedia is far larger, so
   these numbers are a **lower bound**, but the absolute figure is not a stable constant —
   the depth gradient is the transferable signal, not "73%".
2. **Bridge-jump is encoder-relative** (defined by a baseline encoder's TopK miss). Two
   encoders agree to within a few points, bounding this.
3. **Confirms the problem, not the solution.** High bridge prevalence means a bridge-aware
   method *could* help; whether STP's specific Bridge Score actually beats a strong
   iterative baseline at matched cost is Gate 1 (synthetic sim) and then the full benchmark.

## Reproduce

```bash
python gate0.py --dataset musique --out results_musique.json
```
Raw numbers: [`results_musique.json`](results_musique.json). Method + definitions:
[`README.md`](README.md).
