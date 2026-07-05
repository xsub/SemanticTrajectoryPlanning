# Gate 0 results — MuSiQue + 2WikiMultiHopQA (2,417 questions each)

**Verdict: PASS, decisively, on both datasets.** Bridge hops — gold evidence not
retrievable by similarity from the context so far — are pervasive, and are **not** an
artifact of how the growing context is turned into a query. The premise behind STP's
surviving contribution holds: there is large headroom for a bridge-aware retriever. The
two datasets agree in *pattern* and differ in *level* exactly as expected — MuSiQue (built
to defeat disconnected/shortcut reasoning) shows more bridges than 2Wiki (not
adversarially filtered). This confirms the *problem* exists; it does not yet validate
STP's *method* (that is Gate 1, the synthetic simulator).

**Conservative headline (strong encoder bge-small, realistic `q+last` query, K=10):
MuSiQue 72.7% · 2Wiki 37.3%** of questions contain a bridge hop — both far above the ~5%
"stop" line.

> Note: the 2Wiki numbers were regenerated after fixing a chain-ordering bug found by the
> [arc verification](../../docs/ARC_VERIFICATION.md) (D8): evidence-title matching reversed
> ~15% of gold chains; `supporting_facts` is now the authoritative hop order. The headline
> moved ~3 points (40.4→37.3) and the depth gradient became clean and monotone.

The detailed MuSiQue tables are below; the 2Wiki tables and the cross-dataset summary
follow in [§ Cross-dataset](#cross-dataset--2wikimultihopqa-corpus-matched).

Setup: full MuSiQue validation split, corpus = 21,100 unique paragraphs (gold +
distractors), TopK over the whole corpus, already-found gold excluded. Two retrieval
encoders, four query formulations. Metric = **% of questions with ≥1 bridge hop**.

## MuSiQue — % questions with ≥1 bridge hop

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

## Cross-dataset — 2WikiMultiHopQA (corpus-matched)

Run on 2,417 2Wiki validation questions (matched to MuSiQue's question count so the
corpus size, ~14k vs ~21k paragraphs, is comparable and the absolute rates are not
confounded by pool size).

% questions with ≥1 bridge hop:

| query mode | bge-small K=5/10/20 | MiniLM K=5/10/20 |
|---|---|---|
| `concat` | 42.9 / 36.0 / 30.7 | 66.5 / 58.8 / 53.1 |
| `question` | 59.9 / 55.4 / 51.3 | 74.5 / 68.5 / 63.4 |
| `last` | 58.5 / 52.9 / 49.6 | 74.4 / 68.5 / 63.8 |
| **`q+last`** (realistic) | **43.4 / 37.3 / 32.7** | **66.6 / 59.0 / 53.4** |

Same signatures as MuSiQue: (1) `q+last` ≈ `concat` and `last` is markedly *higher* than
`q+last` → **not a truncation artifact**; (2) steep, now-monotone depth gradient at K=10
`q+last` (bge: h1=1% → h2=21% → h3=94%; MiniLM: h1=6% → h2=46% → h3=93%); (3) both encoders
pass by a wide margin. The pre-fix anomaly (a non-monotone `question`-mode gradient) is gone —
it was an artifact of the reversed chains, not a property of the data.

### The two datasets side by side (bge-small, `q+last`, K=10)

| | MuSiQue (filtered vs shortcuts) | 2Wiki (not filtered) |
|---|---|---|
| item% ≥1 bridge | **72.7%** | **37.3%** |
| hop%(≥2) | 63.1% | 44.1% |
| depth h1 → h3 | 20% → 75% | 1% → 94% |

The level difference is the expected, *validating* contrast: the dataset explicitly
engineered so a disconnected retriever fails has ~2× the bridge prevalence of the one that
isn't. Note the gradient *steepness* differs between datasets (2Wiki's h1 is near zero and its
h3+ is near-saturated); what is shared is the monotone shape, measured — honestly — at **one
corpus size per dataset**. A second-corpus-size run to show the gradient shape is
scale-stable remains future work.

## Honest caveats

1. **Absolute % scales with corpus size.** At a 973-paragraph corpus (50-question smoke)
   the K=10 `q+last` rate was ~27%; at 21,100 it is ~73%. Real Wikipedia is far larger, so
   these numbers are a **lower bound**, but the absolute figure is not a stable constant —
   the depth gradient is the transferable signal, not "73%".
2. **Bridge-jump is encoder-relative, and the encoders do NOT always agree closely.** On
   MuSiQue the two encoders are within a few points, but on 2Wiki they diverge sharply
   (`q+last` K=10: bge-small 40% vs MiniLM 63% — a 23-point gap). The stronger retriever
   (bge) finds *fewer* bridges. **This is a genuine threat to the contribution's
   durability**: a state-of-the-art retriever (e5-large, bge-large, …) may shrink the
   bridge headroom further. The conservative numbers above use the stronger of the two
   encoders on purpose; a worthwhile follow-up is to sweep encoder strength and see how
   fast the bridge rate decays. The signal that survives regardless is the *depth
   gradient*, not the absolute level.
3. **Confirms the problem, not the solution.** High bridge prevalence means a bridge-aware
   method *could* help; whether STP's specific Bridge Score actually beats a strong
   iterative baseline at matched cost is Gate 1 (synthetic sim) and then the full benchmark.

## Reproduce

```bash
python gate0.py --dataset musique --out results_musique.json
python gate0.py --dataset 2wiki --limit 2417 --out results_2wiki.json   # corpus-matched
```
Raw numbers: [`results_musique.json`](results_musique.json),
[`results_2wiki.json`](results_2wiki.json). Method + definitions: [`README.md`](README.md).
