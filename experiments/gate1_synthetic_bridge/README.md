# Gate 1 — Does a Bridge Score solve the problem, or does information gain already?

The oracle simulator that tests whether STP's *mechanism* works, after Gate 0 confirmed the
*problem* exists.

> **Result (corrected after [arc verification](../../docs/ARC_VERIFICATION.md), D4/D5): the
> mechanism survives, but both its narratives were fixed.** At matched budget, myopic info-gain
> scores 0% while EIG+BS scores 78% — but (1) the no-bridge control shows the separation comes
> from **frontier-stepping**, not bridge-crossing (eig+bs still hits 69% with all bridges
> removed), and (2) the myopic 0% is **budget starvation, not impossibility** (same policy
> reaches 74% at ~5.5× budget). BS remains **statistically indistinguishable from 1-step
> lookahead EIG** (78% vs 79%), so the honest claim is unchanged: a cheap reachability
> surrogate for shallow planning under tight budgets — not a novel signal. Full numbers +
> both controls: [`RESULTS.md`](RESULTS.md).

## Why this experiment exists

[Gate 0](../gate0_bridge_jump/RESULTS.md) proved bridge hops are common in real multi-hop data
— the *problem* is real. Gate 1 asks the next question: does STP's proposed *mechanism* (a
Bridge Score) actually solve it, and is it distinct from what information gain already gives?

The [verification](../../docs/VERIFICATION.md) reduced STP's contribution to a bet that bridge
value is a *distinct instrumental / reachability* quantity that similarity and information gain
miss. This simulator tests that bet under oracle conditions, on a graph engineered so the one
deciding variable is explicit and swept.

## The decisive knob

`bridge_info ∈ [0,1]` = how much answer-information a stepping-stone (bridge) node carries.

- `bridge_info = 0` — a pure stepping stone (looks like a distractor to any info/similarity
  scorer). If the mechanism only helps here, it is a knife-edge case.
- `bridge_info = 1` — the bridge is as informative as gold; myopic info gain should cross on
  its own, and a Bridge Score should be redundant.

Sweeping it shows *when* a Bridge Score matters and *whether* it does anything a shallow
lookahead does not.

## Model (pure NumPy)

- **Graph:** C Gaussian-blob clusters whose centres sit on a line (spacing ≫ blob width, so
  clusters separate). One **bridge node** at each gap's midpoint. One **gold** node per
  cluster; the answer is the last cluster's gold. Similarity = negative Euclidean distance.
- **Local retrieval:** each step you may retrieve from the kNN **frontier** of the visited set.
  `knn` is tuned so a cluster's own bridge is barely reachable, but the *next* cluster's gold is
  not — until the bridge is crossed. This is what makes the bridge instrumental.
- **Oracle signals:** `EIG(v)` = answer-info of v (gold 1.0, bridge `bridge_info`, else 0);
  `BS(v|V)` = # gold nodes that enter the frontier when v is retrieved (the POSITIONING §5
  reachable-gold-gain target).
- **Policies:** `random`, `query` (sim to seed), `cosine` (sim to visited), `curiosity`
  (novelty), `eig` (myopic), `eig+lookahead` (EIG + best reachable EIG next), `bs`, `eig+bs`.

## How to read it

At each `bridge_info`, compare `eig` vs `eig+bs` vs `eig+lookahead`:

| pattern | meaning |
|---|---|
| `eig ≈ eig+bs` everywhere | Bridge Score inert with oracle access → **program dies** |
| `eig+bs ≫ eig` at low `bridge_info` | mechanism matters where stepping stones are uninformative → **worth benchmarking** |
| `eig+bs ≈ eig+lookahead` | BS is a cheap 1-step surrogate for lookahead, not a new signal (honest, narrower) |

The observed outcome is the second **and** third rows: the mechanism matters, but it is
lookahead-equivalent.

## Run it

```bash
# reuses the Gate 0 venv (numpy only)
../gate0_bridge_jump/.venv/bin/python gate1.py --trials 200 --out results_gate1.json

# chain-length robustness
for C in 4 6 8 10; do ../gate0_bridge_jump/.venv/bin/python gate1.py --C $C --bridge_infos 0.0 --trials 150; done
```

## Limitations

Oracle throughout (roles and reachability known), clean separated-blob geometry, one bridge per
gap, tuned knn/budget. This is an **existence proof of the mechanism**, not evidence it is
learnable or that it survives a real corpus — that is the full benchmark's job (with
lookahead-EIG as a mandatory baseline). See [`RESULTS.md`](RESULTS.md) § caveats.
