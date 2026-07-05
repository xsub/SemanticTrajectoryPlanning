#!/usr/bin/env python3
"""
Gate 1 — Does a Bridge Score solve the bridge problem, or does information gain already?

Gate 0 proved the PROBLEM exists (bridge hops are common in real multi-hop data).
Gate 1 tests whether STP's proposed MECHANISM actually solves it, under ORACLE conditions
on a synthetic graph where we control the one variable that decides the whole thing.

THE DECISIVE QUESTION
---------------------
A bridge / stepping-stone node must be retrieved to unlock the next cluster, but it may
carry little information about the final answer. Two hypotheses:
  * If ORACLE information gain (even myopic) already values bridges, a Bridge Score adds
    nothing -> the program dies.
  * If bridge value is a distinct *instrumental / reachability* quantity that myopic EIG
    misses, a Bridge Score (or lookahead) is needed -> the mechanism is worth benchmarking.

One knob makes it falsifiable: `bridge_info` in [0,1] = how much answer-information a
stepping-stone node carries. We sweep it and watch when EIG+BS separates from plain EIG —
and whether EIG+BS is merely a cheap surrogate for lookahead EIG.

MODEL (explicit geometry, pure NumPy)
-------------------------------------
C clusters are Gaussian blobs whose centres sit on a line, spacing `sep` >> blob width
`sigma`, so clusters are well separated. Between consecutive clusters sits one *bridge node*
at the midpoint. Each cluster has one *gold* node (evidence to collect); the answer is the
last cluster's gold. "Embedding similarity" = negative Euclidean distance.

Retrieval is LOCAL: the candidate set each step is the kNN frontier of the visited set.
`knn` is tuned so that from inside a cluster the ONLY reachable out-of-cluster node is its
bridge (midpoint, ~sep/2 away), while the next cluster's gold (~sep away) is NOT reachable
until the bridge is crossed. That is what makes the bridge instrumental: you must retrieve a
low-similarity, possibly low-information node to unlock the next cluster.

Oracle signals:
  EIG(v)  = answer-information of v : gold -> 1.0, bridge -> bridge_info, else 0.
  BS(v|V) = reachable-gold gain     : # gold nodes that ENTER the frontier when v is added
            (the POSITIONING §5 target y_t(v)) — the instrumental reachability value.

Policies pick from the frontier: random, query (sim to seed), cosine (sim to visited),
curiosity (novelty), eig, eig+lookahead (EIG + best reachable EIG next), bs, eig+bs.

Metric: success = collected ALL gold (reached the answer across every bridge), and recall @
budget, averaged over random instances.

READING IT
----------
At each bridge_info, compare eig vs eig+bs vs eig+lookahead:
  eig ~= eig+bs everywhere              -> Bridge Score inert with oracle access. PROGRAM DIES.
  eig+bs >> eig where bridge_info is low -> mechanism matters exactly where stepping stones
     are uninformative (the STP-relevant regime). WORTH BENCHMARKING.
  eig+bs ~= eig+lookahead              -> BS is a cheap 1-step surrogate for lookahead EIG,
     not a new signal (honest, narrower claim, still useful).
"""
import argparse, json, sys
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)

# --------------------------------------------------------------------- instance
def build_instance(C, n, sep, sigma, d, seed):
    rng = np.random.default_rng(seed)
    centers = np.zeros((C, d)); centers[:, 0] = np.arange(C) * sep
    pts, clusters = [], []
    for c in range(C):
        P = centers[c] + sigma * rng.standard_normal((n, d))
        clusters.append(list(range(len(pts), len(pts) + n)))
        pts.extend(P)
    gold = [clusters[c][int(rng.integers(0, n))] for c in range(C)]
    bridges = []
    for c in range(C - 1):
        b = (centers[c] + centers[c + 1]) / 2 + 0.05 * sigma * rng.standard_normal(d)
        bridges.append(len(pts)); pts.append(b)
    X = np.asarray(pts)
    return X, gold, bridges, clusters, gold[-1], len(pts)

def sim_matrix(X):
    # similarity = negative Euclidean distance (blob geometry)
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.maximum(D2, 0, out=D2)
    return -np.sqrt(D2)

def knn_sets(S, k):
    A = S.copy(); np.fill_diagonal(A, -np.inf)
    idx = np.argpartition(-A, k, axis=1)[:, :k]
    return [set(row.tolist()) for row in idx]

def frontier(V, knn):
    f = set()
    for u in V:
        f |= knn[u]
    return f - V

# --------------------------------------------------------------------- policies
POLICIES = ["random", "query", "cosine", "curiosity", "eig", "bs", "eig+lookahead", "eig+bs"]

def _new_unlocked(v, V, knn, F):
    return knn[v] - V - F - {v}

def run_policy(policy, S, knn, goldset, answer, seed0, info, budget, beta, rng):
    V = {seed0}
    for _ in range(budget):
        F = frontier(V, knn)
        if not F:
            break
        Fl = list(F)
        reach_gold = goldset & F
        sc = np.empty(len(Fl))
        for i, v in enumerate(Fl):
            if policy == "random":     sc[i] = rng.random()
            elif policy == "query":    sc[i] = S[v, seed0]
            elif policy == "cosine":   sc[i] = max(S[v, u] for u in V)
            elif policy == "curiosity":sc[i] = -max(S[v, u] for u in V)
            elif policy == "eig":      sc[i] = info[v]
            elif policy == "bs":       sc[i] = len((goldset & _new_unlocked(v, V, knn, F)) - reach_gold)
            elif policy == "eig+bs":   sc[i] = info[v] + beta * len((goldset & _new_unlocked(v, V, knn, F)) - reach_gold)
            elif policy == "eig+lookahead":
                nu = _new_unlocked(v, V, knn, F)
                sc[i] = info[v] + (0.9 * max((info[w] for w in nu), default=0.0))
            else: raise ValueError(policy)
        sc += 1e-6 * rng.random(len(Fl))     # fair tie-breaking
        V.add(Fl[int(np.argmax(sc))])
    return len(V & goldset) / len(goldset), float(answer in V)

# --------------------------------------------------------------------- sweep
def sweep(C, n, sep, sigma, d, knn, budget, beta, bridge_infos, trials, base_seed):
    out = {}
    for bi in bridge_infos:
        agg = {p: [0.0, 0.0] for p in POLICIES}
        for t in range(trials):
            X, gold, bridges, clusters, answer, N = build_instance(C, n, sep, sigma, d, base_seed + t)
            S = sim_matrix(X)
            knn_s = knn_sets(S, knn)
            info = np.zeros(N)
            for g in gold: info[g] = 1.0
            for b in bridges: info[b] = bi
            seed0 = clusters[0][0]
            rng = np.random.default_rng(9000 + base_seed + t)
            for p in POLICIES:
                r, s = run_policy(p, S, knn_s, set(gold), answer, seed0, info, budget, beta, rng)
                agg[p][0] += r; agg[p][1] += s
        out[bi] = {p: {"recall": round(agg[p][0]/trials, 3), "success": round(agg[p][1]/trials, 3)}
                   for p in POLICIES}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=6, help="chain length (clusters)")
    ap.add_argument("--n", type=int, default=30, help="nodes per cluster")
    ap.add_argument("--sep", type=float, default=6.0, help="cluster spacing")
    ap.add_argument("--sigma", type=float, default=0.3, help="blob width")
    ap.add_argument("--d", type=int, default=8, help="embedding dim")
    ap.add_argument("--knn", type=int, default=0, help="frontier width (0 => n+3)")
    ap.add_argument("--budget", type=int, default=0, help="0 => 3*C")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--bridge_infos", default="0.0,0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_gate1.json")
    a = ap.parse_args()
    knn = a.knn or (a.n + 3)
    budget = a.budget or 3 * a.C
    bis = [float(x) for x in a.bridge_infos.split(",")]
    log(f"[gate1] C={a.C} n={a.n} sep={a.sep} knn={knn} budget={budget} beta={a.beta} trials={a.trials}")

    res = sweep(a.C, a.n, a.sep, a.sigma, a.d, knn, budget, a.beta, bis, a.trials, a.seed)
    json.dump({"config": vars(a), "knn": knn, "budget": budget, "results": res},
              open(a.out, "w"), indent=2)

    def table(field, title):
        print(f"\n  {title}")
        hdr = "  bridge_info | " + " | ".join(f"{p:>13}" for p in POLICIES)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for bi in bis:
            print(f"  {bi:>10} | " + " | ".join(f"{res[bi][p][field]*100:12.0f}%" for p in POLICIES))

    print("\n" + "=" * 92)
    print(f"GATE 1 — synthetic bridge sim   C={a.C} clusters  budget={budget}  trials={a.trials}")
    print("=" * 92)
    table("success", "SUCCESS rate (collected ALL gold = crossed every bridge to the answer)")
    table("recall", "recall @ budget")

    lo = bis[0]
    eig, bs = res[lo]["eig"]["success"], res[lo]["eig+bs"]["success"]
    la, cos = res[lo]["eig+lookahead"]["success"], res[lo]["cosine"]["success"]
    print("\n" + "-" * 92)
    print(f"VERDICT @ bridge_info={lo} (pure stepping stones — the STP-relevant regime):")
    print(f"  cosine={cos*100:.0f}%  eig={eig*100:.0f}%  eig+bs={bs*100:.0f}%  eig+lookahead={la*100:.0f}%")
    if bs - eig < 0.05:
        print("  => eig+bs ~= eig : Bridge Score INERT even with oracle access. PROGRAM DIES.")
    else:
        print(f"  => eig+bs beats eig by {(bs-eig)*100:.0f} pts : the bridge mechanism MATTERS here.")
        print("     " + ("eig+bs ~= eig+lookahead : BS is a cheap surrogate for 1-step lookahead."
                         if abs(bs-la) < 0.05 else
                         "eig+bs != eig+lookahead : BS carries value beyond 1-step lookahead."))
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
