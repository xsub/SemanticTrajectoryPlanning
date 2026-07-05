#!/usr/bin/env python3
"""
Gate 4 — Does a GRAPH / PPR-augmented re-ranker clear Gate 3's 28% bar?

Gate 2: a learned Bridge Score with 8 embedding features recovers 23% of bridges.
Gate 3: selective triggering needs the re-ranker to recover >28% of bridges to pay off.
Gate 4 (this): make the re-ranker faithful to POSITIONING §5 — the Bridge Score is a
graph-reachability quantity, not a similarity. Add personalized-PageRank + graph-structural
features over the embedding kNN graph and ask whether bridge recovery clears 28%.

Why not a vanilla cross-encoder? A cross-encoder scores query-document RELEVANCE, and a bridge
is low-relevance to the accumulated context BY DEFINITION — it would miss bridges by the same
logic as dense retrieval. The principled stronger signal is graph reachability from the
explored set (the actual Bridge Score functional), which is what this adds. A *trained* GNN is
the GPU-tier extension of the same idea; here we compute the graph features explicitly (CPU).

WHAT'S ADDED (over Gate 2's 8 embedding features)
  ppr        : personalized PageRank of a candidate, seeded at the retrieved/explored set —
               "reachable from where I am via the graph", even at low direct cosine.
  ppr_rank   : normalized rank of ppr (position, robust to scale)
  graph_hop  : min hops from the explored set to the candidate on the kNN graph (capped)
  onehop     : is the candidate in the kNN of any explored node (frontier indicator)
  degree     : global kNN in-degree (hub-ness), normalized

Same protocol as Gate 2/3: teacher-forced next-hop recovery on MuSiQue, next-gold training
target, sklearn GBM head, bridge vs non-bridge vs net recovery. Two heads are trained and
compared head-to-head on the SAME states: emb8 (Gate 2 features) and emb8+graph.
"""
import argparse, json, math, os, sys
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gate2_learned_rerank"))
import gate2  # noqa: E402

def log(*a): print(*a, file=sys.stderr, flush=True)

# ---------------------------------------------------------------- graph
def build_knn_graph(E, k, chunk=2048):
    """Symmetric kNN graph over normalized embeddings. Returns CSR adjacency + row-norm transpose.
    Chunked so the full N x N similarity matrix is never materialized."""
    N = len(E)
    rows_l, cols_l = [], []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Sb = E[start:end] @ E.T                          # (b, N)
        Sb[np.arange(end - start), np.arange(start, end)] = -1.0   # drop self
        nn = np.argpartition(-Sb, k, axis=1)[:, :k]
        rows_l.append(np.repeat(np.arange(start, end), k))
        cols_l.append(nn.reshape(-1))
    rows = np.concatenate(rows_l); cols = np.concatenate(cols_l)
    A = sp.csr_matrix((np.ones(len(rows), "float32"), (rows, cols)), shape=(N, N))
    A = ((A + A.T) > 0).astype("float32")               # symmetric, unweighted
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    P = A.multiply(1.0 / deg[:, None]).tocsr()           # row-stochastic
    return A.tocsr(), P.T.tocsr(), np.asarray(A.sum(1)).ravel()

def ppr(seed_idx, PT, N, alpha=0.85, iters=20):
    s = np.zeros(N, "float32")
    if len(seed_idx):
        s[np.asarray(seed_idx)] = 1.0 / len(seed_idx)
    else:
        return s
    p = s.copy()
    for _ in range(iters):
        p = (1 - alpha) * s + alpha * (PT @ p)
    return p

def graph_hops(seed_idx, A, N, cap=4):
    """min hop distance from seed set to every node on the kNN graph, capped."""
    dist = np.full(N, cap + 1, "int16")
    if not len(seed_idx):
        return dist
    frontier = np.asarray(seed_idx)
    dist[frontier] = 0
    indptr, indices = A.indptr, A.indices
    for d in range(1, cap + 1):
        nxt = []
        for u in frontier:
            nbrs = indices[indptr[u]:indptr[u + 1]]
            new = nbrs[dist[nbrs] > d]
            dist[new] = d; nxt.append(new)
        if not nxt:
            break
        frontier = np.unique(np.concatenate(nxt))
    return dist

def graph_feats(cand, seed_idx, PT, A, deg_norm, N, cap):
    # SCALE-INVARIANT graph features only (raw PPR magnitude dropped: it depends on corpus size
    # and caused train/val distribution shift between the 33k-train and 21k-val graphs).
    p = ppr(seed_idx, PT, N)
    order = np.argsort(-p)
    rank = np.empty(N); rank[order] = np.arange(N)
    hop = graph_hops(seed_idx, A, N, cap)
    onehop = (hop == 1).astype("float32")
    return np.stack([1 - rank[cand] / N, hop[cand].astype("float32") / (cap + 1),
                     onehop[cand], deg_norm[cand]], 1)

# ---------------------------------------------------------------- data build
def make_rows(model, items, k, cap, neg_hard, neg_rand, rng, with_graph):
    key2idx, texts = gate2.build_corpus(items)
    E = gate2.encode(model, texts); N = len(texts)
    A = PT = deg_norm = None
    if with_graph:
        A, PT, deg = build_knn_graph(E, k)
        deg_norm = (deg / deg.max()).astype("float32")
    states = list(gate2.hop_states(items, key2idx))
    S = gate2.encode(model, [s["state_text"] for s in states])
    Q = gate2.encode(model, [s["q"] for s in states])
    allidx = np.arange(N)
    X8, Xg, y = [], [], []
    for i, s in enumerate(states):
        prev = s["prev"]; prevset = set(prev)
        sim = E @ S[i]
        order = np.argsort(-sim)
        hard = [j for j in order if j != s["gold"] and j not in prevset and j not in s["remaining"]][:neg_hard]
        rand = [j for j in rng.choice(allidx, neg_rand * 3, replace=False)
                if j != s["gold"] and j not in prevset and j not in s["remaining"]][:neg_rand]
        cand = np.array([s["gold"]] + list(s["remaining"] - {s["gold"]}) + hard + rand)
        X8.append(gate2.features(E, cand, S[i], Q[i], prev))
        if with_graph:
            seed = prev if prev else list(order[:5])
            Xg.append(graph_feats(cand, seed, PT, A, deg_norm, N, cap))
        lb = np.zeros(len(cand)); lb[0] = 1
        y.append(lb)
    X8 = np.vstack(X8); Y = np.concatenate(y)
    Xg = np.vstack(Xg) if with_graph else None
    return (X8, Xg, Y)

def evaluate(model, items, head8, headg, k, cap, Ks, Kbridge):
    key2idx, texts = gate2.build_corpus(items)
    E = gate2.encode(model, texts); N = len(texts)
    A, PT, deg = build_knn_graph(E, k)
    deg_norm = (deg / deg.max()).astype("float32")
    states = list(gate2.hop_states(items, key2idx))
    S = gate2.encode(model, [s["state_text"] for s in states])
    Q = gate2.encode(model, [s["q"] for s in states])
    allidx = np.arange(N)
    POL = ["dense", "emb8", "emb8+graph", "oracle"]
    cnt = {p: {b: {K: [0, 0] for K in Ks} for b in (True, False, "all")} for p in POL}
    for i, s in enumerate(states):
        prev = s["prev"]; prevset = set(prev)
        mask = np.array([j not in prevset for j in allidx]); cand = allidx[mask]
        F8 = gate2.features(E, cand, S[i], Q[i], prev)
        sim = F8[:, 0]
        order = np.argsort(-sim)
        seed = prev if prev else list(order[:5])
        Fg = np.hstack([F8, graph_feats(cand, seed, PT, A, deg_norm, N, cap)])
        gpos = int(np.where(cand == s["gold"])[0][0])
        p8 = head8.predict_proba(F8)[:, 1]
        pg = headg.predict_proba(Fg)[:, 1]
        rp = {"dense": int((sim > sim[gpos]).sum()), "emb8": int((p8 > p8[gpos]).sum()),
              "emb8+graph": int((pg > pg[gpos]).sum()), "oracle": 0}
        is_bridge = rp["dense"] >= Kbridge
        for K in Ks:
            for p in POL:
                for bucket in (is_bridge, "all"):
                    cnt[p][bucket][K][1] += 1
                    if rp[p] < K:
                        cnt[p][bucket][K][0] += 1
    return cnt, len(states), int(sum(1 for i in range(len(states)) if True))

def rate(c): return 100.0 * c[0] / c[1] if c[1] else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--train_limit", type=int, default=4000)
    ap.add_argument("--val_limit", type=int, default=2417)
    ap.add_argument("--k", type=int, default=15, help="kNN graph degree")
    ap.add_argument("--cap", type=int, default=4, help="max graph hops")
    ap.add_argument("--K", default="5,10")
    ap.add_argument("--Kbridge", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_gate4.json")
    a = ap.parse_args()
    Ks = [int(x) for x in a.K.split(",")]; rng = np.random.default_rng(a.seed)
    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = SentenceTransformer(a.model, device="cpu")

    log("[gate4] loading + building training rows (emb8 + graph) ...")
    tr, _ = gate2.LOADERS[a.dataset]("train", a.train_limit)
    va, _ = gate2.LOADERS[a.dataset]("validation", a.val_limit)
    X8, Xg, Y = make_rows(model, tr, a.k, a.cap, 40, 40, rng, with_graph=True)
    log(f"[gate4] train rows {X8.shape} (+graph {Xg.shape}); training two heads ...")
    head8 = HistGradientBoostingClassifier(max_iter=300, random_state=a.seed).fit(X8, Y)
    headg = HistGradientBoostingClassifier(max_iter=300, random_state=a.seed).fit(np.hstack([X8, Xg]), Y)

    log("[gate4] evaluating on validation ...")
    cnt, n, nb_all = evaluate(model, va, head8, headg, a.k, a.cap, Ks, a.Kbridge)
    nb = cnt["dense"][True][Ks[-1]][1]; nn = cnt["dense"][False][Ks[-1]][1]
    POL = ["dense", "emb8", "emb8+graph", "oracle"]
    R = {p: {name: {K: round(rate(cnt[p][b][K]), 1) for K in Ks}
             for name, b in (("bridge", True), ("nonbridge", False), ("overall", "all"))} for p in POL}
    json.dump({"config": vars(a), "n_states": n, "n_bridge": nb, "results": R},
              open(a.out, "w"), indent=2)

    print("\n" + "=" * 74)
    print(f"GATE 4 — graph/PPR re-ranker, {a.dataset} val   encoder={a.model.split('/')[-1]}")
    print(f"  {n} hop-states | bridges={nb} ({nb/n*100:.0f}%) | kNN graph k={a.k}")
    print("=" * 74)
    for split in ("bridge", "nonbridge", "overall"):
        lab = {"bridge": "BRIDGE recovery (dense=0% by construction) — the target",
               "nonbridge": "NON-bridge (control)", "overall": "OVERALL / net"}[split]
        print(f"\n  {lab}:")
        print(f"    {'policy':>12} | " + " | ".join(f"K={K:<3}" for K in Ks))
        print("    " + "-" * (15 + 8 * len(Ks)))
        for p in POL:
            print(f"    {p:>12} | " + " | ".join(f"{R[p][split][K]:5.1f}" for K in Ks))

    kb = Ks[-1]; b8 = R["emb8"]["bridge"][kb]; bg = R["emb8+graph"]["bridge"][kb]
    print("\n" + "-" * 74)
    print(f"VERDICT @K={kb}:  emb8 bridge recovery={b8:.0f}%  ->  emb8+graph={bg:.0f}%"
          f"   (Gate 3 bar = 28%)")
    if bg >= 28 and bg > b8 + 1:
        print(f"  => graph/PPR clears the 28% bar ({bg:.0f}%): selective triggering becomes viable.")
    elif bg > b8 + 1:
        print(f"  => graph features help (+{bg-b8:.0f} pts) but stay below 28%: not yet enough.")
    else:
        print(f"  => graph features do NOT help bridge recovery. Signal isn't simple graph reachability;"
              f" needs a trained model (GNN/cross-encoder, GPU tier).")
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
