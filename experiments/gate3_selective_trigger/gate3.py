#!/usr/bin/env python3
"""
Gate 3 — Can we DETECT bridge states and trigger the Bridge Score selectively?

Gate 2 showed a learned Bridge Score recovers 23% of bridges dense misses, but a UNIFORM
re-rank is net-negative (it demotes easy hops). The fix must be selective: run dense normally,
escalate to BS only when the retriever is stuck on a bridge. This gate asks the two questions
that decide whether that path exists:

  Q1  Is a bridge STATE detectable at all — cheaply, from the query/context, BEFORE paying for
      BS — given that at a bridge the dense retriever is *confidently wrong* (so top-1 confidence
      alone may not fire)?
  Q2  Does gating BS on that detector beat plain dense retrieval NET?

METHOD
------
Reuses Gate 2's per-candidate Bridge Score head. Adds a per-STATE detector:
  detector features (cheap, dense-only, no BS call):
    hop depth, #retrieved, dense top-1 sim, top-5 mean, top1-top5 gap, top1-top10 gap,
    entropy of the top-20 dense-similarity distribution, retrieved-set tightness,
    state-vs-last-retrieved sim.
  label = is_bridge (next gold not in dense top-Kbridge).
Trained on the train split, AUC reported on validation (held out).

Selective policy at threshold tau: for each hop-state, if P(bridge) >= tau escalate to the BS
re-ranker, else keep dense. Sweep tau and read the net recovery frontier vs the dense baseline.

READING IT
  detector AUC ~ 0.5              -> bridge states are NOT detectable. Selective path BLOCKED.
  best selective net > dense net  -> selective triggering WINS: the deployment path exists.
  best selective net <= dense     -> detector too weak to pay off; needs better features/signal.
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gate2_learned_rerank"))
import gate2  # noqa: E402  (reuse loaders, encoder wrapper, features, BS-head training)

def log(*a): print(*a, file=sys.stderr, flush=True)

SFEATS = ["depth", "n_ret", "top1", "top5mean", "gap1_5", "gap1_10", "entropy20",
          "ret_tight", "state_last"]

def _entropy(x):
    x = x - x.max(); p = np.exp(x); p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def state_feature(sim, depth, prev_idx, E, state_emb):
    top = np.partition(sim, -20)[-20:]
    top = np.sort(top)[::-1]
    if len(prev_idx) >= 2:
        R = E[np.asarray(prev_idx)]
        P = R @ R.T
        tight = float((P.sum() - np.trace(P)) / (len(prev_idx) * (len(prev_idx) - 1)))
    else:
        tight = 0.0
    last = float(state_emb @ E[prev_idx[-1]]) if prev_idx else 0.0
    return [float(depth), float(len(prev_idx)), float(top[0]), float(top[:5].mean()),
            float(top[0] - top[4]), float(top[0] - top[9]), _entropy(top),
            tight, last]

def collect(model, items, corpus, Kbridge, K, bs_head=None):
    """Per hop-state: state features, is_bridge, dense-recovered@K, bs-recovered@K (if head)."""
    key2idx, texts = gate2.build_corpus(items, corpus)
    E = gate2.encode(model, texts); N = len(texts)
    states = list(gate2.hop_states(items, key2idx))
    S = gate2.encode(model, [s["state_text"] for s in states])
    Q = gate2.encode(model, [s["q"] for s in states])
    allidx = np.arange(N)
    SF, isb, dense_rec, bs_rec = [], [], [], []
    for i, s in enumerate(states):
        prev = s["prev"]; prevset = set(prev)
        sim_all = E @ S[i]
        gsim = sim_all[s["gold"]]
        drank = int((sim_all > gsim).sum()) - sum(1 for j in prev if sim_all[j] > gsim)
        SF.append(state_feature(sim_all, len(prev) + 1, prev, E, S[i]))
        is_bridge = drank >= Kbridge
        isb.append(is_bridge); dense_rec.append(drank < K)
        if bs_head is not None:
            mask = np.array([j not in prevset for j in allidx])
            cand = allidx[mask]
            F = gate2.features(E, cand, S[i], Q[i], prev)
            p = bs_head.predict_proba(F)[:, 1]
            gpos = int(np.where(cand == s["gold"])[0][0])
            bs_rec.append(int((p > p[gpos]).sum()) < K)
        else:
            bs_rec.append(False)
    return (np.array(SF, dtype="float32"), np.array(isb), np.array(dense_rec), np.array(bs_rec))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--train_limit", type=int, default=4000)
    ap.add_argument("--val_limit", type=int, default=2417)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--Kbridge", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_gate3.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    model = SentenceTransformer(a.model, device="cpu")

    log(f"[gate3] loading data; training BS head (reuse Gate 2) ...")
    train_items, train_corpus = gate2.LOADERS[a.dataset]("train", a.train_limit)
    val_items, val_corpus = gate2.LOADERS[a.dataset]("validation", a.val_limit)
    log(f"[gate3] FULL corpora: train={len(train_corpus)} val={len(val_corpus)} paragraphs")
    X, y_bs, _ = gate2.make_training(model, train_items, train_corpus, 40, 40, rng)
    bs_head = HistGradientBoostingClassifier(max_iter=300, random_state=a.seed).fit(X, y_bs)

    log("[gate3] building state-detector training data ...")
    SFtr, isbtr, _, _ = collect(model, train_items, train_corpus, a.Kbridge, a.K, bs_head=None)
    det = HistGradientBoostingClassifier(max_iter=300, random_state=a.seed).fit(SFtr, isbtr)

    log("[gate3] evaluating on validation (dense + bs + detector) ...")
    SFv, isbv, drec, brec = collect(model, val_items, val_corpus, a.Kbridge, a.K, bs_head=bs_head)
    P = det.predict_proba(SFv)[:, 1]
    auc = roc_auc_score(isbv, P)
    n = len(isbv); nb = int(isbv.sum())
    dense_net = float(drec.mean())
    uniform_bs_net = float(brec.mean())

    # selective: escalate to BS where P>=tau, else dense
    taus = [round(t, 3) for t in np.linspace(0.05, 0.95, 19)]
    sweep = []
    for tau in taus:
        fire = P >= tau
        sel = np.where(fire, brec, drec)
        # detector quality at this operating point
        tp = int((fire & isbv).sum()); fp = int((fire & ~isbv).sum())
        fired = int(fire.sum())
        prec = tp / fired if fired else 0.0
        rec = tp / nb if nb else 0.0
        # D2 fix (ARC_VERIFICATION): the break-even must use the EMPIRICAL loss/gain on the
        # fired subset, not the population averages — the detector fires on a selection-biased,
        # bridge-like subset where the non-bridge loss is much larger than the population 0.22.
        bfire = fire & isbv; nfire = fire & ~isbv
        R_fired = float(brec[bfire].mean() - drec[bfire].mean()) if bfire.any() else None
        L_fired = float(drec[nfire].mean() - brec[nfire].mean()) if nfire.any() else None
        sweep.append({"tau": tau, "fire_rate": round(fired / n, 3),
                      "precision": round(prec, 3), "recall": round(rec, 3),
                      "R_fired": None if R_fired is None else round(R_fired, 3),
                      "L_fired": None if L_fired is None else round(L_fired, 3),
                      "net": round(float(sel.mean()), 4)})
    best = max(sweep, key=lambda r: r["net"])

    out = {"config": vars(a), "n_states": n, "n_bridge": nb, "auc": round(float(auc), 3),
           "dense_net": round(dense_net, 4), "uniform_bs_net": round(uniform_bs_net, 4),
           "best_selective": best, "sweep": sweep,
           "feat_importance": None}
    json.dump(out, open(a.out, "w"), indent=2)

    print("\n" + "=" * 76)
    print(f"GATE 3 — selective triggering, {a.dataset} val   encoder={a.model.split('/')[-1]}")
    print(f"  {n} hop-states | bridges={nb} ({nb/n*100:.0f}%) | detector ROC-AUC = {auc:.3f}")
    print("=" * 76)
    print(f"\n  Q1 detectability: bridge-state detector AUC = {auc:.3f}  "
          f"({'detectable' if auc > 0.65 else 'weak' if auc > 0.55 else 'NOT detectable'})")
    print(f"\n  baselines (net recovery@K={a.K}):  dense={dense_net*100:.1f}%   "
          f"uniform-BS={uniform_bs_net*100:.1f}%")
    print(f"\n  selective sweep (escalate to BS where P(bridge) >= tau):")
    print(f"    {'tau':>5} | {'fire%':>6} | {'prec':>5} | {'recall':>6} | {'R_fired':>7} |"
          f" {'L_fired':>7} | {'NET%':>6}")
    print("    " + "-" * 62)
    for r in sweep[::2]:
        star = "  <- best" if r is best else ""
        rf = "  n/a" if r["R_fired"] is None else f"{r['R_fired']:5.2f}"
        lf = "  n/a" if r["L_fired"] is None else f"{r['L_fired']:5.2f}"
        print(f"    {r['tau']:>5} | {r['fire_rate']*100:5.0f}% | {r['precision']:5.2f} |"
              f" {r['recall']*100:5.0f}% | {rf:>7} | {lf:>7} | {r['net']*100:5.1f}%{star}")
    # empirical break-even from the fired-subset quantities (D2 fix)
    ops = [r for r in sweep if r["R_fired"] is not None and r["L_fired"] is not None
           and r["fire_rate"] >= 0.05]
    if ops:
        Rbar = float(np.mean([r["R_fired"] for r in ops]))
        Lbar = float(np.mean([r["L_fired"] for r in ops]))
        if Rbar + Lbar > 0:
            pstar = Lbar / (Rbar + Lbar)
            print(f"\n  empirical break-even (fired-subset averages over tau>=5% fire):"
                  f" R={Rbar:.2f} L={Lbar:.2f} -> precision needed {pstar:.2f}")
    print("\n" + "-" * 76)
    print(f"VERDICT @K={a.K}:")
    print(f"  best selective net = {best['net']*100:.1f}%  (tau={best['tau']},"
          f" fires on {best['fire_rate']*100:.0f}% of states, precision {best['precision']:.2f})")
    if best["net"] > dense_net + 0.003:
        print(f"  => SELECTIVE TRIGGERING WINS: {best['net']*100:.1f}% > dense {dense_net*100:.1f}%."
              f" The net-positive deployment path EXISTS.")
    elif auc < 0.6:
        print(f"  => bridge states barely detectable (AUC {auc:.2f}); selective path BLOCKED"
              f" with these cheap features.")
    else:
        print(f"  => detector works (AUC {auc:.2f}) but selective net {best['net']*100:.1f}% does not"
              f" beat dense {dense_net*100:.1f}%: BS recovery too weak to pay even when targeted.")
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
