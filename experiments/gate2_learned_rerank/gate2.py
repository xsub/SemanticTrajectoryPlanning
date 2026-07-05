#!/usr/bin/env python3
"""
Gate 2 — Can a LEARNED re-ranker recover bridge hops on real data (no LLM)?

Gate 0: bridge hops are common in real multi-hop data (the problem is real).
Gate 1: under ORACLE conditions a Bridge Score solves it, but ~= 1-step lookahead.
Gate 2 (this): drop the oracle. Train a cheap re-ranker on real data and ask whether it
recovers the bridge hops that plain dense retrieval misses BY CONSTRUCTION — and whether a
"bridge" training target differs from a broader "lookahead / on-path" target.

MEASUREMENT (teacher-forced next-hop recovery — same framing as Gate 0)
-----------------------------------------------------------------------
For each question with ordered gold chain g_1..g_m, and each hop t, the state is
  C_<t = question + gold_texts(g_1..g_{t-1})   (true prefix; teacher-forced, no error compounding)
and the target to recover is g_t, ranked against the whole corpus (already-collected gold
excluded). A hop is a BRIDGE at K iff g_t is NOT in the dense top-K of C_<t (Gate 0 definition)
-> dense retrieval recovers 0% of bridges by construction. The question is what a learned
score recovers.

POLICIES (all score every corpus candidate; no LLM anywhere)
  dense     : rank by cos(candidate, state query)            [baseline; 0% on bridges by def]
  bs        : learned head P(candidate == next gold g_t | features)      [the Bridge Score]
  lookahead : learned head P(candidate on remaining gold path g_t..g_m)  [broader reachability]
  oracle    : candidate == g_t                                            [upper bound / sanity]

FEATURES (per candidate v given state, cheap, embedding-only)
  cos(v,state), cos(v,question), cos(v,last_gold), max/mean/min cos(v, retrieved),
  novelty (1-max), and cos(v,state)-max (relevance-minus-redundancy).

Heads are gradient-boosted trees (sklearn), trained on the TRAIN split with sampled hard+random
negatives, evaluated on VALIDATION. Metric = recovery@K, reported SEPARATELY for bridge and
non-bridge hops.

READING IT
  bs recovers a meaningful fraction of bridges (dense=0%)   -> mechanism transfers to real data.
  bs ~= lookahead on bridge recovery                         -> matches Gate 1 (BS not a distinct
                                                                signal, just shallow reachability).
  bs badly hurts non-bridge recovery                         -> the re-rank is not free.
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gate0_bridge_jump"))
from gate0 import LOADERS  # noqa: E402  (reuse the Gate 0 dataset loaders)

def log(*a): print(*a, file=sys.stderr, flush=True)

FEATS = ["cos_state", "cos_q", "cos_last", "max_ret", "mean_ret", "min_ret", "novelty", "relmredund"]

def encode(model, texts):
    return model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype("float32")

def features(E, cand, state_emb, q_emb, retrieved):
    """cand: 1D idx array. Returns (len(cand), len(FEATS))."""
    Ec = E[cand]
    f_state = Ec @ state_emb
    f_q = Ec @ q_emb
    if len(retrieved):
        R = E[np.asarray(retrieved)]
        simR = Ec @ R.T
        f_max, f_mean, f_min = simR.max(1), simR.mean(1), simR.min(1)
        f_last = Ec @ E[retrieved[-1]]
    else:
        z = np.zeros(len(cand), "float32")
        f_max = f_mean = f_min = f_last = z
    return np.stack([f_state, f_q, f_last, f_max, f_mean, f_min, 1 - f_max, f_state - f_max], 1)

def build_corpus(items, corpus):
    """Index the FULL loader corpus (gold + distractor paragraphs).

    D1 fix (see docs/ARC_VERIFICATION.md): an earlier version rebuilt the corpus from the
    gold chains only (~2.6k paragraphs), silently discarding the loaders' distractors and
    making the ranking task easier than documented. The loaders guarantee every gold key
    is present in `corpus`, so indexing `corpus` directly gives the honest pool (~21k val)."""
    key2idx, texts = {}, []
    for key, emb_text in corpus.items():
        key2idx[key] = len(texts); texts.append(emb_text)
    for it in items:                      # sanity: every gold must be in the pool
        for g in it["chain"]:
            assert g["key"] in key2idx, f"gold key missing from corpus: {g['key'][0][:40]}"
    return key2idx, texts

def hop_states(items, key2idx):
    """Yield teacher-forced hop states. Each: question, gold_idx, prev_idxs, state_text."""
    for it in items:
        gidx = [key2idx[g["key"]] for g in it["chain"]]
        acc = it["question"]
        for t in range(len(it["chain"])):
            yield {"q": it["question"], "state_text": acc, "gold": gidx[t],
                   "prev": gidx[:t], "remaining": set(gidx[t:])}
            acc = acc + " " + it["chain"][t]["text"]

def make_training(model, items, corpus, neg_hard, neg_rand, rng):
    key2idx, texts = build_corpus(items, corpus)
    E = encode(model, texts)
    N = len(texts)
    states = list(hop_states(items, key2idx))
    S = encode(model, [s["state_text"] for s in states])
    Q = encode(model, [s["q"] for s in states])
    X, y_bs, y_la = [], [], []
    allidx = np.arange(N)
    for i, s in enumerate(states):
        prev = set(s["prev"])
        sim = E @ S[i]
        # hard negatives = top by state-sim that are not gold/prev; plus random
        order = np.argsort(-sim)
        hard = [j for j in order if j != s["gold"] and j not in prev and j not in s["remaining"]][:neg_hard]
        rand = rng.choice(allidx, size=neg_rand * 3, replace=False)
        rand = [j for j in rand if j != s["gold"] and j not in prev and j not in s["remaining"]][:neg_rand]
        cand = np.array([s["gold"]] + list(s["remaining"] - {s["gold"]}) + hard + rand)
        F = features(E, cand, S[i], Q[i], s["prev"])
        X.append(F)
        lb = np.zeros(len(cand)); lb[0] = 1                      # next-gold only
        la = np.array([1 if c in s["remaining"] else 0 for c in cand])  # anywhere on remaining path
        y_bs.append(lb); y_la.append(la)
    return np.vstack(X), np.concatenate(y_bs), np.concatenate(y_la)

def evaluate(model, items, corpus, heads, Ks, Kbridge):
    key2idx, texts = build_corpus(items, corpus)
    E = encode(model, texts); N = len(texts)
    states = list(hop_states(items, key2idx))
    S = encode(model, [s["state_text"] for s in states])
    Q = encode(model, [s["q"] for s in states])
    allidx = np.arange(N)
    POL = ["dense", "bs", "lookahead", "oracle", "hybrid"]
    # counters[pol][bucket][K] = [recovered, total]; bucket in {True,False,"all"}
    cnt = {p: {b: {K: [0, 0] for K in Ks} for b in (True, False, "all")} for p in POL}
    for i, s in enumerate(states):
        prev = set(s["prev"])
        mask = np.array([j not in prev for j in allidx])
        cand = allidx[mask]
        F = features(E, cand, S[i], Q[i], s["prev"])
        sim = F[:, 0]                                            # cos_state
        gpos = int(np.where(cand == s["gold"])[0][0])
        # rank position of the gold under each score (#candidates strictly better)
        pbs = heads["bs"].predict_proba(F)[:, 1]
        pla = heads["la"].predict_proba(F)[:, 1]
        rp = {"dense": int((sim > sim[gpos]).sum()),
              "bs": int((pbs > pbs[gpos]).sum()),
              "lookahead": int((pla > pla[gpos]).sum()),
              "oracle": 0}
        is_bridge = rp["dense"] >= Kbridge
        for K in Ks:
            rec = {"dense": rp["dense"] < K, "bs": rp["bs"] < K,
                   "lookahead": rp["lookahead"] < K, "oracle": True,
                   # hybrid: split the budget — dense top-ceil(K/2) OR bs top-ceil(K/2)
                   "hybrid": (rp["dense"] < math.ceil(K / 2)) or (rp["bs"] < math.ceil(K / 2))}
            for p in POL:
                for bucket in (is_bridge, "all"):
                    cnt[p][bucket][K][1] += 1
                    if rec[p]:
                        cnt[p][bucket][K][0] += 1
    return cnt, len(states)

def rate(c):  # [rec,tot] -> pct
    return 100.0 * c[0] / c[1] if c[1] else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique", choices=list(LOADERS))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--train_limit", type=int, default=2500)
    ap.add_argument("--val_limit", type=int, default=600)
    ap.add_argument("--neg_hard", type=int, default=40)
    ap.add_argument("--neg_rand", type=int, default=40)
    ap.add_argument("--K", default="5,10")
    ap.add_argument("--Kbridge", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_gate2.json")
    a = ap.parse_args()
    Ks = [int(x) for x in a.K.split(",")]
    rng = np.random.default_rng(a.seed)

    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = SentenceTransformer(a.model, device="cpu")

    log(f"[gate2] loading train ({a.train_limit}) + val ({a.val_limit}) of {a.dataset}")
    train_items, train_corpus = LOADERS[a.dataset]("train", a.train_limit)
    val_items, val_corpus = LOADERS[a.dataset]("validation", a.val_limit)
    log(f"[gate2] FULL corpora: train={len(train_corpus)} val={len(val_corpus)} paragraphs")

    log("[gate2] building training tuples ...")
    X, y_bs, y_la = make_training(model, train_items, train_corpus, a.neg_hard, a.neg_rand, rng)
    log(f"[gate2] train matrix {X.shape}  pos_bs={int(y_bs.sum())} pos_la={int(y_la.sum())}")
    heads = {}
    heads["bs"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                 random_state=a.seed).fit(X, y_bs)
    heads["la"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                 random_state=a.seed).fit(X, y_la)
    log("[gate2] evaluating on validation ...")
    cnt, n_states = evaluate(model, val_items, val_corpus, heads, Ks, a.Kbridge)

    nb = {b: cnt["dense"][b][Ks[-1]][1] for b in (True, False)}
    POL = ["dense", "bs", "lookahead", "hybrid", "oracle"]
    out = {"config": vars(a), "n_val_states": n_states,
           "corpus_train": len(train_corpus), "corpus_val": len(val_corpus),
           "n_bridge": nb[True], "n_nonbridge": nb[False], "results": {}}
    for p in POL:
        out["results"][p] = {name: {K: round(rate(cnt[p][b][K]), 1) for K in Ks}
                             for name, b in (("bridge", True), ("nonbridge", False), ("overall", "all"))}
    json.dump(out, open(a.out, "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"GATE 2 — learned re-ranker, {a.dataset} val   encoder={a.model.split('/')[-1]}")
    print(f"  {n_states} hop-states  |  bridges (dense-miss @K={a.Kbridge}): {nb[True]}"
          f"   non-bridge: {nb[False]}")
    print("=" * 78)
    titles = {"bridge": "BRIDGE hops (dense=0% by construction) — the recovery test",
              "nonbridge": "NON-bridge hops (control — does re-rank break easy hops?)",
              "overall": "OVERALL / NET (all hops weighted) — the deployment metric"}
    for split in ("bridge", "nonbridge", "overall"):
        print(f"\n  recovery@K — {titles[split]}:")
        print(f"    {'policy':>10} | " + " | ".join(f"K={K:<3}" for K in Ks))
        print("    " + "-" * (13 + 8 * len(Ks)))
        for p in POL:
            print(f"    {p:>10} | " + " | ".join(f"{out['results'][p][split][K]:5.1f}" for K in Ks))

    kb = Ks[-1]
    R = out["results"]
    print("\n" + "-" * 78)
    print(f"VERDICT @K={kb}:")
    print(f"  bridge recovery : dense={R['dense']['bridge'][kb]:.0f}%  bs={R['bs']['bridge'][kb]:.0f}%"
          f"  lookahead={R['lookahead']['bridge'][kb]:.0f}%  hybrid={R['hybrid']['bridge'][kb]:.0f}%")
    if R["bs"]["bridge"][kb] < 5:
        print("  => learned BS barely recovers bridges. Mechanism does NOT transfer to real data.")
    else:
        print(f"  => learned BS recovers {R['bs']['bridge'][kb]:.0f}% of bridges dense misses"
              f" (from 0). Mechanism transfers.")
        print("     " + ("bs ~= lookahead (matches Gate 1)."
                         if abs(R['bs']['bridge'][kb] - R['lookahead']['bridge'][kb]) < 5 else
                         "bs > lookahead : the sharp next-gold target beats the broad on-path target."))
    print(f"  net (overall)   : dense={R['dense']['overall'][kb]:.0f}%  bs={R['bs']['overall'][kb]:.0f}%"
          f"  hybrid={R['hybrid']['overall'][kb]:.0f}%")
    if R["bs"]["overall"][kb] < R["dense"]["overall"][kb]:
        print("  => UNIFORM BS re-rank is NET-NEGATIVE (bridge gain < easy-hop loss).")
        if R["hybrid"]["overall"][kb] >= R["dense"]["overall"][kb]:
            print(f"     but HYBRID (dense OR bs, split budget) is net-positive"
                  f" ({R['hybrid']['overall'][kb]:.0f}% >= {R['dense']['overall'][kb]:.0f}%):"
                  f" BS must be a SELECTIVE fallback, not a global re-rank.")
        else:
            print("     and hybrid does not recover the loss either — deployment path unclear.")
    else:
        print("  => BS re-rank is net-positive even applied uniformly.")
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
