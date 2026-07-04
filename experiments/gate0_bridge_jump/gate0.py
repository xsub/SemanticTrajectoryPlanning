#!/usr/bin/env python3
"""
Gate 0 — Bridge-jump prevalence on real multi-hop QA data.

QUESTION THIS ANSWERS
---------------------
STP's whole surviving contribution only has room to win on "bridge" hops:
gold evidence that is NOT a nearest neighbour of the context accumulated so far,
so a similarity retriever misses it. If real multi-hop datasets contain very few
such hops, the premise is empirically dead — no LLM, GPU, or method needed to know it.

DEFINITION (faithful to POSITIONING.md §6 / README bridge-jump labeling)
------------------------------------------------------------------------
For a question with an ORDERED gold evidence chain g_1, ..., g_m over a corpus:
  context C_{<i} = original_question + gold_texts(g_1 .. g_{i-1})
  hop i is a BRIDGE-JUMP at K  iff  g_i is NOT in TopK( corpus | cos(., enc(C_{<i})) ),
  where already-found gold {g_1..g_{i-1}} are excluded from the TopK candidates.
TopK is over the WHOLE split corpus (all unique paragraphs), not just the question's
distractor pool — this is the honest "retrieval over a corpus" setting.

We report, per encoder and per K in {5,10,20}:
  - hop-level bridge-jump rate (all hops, and hops>=2 — the meaningful ones; hop 1 is
    just "is the first gold a top neighbour of the bare question")
  - item-level rate: fraction of questions with >=1 bridge hop  (this is the number
    that matters: STP only competes on these items)
  - breakdown by hop depth

READING THE RESULT
------------------
  item-level (hops>=2) < ~5%   -> phenomenon absent in this data; STP has no room. STOP.
  item-level ~15-30%           -> real phenomenon; the synthetic pilot + method are worth building.

A weak encoder inflates the rate (gold missed for encoder reasons, not real distance),
so we deliberately use real retrieval encoders and report >=2 of them for robustness.

Usage:
  gate0.py --dataset musique --models BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2
  gate0.py --dataset musique --limit 100     # quick smoke test
"""
import argparse, json, os, sys, time
from collections import defaultdict

def log(*a): print(*a, file=sys.stderr, flush=True)

# ---------------------------------------------------------------- data loaders
def _add(corpus, key, title, text):
    if key not in corpus:
        corpus[key] = f"{title}. {text}"

def load_musique(split, limit):
    """Return (items, corpus). items: {question, chain:[{text,title,key}]} with ordered
    gold chain via question_decomposition -> paragraph_support_idx. corpus: key->emb_text
    built from ALL paragraphs (gold AND distractor) of every loaded item, so TopK retrieval
    happens against a realistic distractor sea, not a pool of only-gold passages."""
    from datasets import load_dataset
    last = None
    for repo in ("dgslibisey/MuSiQue", "bdsaglam/musique"):
        try:
            ds = load_dataset(repo, split=split)
            log(f"[data] loaded {repo} split={split} n={len(ds)}")
            break
        except Exception as e:
            last = e; log(f"[data] {repo} failed: {e}")
    else:
        raise RuntimeError(f"could not load MuSiQue: {last}")

    items, corpus, dropped = [], {}, 0
    for ex in ds:
        if limit and len(items) >= limit: break
        if not ex.get("answerable", True):
            dropped += 1; continue
        paras = ex["paragraphs"]
        pos_by_idx = {p["idx"]: p for p in paras}
        chain, ok = [], True
        for hop in ex["question_decomposition"]:
            psi = hop.get("paragraph_support_idx")
            if psi is None:
                ok = False; break
            p = pos_by_idx.get(psi, paras[psi] if psi < len(paras) else None)
            if p is None:
                ok = False; break
            text = p["paragraph_text"].strip()
            key = (p["title"].strip(), text)
            chain.append({"text": text, "title": p["title"].strip(), "key": key})
        if not ok or len(chain) < 2:
            dropped += 1; continue
        # add EVERY paragraph of this item (gold + distractor) to the shared corpus
        for p in paras:
            _add(corpus, (p["title"].strip(), p["paragraph_text"].strip()),
                 p["title"].strip(), p["paragraph_text"].strip())
        items.append({"question": ex["question"].strip(), "chain": chain})
    log(f"[data] usable items={len(items)} dropped={dropped} corpus={len(corpus)}")
    return items, corpus

def load_2wiki(split, limit):
    """Best-effort 2WikiMultiHopQA loader. context=[[title,[sents]]], supporting_facts,
    evidences=[[subj,rel,obj]] give hop order via subject/object titles."""
    from datasets import load_dataset
    last = None
    for repo in ("scholarly-shadows-syndicate/2WikiMultihopQA",
                 "voidful/2WikiMultihopQA", "xanhho/2WikiMultihopQA"):
        try:
            ds = load_dataset(repo, split=split); log(f"[data] loaded {repo} n={len(ds)}"); break
        except Exception as e:
            last = e; log(f"[data] {repo} failed: {e}")
    else:
        raise RuntimeError(f"could not load 2Wiki: {last}")
    items, corpus, dropped = [], {}, 0
    for ex in ds:
        if limit and len(items) >= limit: break
        ctx = {t.strip(): " ".join(s).strip() for t, s in ex["context"]}
        # order hops by evidences (subject then object titles), fallback supporting_facts order
        order = []
        for ev in ex.get("evidences", []) or []:
            for t in (ev[0], ev[-1]):
                t = (t or "").strip()
                if t in ctx and t not in order: order.append(t)
        for sf in ex.get("supporting_facts", []) or []:
            t = sf[0].strip()
            if t in ctx and t not in order: order.append(t)
        chain = [{"text": ctx[t], "title": t, "key": (t, ctx[t])} for t in order if ctx[t]]
        if len(chain) < 2:
            dropped += 1; continue
        for t, txt in ctx.items():
            if txt: _add(corpus, (t, txt), t, txt)   # all context paras (gold + distractor)
        items.append({"question": ex["question"].strip(), "chain": chain})
    log(f"[data] usable items={len(items)} dropped={dropped} corpus={len(corpus)}")
    return items, corpus

LOADERS = {"musique": load_musique, "2wiki": load_2wiki}

# Query formulations for context C_<i. Disentangles a real "bridge" from an artifact of
# how the growing context is turned into ONE dense query (encoders truncate long inputs,
# so "concat" under-weights the most recent stepping stone). If bridges persist under
# 'last'/'q+last' too, they are real; if only 'concat' is high, it was an artifact.
QUERY_MODES = ("concat", "question", "last", "q+last")

def _mode_query(mode, question, prev_texts):
    if mode == "question" or not prev_texts:
        return question
    if mode == "concat":
        return question + " " + " ".join(prev_texts)
    if mode == "last":
        return prev_texts[-1]
    if mode == "q+last":
        return question + " " + prev_texts[-1]
    raise ValueError(mode)

# ---------------------------------------------------------------- measurement
def measure(items, corpus, model_name, Ks):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(model_name, device="cpu")

    # 1) corpus = ALL unique paragraphs (gold + distractor), passed in from the loader
    key2idx, corpus_texts = {}, []
    for key, emb_text in corpus.items():
        key2idx[key] = len(corpus_texts)
        corpus_texts.append(emb_text)
    log(f"[{model_name}] corpus paragraphs = {len(corpus_texts)}")
    E = model.encode(corpus_texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True).astype("float32")

    # 2) per-hop metadata + prev gold TEXTS (to build each query mode)
    hop_meta = []   # (item_idx, depth, gold_idx, prev_idxs, question, prev_texts)
    for qi, it in enumerate(items):
        prev_idx, prev_txt = [], []
        for depth, g in enumerate(it["chain"], start=1):
            hop_meta.append((qi, depth, key2idx[g["key"]], tuple(prev_idx),
                             it["question"], list(prev_txt)))
            prev_idx.append(key2idx[g["key"]]); prev_txt.append(g["text"])
    n_items, maxK = len(items), max(Ks)

    res = {"model": model_name, "n_items": n_items, "n_hops": len(hop_meta),
           "corpus_size": len(corpus_texts), "modes": {}}
    for mode in QUERY_MODES:
        Q = model.encode([_mode_query(mode, q, pt) for (_, _, _, _, q, pt) in hop_meta],
                         batch_size=64, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype("float32")
        hop_tot = defaultdict(int); hop_bridge = {K: defaultdict(int) for K in Ks}
        depth_tot = defaultdict(int); depth_bridge = {K: defaultdict(int) for K in Ks}
        item_has_bridge = {K: set() for K in Ks}
        for row, (qi, depth, gold_idx, prev, _, _) in enumerate(hop_meta):
            sims = E @ Q[row]
            if prev:
                sims[list(prev)] = -1e30       # exclude already-found gold from candidates
            top = np.argpartition(-sims, maxK)[:maxK]
            top = top[np.argsort(-sims[top])]
            rank_set_by_K = {K: set(top[:K].tolist()) for K in Ks}
            depth_key = min(depth, 4)
            hop_tot["all"] += 1
            if depth >= 2: hop_tot[">=2"] += 1
            depth_tot[depth_key] += 1
            for K in Ks:
                if gold_idx not in rank_set_by_K[K]:
                    hop_bridge[K]["all"] += 1
                    if depth >= 2: hop_bridge[K][">=2"] += 1
                    depth_bridge[K][depth_key] += 1
                    item_has_bridge[K].add(qi)
        res["modes"][mode] = {"K": {K: {
            "hop_rate_all": round(hop_bridge[K]["all"] / max(hop_tot["all"], 1), 4),
            "hop_rate_ge2": round(hop_bridge[K][">=2"] / max(hop_tot[">=2"], 1), 4),
            "item_rate_any_bridge": round(len(item_has_bridge[K]) / max(n_items, 1), 4),
            "by_depth": {d: round(depth_bridge[K][d] / max(depth_tot[d], 1), 4)
                          for d in sorted(depth_tot)},
        } for K in Ks}}
    res["secs"] = round(time.time() - t0, 1)
    return res

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique", choices=list(LOADERS))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--models", default="BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--K", default="5,10,20")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    Ks = [int(x) for x in args.K.split(",")]
    items, corpus = LOADERS[args.dataset](args.split, args.limit or None)
    if not items:
        log("no items — aborting"); sys.exit(2)

    all_res = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        log(f"=== encoder: {m} ===")
        all_res.append(measure(items, corpus, m, Ks))

    out = {"dataset": args.dataset, "split": args.split, "limit": args.limit,
           "n_items": len(items), "Ks": Ks, "results": all_res}
    json.dump(out, open(args.out, "w"), indent=2)

    # human summary
    Kmid = Ks[len(Ks)//2]
    print("\n" + "=" * 84)
    print(f"GATE 0 — bridge-jump prevalence   dataset={args.dataset} split={args.split}"
          f"  items={len(items)}")
    print("query modes: concat=Q+all prev golds | question=Q only | last=prev gold only"
          " | q+last=Q+prev gold")
    print("=" * 84)
    for r in all_res:
        print(f"\nEncoder: {r['model']}   (corpus={r['corpus_size']} paras, {r['secs']}s)")
        print(f"  item% with >=1 bridge hop, by query mode:")
        print(f"    {'mode':>9} | " + " | ".join(f"K={K:<3}" for K in Ks))
        print("    " + "-" * (12 + 8*len(Ks)))
        for mode in QUERY_MODES:
            cells = " | ".join(f"{r['modes'][mode]['K'][K]['item_rate_any_bridge']*100:5.1f}"
                               for K in Ks)
            print(f"    {mode:>9} | {cells}")
        print(f"  hop%(>=2) and depth breakdown at K={Kmid}:")
        for mode in QUERY_MODES:
            k = r["modes"][mode]["K"][Kmid]
            dd = "  ".join(f"h{d}{'+' if d==4 else ''}={v*100:.0f}%" for d, v in k["by_depth"].items())
            print(f"    {mode:>9}: hop>=2={k['hop_rate_ge2']*100:4.1f}%   {dd}")
    print("\nInterpretation: the honest number is the LOWEST across query modes (esp. 'last'"
          " / 'q+last').")
    print("  item%(>=1 bridge) < ~5% => phenomenon absent (STOP); ~15-30% => real, worth the pilot.")
    print("  If only 'concat' is high, the effect was a context-truncation artifact, not a bridge.")
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
