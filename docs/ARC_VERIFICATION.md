# VERIFICATION REPORT — Semantic Trajectory Planning (STP) Gate 0→4 Arc

## 1. BOTTOM LINE

The arc's **directional conclusions all hold** — every gate's qualitative headline survives adversarial re-derivation from the committed code and JSON artifacts. Myopic retrieval underperforms a reachability-aware signal (Gate 1); a learned Bridge Score recovers real, non-chance bridge signal (Gate 2); naive selective triggering is net-negative and "never fire" wins under the current re-ranker (Gate 3); and adding embedding-kNN graph features degrades recovery (Gate 4). **However, the arc is undermined at the level of specific quantitative claims and one causal-mechanism story, not at the level of go/no-go verdicts.** The single most serious problem is real and touches Gates 2–4 simultaneously: all three gates silently evaluate on a **gold-only corpus (~2,629 val paragraphs), not the "~21k dev split" the docs repeatedly claim**, and the stated failure mechanism ("dense confidently retrieves an in-cluster distractor") **cannot occur because there are no distractors in the corpus that was actually run**. Adversarial verifiers split on whether this is a "real defect" or an "overclaim" — the honest reading is that gate-internal *relative* comparisons remain apples-to-apples, but every *absolute* claim (corpus size, bridge rate, chance floor, the distractor mechanism) is measured on a different, easier experiment than the one described. Two forward-looking quantitative deliverables are also wrong: Gate 3's "**>28% bridge-recovery target**" understates the true bar by ~16 points, and Gate 4's "scale-confound rebuttal" cites numbers with no committed artifact and no code path that could produce them. Trust the arrows; do not trust the absolute numbers or the mechanism narratives as written.

## 2. CONFIRMED DEFECTS

Only findings that an adversarial pass rated CONFIRMED_REAL_DEFECT or CONFIRMED_OVERCLAIM are listed. Where verifiers disagreed on the same finding, I report the **stronger (real-defect) verdict** and note the split.

### D1 — Gold-only evaluation corpus in Gates 2/3/4 (verdict split: CONFIRMED_REAL_DEFECT / CONFIRMED_OVERCLAIM)
- **Gate:** 2 (root), inherited by 3 and 4.
- **What's wrong:** `build_corpus` (gate2.py:69-75) rebuilds a corpus iterating only `it["chain"]` (gold paragraphs); the loader's full gold+distractor corpus is discarded via `train_items, _ = LOADERS[...]` (gate2.py:173-174; mirrored gate3.py:109-110, gate4.py:184-185). Verified live: the actual val corpus is **N=2,629 (gold-only)**, not the **21,100** the docs cite — exactly 8.0× smaller. Train side: 3,659 gold-only vs 33,160 full.
- **Effect on conclusion:** (a) "corpus = dev split (~21k)" (RESULTS.md:78, README.md:63) is false by ~8×. (b) The random-rerank chance floor "≈ K/N ≈ 0.05%" (RESULTS.md:30) is wrong; true value is 10/2,629 = **0.38%** (the "23% is real signal" conclusion still holds — 23% is ~60× the corrected floor). (c) The deployment mechanism "dense confidently retrieves an in-cluster distractor" (RESULTS.md:53, gate3 RESULTS.md:22) **is structurally impossible** — no distractors exist in the pool; the reranker's real task is "outrank other questions' golds." (d) Measured bridge rate collapses from gate0's realistic 48.4% (concat, K=10, 21k corpus) to 29.1% (1,864/6,404) on gold-only — so gate2's "same framing as Gate 0" (RESULTS.md:14) is a materially easier, non-comparable definition. Gate-internal *relative* results (bs>0, uniform rerank net-negative, graph hurts) survive because they run on the same gold-only pool throughout.
- **Severity:** High.
- **Exact fix:** Capture the loader corpus (`train_items, train_corpus = LOADERS[...]`); rewrite `build_corpus` to ingest the full gold+distractor dict exactly as gate0.py:168-171 does (`for key,txt in corpus.items(): key2idx[key]=len(texts); texts.append(txt)`); thread `corpus` through `make_training`/`evaluate`/`collect`/`make_rows` in all three gates so E encodes ~21k/~33k texts; re-run `python gate2.py --train_limit 4000 --val_limit 2417` and regenerate all downstream numbers. If a gold-only setting is intentionally kept, then *mandatorily* strike "corpus = dev split (~21k)", correct the chance floor to ~0.38%, rewrite RESULTS.md:53 to not invoke distractors, add `corpus_size` to the JSON, and add a caveat that gold-only ranking is easier than the realistic distractor sea.

### D2 — Gate 3 break-even uses population-average loss on a selection-biased subset (verdict split: CONFIRMED_REAL_DEFECT / CONFIRMED_OVERCLAIM)
- **Gate:** 3.
- **What's wrong:** The break-even (RESULTS.md:47-48,53) plugs the **population-average** non-bridge loss **L=0.22** (gate2 non-bridge 100%→78%) into a formula that only governs states the detector **fires** on — a strictly harder, bridge-like subset. Re-derived from gate3's own `results_gate3.json` (valid because K==Kbridge==10 makes dense-recovery exactly complementary): the implied **fired-subset loss is ~0.35**, stable across tau=0.05–0.40. Predictive check: L=0.22 over-predicts actual net by up to +5.1 pts (tau=0.05: predicted 0.6829 vs actual 0.6319); L=0.35 reproduces the sweep to <0.3 pt. R=0.226 and L=0.22 cannot both reproduce the sweep (they predict delta_sum=−172 vs actual −493).
- **Effect on conclusion:** The directional verdict ("never-fire wins; selective does not beat dense") **holds and is in fact stronger** (true loss is worse). But the two forward-looking deliverables are wrong: break-even precision is **~0.61, not 0.489**, and the Gate 3b re-ranker target is **~44% bridge recovery, not >28%**. The current re-ranker recovers 23%, so the real gap is +21 pts (23→44), nearly double the advertised +5 pts (23→28). This materially misdirects the next experiment and makes the deployment path look far more achievable than it is.
- **Severity:** Medium.
- **Exact fix:** In gate3.py, emit per-tau the empirical fired-subset loss and recovery: `L_fired = mean(drec[fire & ~isbv]) - mean(brec[fire & ~isbv])`, `R_fired = mean(brec[fire & isbv]) - mean(drec[fire & isbv])`. In RESULTS.md restate break-even precision as 0.35/(0.226+0.35) ≈ 0.61 and the target as (1−0.44)/0.44 × 0.35 ≈ 0.445; state ~44% as a lower bound (R_fired/L_fired are collinear across the flat-precision sweep; a free joint fit gives ~0.61 target). Amend caveat #2: the unsafe estimate is L *now*, not just under a future re-ranker.

### D3 — Gate 4 "scale-invariant rerun" rebuttal is non-reproducible (CONFIRMED_OVERCLAIM)
- **Gate:** 4.
- **What's wrong:** RESULTS.md:30-37 claims a "first run" with raw PPR magnitude scored bridge 18.6% / net 47.8%, used to rule out a scale confound. There is **no JSON containing 18.6/47.8** (grep hits only the prose), and `graph_feats()` (gate4.py:88-97) **hardcodes exactly 4 scale-invariant features with no flag/branch** to include raw PPR magnitude. Git shows gate4.py was introduced already in the "raw PPR dropped" form, so no committed version could produce that run.
- **Effect on conclusion:** The headline (graph features hurt: bridge 23%→20%, net 62%→55%) rests on the second run, which **is** fully reproducible from `results_gate4.json`. Deleting the rebuttal paragraph leaves the verdict intact — so this is an unverifiable auxiliary claim, not a headline-breaker. Separately, the committed `smoke_g4.json` (MiniLM, k=15) shows emb8+graph **beating** emb8 on bridge@10 (19.2 vs 11.5) — the opposite sign — and RESULTS.md does not reconcile this flip.
- **Severity:** Medium.
- **Exact fix:** Either add an `--include_raw_ppr` flag that stacks `p[cand]` as a 5th column, run it, and commit `results_gate4_raw.json`; or delete the specific numbers and replace with the code-grounded rationale ("raw PPR magnitude excluded because it scales with corpus size; only scale-invariant features used, gate4.py:88-97"). Also reconcile or caveat the smoke-run sign flip.

### D4 — Gate 1 "provably fails" overstates a budget artifact as structural (CONFIRMED_OVERCLAIM)
- **Gate:** 1.
- **What's wrong:** "myopic information gain **provably fails** to cross bridges (0%)" (RESULTS.md:5, README.md:7-8) frames a **matched-budget starvation** effect as a fundamental limitation. At bridge_info=0 the entire frontier ties at info=0 once a cluster's gold is collected, so eig degenerates to a random walk (1e-6 tie-break, gate1.py:122) and starves at budget=18. Re-running the *identical* policy at larger budgets: 40→3%, 100→84%, 185→100%. So it does not structurally fail.
- **Effect on conclusion:** The go/no-go headline ("mechanism matters; BS ≈ 1-step lookahead, not redundant") **survives** — at matched budget the separation is real and large (eig 0% vs eig+bs 85%). Only the word "provably/fundamentally fails" is disproven.
- **Severity:** High (as a wording overclaim); does not flip numbers.
- **Exact fix:** Replace structural language with budget-relative language, e.g. "at matched budget (3C), myopic info-gain is dramatically less budget-efficient (0% vs ~78%); the same policy reaches 100% at ~10× budget — the 0% is random-walk starvation, not inability to value bridges." Optionally add a knn/budget sweep row.

### D5 — Gate 1 "instrumental bridge" mechanism is empirically false (CONFIRMED_OVERCLAIM)
- **Gate:** 1.
- **What's wrong:** The claim that "the ONLY reachable out-of-cluster node is its bridge… that is what makes the bridge instrumental" (gate1.py:30-33, README:42-43, RESULTS:12-13) is false in the actual sim. With knn=33 ≥ cluster size 30, each cluster's frontier directly kNN-reaches 3–7 **plain** nodes of the next cluster in 250/250 gaps. Only the next cluster's *gold* is non-reachable. Successful eig+bs runs pick a mean of ~1.98 of 5 bridges (some pick **0**). Control removing bridges entirely: eig+bs still succeeds ~70–75% (vs ~78% with bridges) — the eig-vs-eig+bs separation the whole gate rests on **does not come from the bridges**.
- **Effect on conclusion:** The numeric headline table reproduces exactly (200 trials) and the "BS ≈ lookahead, myopic loses" conclusion survives the no-bridge control. What is false is the *causal story*: BS rewards leading-edge next-cluster frontier nodes, not "crossing a bridge you must cross." This is load-bearing for the synthetic design's construct validity, so it materially misleads a reader about *why* the mechanism works.
- **Severity:** High (mechanism/framing).
- **Exact fix:** Rewrite the geometry claims to state the bridge is not a required cut vertex, and rename the intuition to "you must step onto the next cluster's frontier before its gold is reachable"; report the no-bridge control (70% vs 78%). Stronger fix: lower knn / raise sep / lower sigma so the bridge is the only inter-cluster edge, then verify removing bridges drops eig+bs to ~0%.

### D6 — Gate 4 corpus/graph sizes mislabeled (CONFIRMED_OVERCLAIM)
- **Gate:** 4 (same root as D1).
- **What's wrong:** RESULTS.md:33-34 / README:43 cite "train graph 33k / val graph 21k," but the kNN graph is built on gate2's gold-only E: **3,659 / 2,629 nodes** (~8–9× too large). The 33k/21k are gate0's discarded full-corpus figures.
- **Effect on conclusion:** Overstates retrieval difficulty and the train/val distribution-shift magnitude (real ratio 1.4× not 1.6×); the scale-invariant-feature mitigation is unaffected, so the gate4 verdict survives.
- **Severity:** Medium.
- **Exact fix:** Correct to "~3.7k gold / ~2.6k gold," or (better) thread the full corpus per D1 and rebuild the graph over 21k/33k nodes.

### D7 — Gate 0 depth-gradient "corpus-size-robust / survives every query mode" is asserted, not shown (CONFIRMED_OVERCLAIM by the methodological pass)
- **Gate:** 0.
- **What's wrong:** The gradient is measured at a **single corpus size per dataset**; there is no second-corpus-size run showing the h2/h1 *shape* is preserved (the only corpus-size datapoint is the unreproducible 973-para "~27%" smoke, caveat #1). Larger corpora also raise the h1 baseline. And it is **not monotone in every mode**: 2Wiki "question" mode goes h2=43% → h3=12.7%, contradicting "survives every query mode."
- **Effect on conclusion:** The absolute-level robustness claim stands; the *gradient-stability* claim and the "every query mode" universality are unsupported/contradicted.
- **Severity:** Medium.
- **Exact fix:** Either run a second corpus size and show the h2/h1 ratio, or downgrade the claim to "measured at one corpus size; monotone in most but not all query modes (2Wiki-question is non-monotone)."

### D8 — Gate 0 2Wiki chain-ordering bug (bug, corrupts ~15% of 2Wiki order-dependent figures)
- **Gate:** 0.
- **What's wrong:** `load_2wiki` (gate0.py:124-132) builds chain order by matching evidence subject/object titles to context titles, but subject titles frequently don't exactly match ("Polish-Russian War" vs "Polish-Russian War (film)"), so **367/2,417 = 15.2% of chains are fully reversed** and ~23% have ≥1 missing subject. Since the bridge metric is order-dependent, reversed chains measure hop 2 with the wrong stepping stone.
- **Effect on conclusion:** Does **not** touch the MuSiQue primary result (authoritative `question_decomposition` order) nor the item-level MuSiQue headline. It corrupts the 2Wiki hop≥2 / by-depth numbers and directly explains the non-monotone 2Wiki "question" gradient — undercutting the cross-dataset "same shape / robust across datasets" support (overlaps D7).
- **Severity:** Medium.
- **Exact fix:** Use 2Wiki's `supporting_facts` order as the authoritative chain order (it is already the fallback); fall back to evidence scanning only when supporting_facts is absent, and fuzzy/substring-match subject titles to context titles.

### D9 — Gate 4 HopRAG "rather than embedding-kNN edges" overstates purity (CONFIRMED_OVERCLAIM, low)
- **Gate:** 4.
- **What's wrong:** Verified against arXiv 2502.12442v2 + ACL Anthology: HopRAG matches LLM-generated pseudo-queries with a **hybrid** score (½ keyword Jaccard + ½ embedding cosine). Embedding cosine is half the metric, so it is not eliminated, only relocated to pseudo-query embeddings.
- **Effect on conclusion:** Gate 4's substantive point survives; only the wording is imprecise.
- **Severity:** Low.
- **Exact fix:** Change to "LLM-generated pseudo-query edges rather than raw passage-embedding kNN edges."

## 3. PER-GATE VERDICT

- **Gate 0 — HOLDS WITH CAVEATS.** MuSiQue primary result is sound; the depth-gradient "corpus-size-robust / every query mode" claim is asserted-not-shown and contradicted in 2Wiki-question mode (D7), and a ~15% 2Wiki chain-reversal bug (D8) corrupts the cross-dataset support.
- **Gate 1 — HOLDS WITH CAVEATS.** Numeric headline and the BS≈lookahead equivalence reproduce exactly and are correctly powered; but "provably fails" is a disproven overstatement (D4) and the "instrumental bridge" causal story is empirically false (D5) — the conclusion survives, the mechanism narrative does not.
- **Gate 2 — HOLDS WITH CAVEATS (undermined on absolute claims).** bs signal is real (z=6.1, not chance) and uniform rerank is genuinely net-negative; but the gold-only corpus (D1) makes every absolute number and the distractor mechanism non-comparable to the documented setup.
- **Gate 3 — HOLDS WITH CAVEATS.** "Never-fire wins" is correct and robust; the forward-looking >28% target is wrong (~44%, D2) and the "monotone / peaks where you fire on nothing" prose contradicts its own JSON (best net 0.7091 at tau=0.9 firing on ~18 states).
- **Gate 4 — HOLDS WITH CAVEATS.** "Graph features degrade recovery" reproduces from the JSON; but the scale-confound rebuttal is non-reproducible (D3), the smoke run sign-flips unreconciled, corpus/graph sizes are mislabeled (D6), and the mechanism ("inherited geometry") is an untested narrative pinned on a null result with no random-noise-feature control.

## 4. THE TAUTOLOGY QUESTION

**Yes, "dense = 0% on bridges by construction" makes the bs-vs-dense bridge-recovery comparison partly circular — and the arc handles it *honestly but incompletely*.** The mechanism is real: `is_bridge = dense_rank ≥ Kbridge` (gate2.py:136) and `dense_recovery = dense_rank < K` (gate2.py:138) with K==Kbridge==10 are mutually exclusive and exhaustive, so dense-bridge-recovery is **forced to 0.0** and dense-overall@10 = n_nonbridge/n_total = 4,540/6,404 = 70.9% = exactly (1 − bridge_rate). The adversarial pass verified this to the decimal against `results_gate2.json`.

What this means for the comparison:
- The "**bs 23% vs dense 0%**" gap is **partly definitional inflation** — dense is guaranteed 0, so any bs>0 reads as pure gain even though the 0 encodes no scoring quality. The gap's *direction* (bs recovers real signal) is defended by the correct chance baseline (bs 23% ≫ random 0.38%, z=6.1), but its *magnitude as a "beat dense" story* is inflated by the definition.
- The "**dense 71% overall**" headline carries **no independent retrieval information at K=10** — it is a restatement of bridge prevalence, not of dense's ranking quality.

**Is it handled honestly?** Partly. The docs *do* label it "by construction" (README:26/34, RESULTS:4/29) and give the correct chance floor, so it is disclosed, not smuggled — one verifier rated the tautology itself **COSMETIC** for that reason, and correctly noted that dense.overall@5 = 61.7 ≠ 70.9 proves the column is non-tautological at K≠Kbridge, and that bs.overall@10 = 61.9 (the net-negative comparison) is a *real* full-corpus number that genuinely loses to dense. **But the arc does not do the two things that would fully neutralize the circularity:** (1) it never reports a K≠Kbridge column that would make dense.overall a genuine retrieval statistic, and (2) it compounds the circularity with the gold-only-corpus problem (D1) — so the "bridge" being recovered is a bridge *in an easier, distractor-free pool*, not the documented one. Net: the tautology is disclosed and does not by itself overturn any conclusion, but the arc leans on the "0% by construction" framing without flagging that the 71% number is information-free at K=10.

## 5. REFUTED ALLEGATIONS (checked out fine — build confidence)

- **No label leakage anywhere.** `gate2.features()` takes no gold arg (embeddings only); the gate3 detector `state_feature()` uses only the top-20 similarity distribution + depth/tightness/last-sim — `s['gold']`/`gsim` are used **only** to compute the `is_bridge` label, never fed to features, so detector AUC=0.702 is a genuine (weak) signal, not leaked.
- **No train/val contamination.** Heads/detector fit on `train_items`, evaluated on a separately-built `val_items` corpus; MuSiQue train/validation are official disjoint splits.
- **Gate 1 statistical power is sound.** eig+bs 78% vs eig+lookahead 79% at n=200 is a *paired* 156 vs 158 (z≈0.24) — "statistically indistinguishable" is fair, and reproduces across C=4/6/8/10 and beta sweeps.
- **Gate 2's real gaps are real, not noise.** bs 22.6% vs lookahead 14.8% is z=6.1 (n=1,864); net bs 61.9% vs dense 70.9% is z=10.8. Neither the "indistinguishable" nor the "23% vs 15%" claim is under-powered.
- **Cross-gate 23% consistency.** gate2 bs.bridge@10 = gate4 emb8.bridge@10 = 22.6 exactly (same head — gate4 imports gate2). State/bridge counts (6,404 / 1,864) are consistent across gates 2/3/4 and match gate0's n_hops.
- **Gate 3 break-even arithmetic is internally correct** given its (flawed) inputs: 0.22/0.45=0.489 and (1−0.44)/0.44×0.22=0.28 are exact. The error is the input L, not the algebra. Detector features carry no label leakage.
- **The go/no-go guards are correct.** gate3's `best['net'] > dense_net + 0.003` (gate3.py:167) correctly treats the +0.0002 tau=0.9 blip as noise, so the "selective does not beat dense" verdict is right despite the false "monotone" prose.
- **Teacher-forced recovery@K proxy is a fair, disclosed choice** (isolates scoring from trajectory error; caveated in gates 2/3/4). The arc never claims answer-F1.
- **HopRAG core claim is substantially accurate** (edges from LLM pseudo-queries, verified from two independent sources) and the generic-cross-encoder characterization is a fair, literature-supported *design rationale* with trained models explicitly exempted.

## 6. RESIDUAL CAVEATS THE ARC SHOULD STATE BUT DOESN'T

1. **recovery@K is necessary-not-sufficient for the README's answer-F1 hypotheses.** No RESULTS file bridges "23% bridge recovery" to downstream answer gain; a reader could conflate them.
2. **Gate 4's null result is not isolated from generic added-feature noise.** There is no random-noise-feature control (append 4 pure-noise columns to the same GBM) to distinguish "graph reachability is a bad signal" from "any 4 extra features hurt a fixed-capacity GBM." The "inherited geometry" mechanism is an untested narrative on a null result.
3. **Gate 4 first-hop PPR is seeded from dense top-5** (gate4.py:122,147), so the "independent graph signal" is literally cosine-seeded for first-hop states — undercutting "independent of cosine."
4. **Gate 1's baselines are structurally crippled, not just weak.** `query` scores similarity to the fixed seed only (gate1.py:112) and can never leave cluster0 (guaranteed ~0%); `cosine` starves on an over-dense in-cluster frontier (knn=33 ≥ cluster 30). These 0%s are budget/design artifacts, reported as clean structural facts.
5. **eig+bs gets a reachability oracle the eig baseline is denied** (gate1.py:98,116-120 vs 115), so "myopic vs EIG+BS" conflates myopia with oracle access; there is no reachability-but-not-gold-identity baseline to separate them.
6. **Single graph/encoder point** in Gate 4 (k=15, bge-small) — and the smoke run flips the sign — so "degrades" rests on one point in hyperparameter space.
7. **2Wiki sample is the first 2,417 items (no shuffle)**, so it may be order-correlated with dataset construction; treat 2Wiki absolute rates as approximate (compounded by D8).
8. **Gate 0 headline uses two mid-tier encoders at a single fixed K=10**; disclosed as a lower bound, but a SOTA retriever may shrink the headroom.

## 7. VERDICT

**Y — safe to show as a portfolio piece, conditional on the must-fix list below.** The arc's engineering, statistics, and honesty about *directional* conclusions are strong: powered comparisons, disclosed proxies, no label/split leakage, correct go/no-go guards, and mostly-accurate external-literature grounding. What lets it down is a recurring pattern of **absolute quantitative claims and mechanism narratives that outrun the committed evidence** — a reviewer will find these, so they must be fixed or softened before showing.

**Must-fix list:**
1. **Fix or relabel the gold-only corpus (D1)** — either re-run Gates 2/3/4 over the real ~21k distractor corpus, or correct "~21k dev split" → "~2.6k gold-only," fix the 0.05%→0.38% chance floor, and rewrite the "in-cluster distractor" mechanism prose (it describes something the run never contains). **Highest priority.**
2. **Correct Gate 3's forward target (D2)** — the Gate 3b bar is ~44% bridge recovery (break-even precision ~0.61), not >28%/0.489.
3. **Fix or delete Gate 4's scale-confound rebuttal (D3)** — commit the artifact/code path or remove the unverifiable 18.6/47.8 numbers; reconcile the smoke-run sign flip.
4. **Rewrite Gate 1's "provably fails" and "instrumental bridge" claims (D4, D5)** — use budget-relative language and report the no-bridge control (bridges are near-inert); either fix the wording or re-tune the sim so the bridge is a true cut vertex.
5. **Downgrade Gate 0's "corpus-size-robust / every query mode" gradient claim (D7)** and fix the 2Wiki chain-ordering bug (D8), or drop the cross-dataset robustness support.
6. **Add the missing residual caveats** from §6 (especially the recovery@K→F1 necessary-not-sufficient note and the absence of a noise-feature control in Gate 4).

Nothing here flips a go/no-go verdict, so the arc's *conclusions* are defensible as written — but points 1–3 are factual errors a competent reviewer will catch, and shipping them unfixed would undercut the arc's central strength, which is its rigor.