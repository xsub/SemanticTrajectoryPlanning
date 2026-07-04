# STP Documentation — Verification Report

Scope: `POSITIONING.md`, `README.md`, `PAPER_v2_SKELETON.md` (and their shared claim set), verified against citation-fidelity, formal-check, steelman, and cross-document consistency results, with the source docs re-read directly for the cross-doc claims.

---

## 1. Bottom line

The positioning analysis **holds up in its thesis and its verdict, but not in every load-bearing detail.** The central move — conceding C1–C4/C6/C7 to years-old prior art and staking novelty only on the three-factor, state-conditioned, answer-frontier-weighted Bridge Score plus a detour-crediting benchmark — is correct, honest, and unusually rigorous: the great majority of citations are real and accurately characterized, and the self-concessions (O1–O7) are fair. However, the analysis is **too harsh on exactly one axis (C7)**, where it escalates from "unoriginal" to "feasibility refuted" on the back of a citation (Robinson et al.) that does not support an impossibility claim; it contains **one reviewer-obvious formal error** (the Freeman-betweenness reduction, which as written evaluates to 0) plus a cluster of undefined symbols in the very formula that is the paper's sole surviving contribution; and it **overclaims the detour-crediting benchmark as a delivered artifact** in README while the paper skeleton's own Limitations concedes it is unbuilt future work. None of these sink the paper, but each is the kind of defect a hostile reviewer would seize on to discredit an otherwise-strong sweep. The surviving contribution itself is real — it survives the pre-emption attack — but by a thinner margin than the docs imply.

---

## 2. Confirmed solid

The following characterizations checked out against their sources and require no change:

- **FLARE** — confidence-triggered re-retrieval / iterative uncertainty-driven retrieval: **accurate** (Sec 3.2.1).
- **IRCoT** — interleaves retrieval with each CoT step, multi-step not one-shot; a genuine killer cite for C1: **accurate** (title + §3.1).
- **Stop-RAG** — iterative RAG as a finite-horizon MDP with a value-based stopping controller; killer cite for C4: **accurate** (abstract + §4.1).
- **BALD** — EIG-as-acquisition = mutual information about parameters θ; the A-for-θ substitution STP relies on is legitimate: **accurate** (Eqn 1).
- **InfoGain-RAG** — "Document Information Gain" signal used to train a reranker; "most directly in RAG" for C2: **accurate** (abstract, Eq. 3).
- **IGP (arXiv 2601.17532)** — information-gain pruning/reranking in RAG; **confirmed to exist and to do what is claimed** (abstract, §3.1). (Note: this Jan-2026 id post-dates the stated knowledge horizon but is verified real.)
- **MINERVA** — RL walk over a KG with LSTM-encoded path history = history-conditioned MDP; killer cite for C4: **accurate**.
- **DeepPath** — RL KG path-finding with an explicit diversity reward term: **accurate** (abstract + `r_DIVERSITY` definition).
- **BeamDR** — beam-search evidence chaining in dense embedding space; the pre-emption of C7's continuous-space sequential retrieval: **accurate** (§2.2).

The MI identity in POSITIONING §5 line 61 (EIG = I(A;X_a | q,C_t)) is itself **correct** as the BALD symmetry with A substituted for θ.

---

## 3. Overreaches

### 3.1 Robinson et al. — "feasibility refuted" (SEVERITY: HIGH)

- **What the docs say:** POSITIONING §1 ("affirmatively contradicted by primary geometry results"), §3 C7 row ("Covered **and** feasibility refuted … refuted by Robinson et al."), §5 ("geometrically refuted … if the space is not a manifold, geodesic 'trajectories' through interpolated points are ill-defined and most of the volume is semantically empty"). README §4.3/line 66 and §8; PAPER_v2 §2 line 41 and §6 line 163 echo it.
- **What the source supports:** Robinson et al. test **only the discrete input-token embedding subspace** (the fixed vocabulary table), not document/passage/sentence retrieval embeddings — which is exactly the space STP navigates. The paper states geodesic distance is **"well defined" but unstable** (the opposite of "ill-defined"), and makes **no** "volume is semantically empty" claim. Its predicted consequence is reduced prompt-stability under token substitution.
- **Severity:** HIGH. This is a scope mismatch on a load-bearing dismissal; a reviewer who reads Robinson can discredit the C7 verdict and cast doubt on the rest of the sweep.
- **Corrected wording:** Recast C7 from "Covered **and** feasibility refuted" to **"Covered (pre-empted by BeamDR/Das/HyDE) and geometrically *constrained*, not refuted."** Cite Robinson et al. honestly for irregularity of the **token-input** subspace and prompt-stability risk. Attribute the actual case against free continuous navigation to **anisotropy/empty-volume (Cai et al., ICLR 2021)** and **interpolation ≠ geodesic (Shao et al.)**, which motivate the discrete-substrate-with-transient-waypoints design rather than forbidding it. Drop claim (g) as a standalone novelty but sell the drop as "unnecessary and unreliable vs. NN_k-resolved waypoints," not "physics says it's impossible." Also delete the standalone "most of the volume is semantically empty" phrasing wherever it is attributed to Robinson (it belongs, if anywhere, to Cai et al.).

### 3.2 Self-Ask as killer cite for C3 (SEVERITY: MEDIUM)

- **What the docs say:** POSITIONING §2 Theme 1 ("decompose into distant sub-questions and compose back") and §3 C3 killer trio; README §4.2 line 48. Used to imply Self-Ask models "leave the cluster and return."
- **What the source supports:** Self-Ask does **linear factual sub-question decomposition** and is the source of Bamboogle. It has **no** notion of semantic space, clusters, trajectory, or leaving/returning to a region. It is a weak fit as a "controlled semantic jump" antecedent.
- **Severity:** MEDIUM. It does not break the C3 = "Covered" verdict (the union of Granovetter + DeepPath still covers originality), but it over-claims that a single prior method performed the conditioned jump.
- **Corrected wording:** Describe Self-Ask as "sequential sub-question decomposition (source of Bamboogle)" and add: **"no single prior method computes the query- and explored-set-conditioned 'leave the cluster and return' operator — which is why the survivor is C5, not C3."** Do not phrase C3 as "most pre-empted" via a single antecedent.

### 3.3 MuSiQue as "primary bridge benchmark" (SEVERITY: LOW)

- **What the docs say:** POSITIONING §2 Theme 8 and §6; README §1 line 36 ("built to require [bridge hops]").
- **What the source supports:** Correct on mechanism (bottom-up composition + Disconnection Filtering; single-hop model drops ~30 F1). But MuSiQue spans **six reasoning-graph shapes over 2–4 hops**, not only bridge-entity composition.
- **Severity:** LOW. **Corrected wording:** "MuSiQue is adversarially filtered against disconnected/shortcut reasoning across several 2–4-hop reasoning shapes" — do not narrow it to a pure "bridge" benchmark.

### 3.4 Minor characterization softenings (SEVERITY: LOW, verified MOSTLY_ACCURATE)

- **HNSW** — factually a multi-step greedy coarse-to-fine traversal, but it *locates* neighbors; equating that search path with STP's semantic "trajectory over content" is a rhetorical stretch. Keep the cite; add a one-clause caveat.
- **ITAL** — is an information-theoretic **active-learning / annotation-selection** method using **mutual information** (explicitly distinguished from pure info-gain/entropy), not a ranking/retrieval objective per se. Soften "EIG-as-retrieval-objective."
- **HopRAG** — its "Helpfulness" already includes a coarse **visited-set** signal (visit-count IMP term); it lacks only the answer-frontier / (1−coverage) conditioning and its per-hop reasoning is local. The docs' claim that HopRAG "drops the frontier factor and the flow/breadth structure" is right; the incidental implication that it is *purely* pairwise/history-free is slightly off.
- **Patankar et al.** — query-free, purely structural reward is correct, but the CPT variant rewards **compressible/clustered** regions, not "novel" ones; "scores moves into novel graph regions" fits only the IGT half.
- **MA-DPR** — geodesic-over-kNN-graph is right, but the graph **approximates a continuous manifold**; discreteness (hop-counting) is one of several edge-cost options (another uses the continuous graph-Laplacian spectrum). Do not cite it as asserting a discrete substrate is the honest one.

---

## 4. Formal errors

All line references are POSITIONING §5 unless noted; the same errors recur in README §4.3–4.4 and PAPER_v2 §4.5–4.7.

### 4.1 The Freeman-betweenness reduction is false (SEVERITY: HIGH)

- **Text (line 74):** "Setting V_t=V, ρ_t≡1 recovers Freeman betweenness exactly."
- **Problem:** In BS_t(v) the outer sum ranges over a ∈ V_t and the inner over z ∉ V_t. Setting V_t=V makes {z ∉ V_t} **empty**, so BS_t(v) ≡ 0 for every v. Zero is not Freeman betweenness; the cut structure (a inside, z outside) structurally cannot reduce to Freeman, whose sum requires **both** endpoints to range over all of V∖{v}. No charitable reading rescues it.
- **Fix:** Freeman is recovered only by **dissolving the cut**, not by V_t=V. Corrected statement: *"Removing the explored/frontier partition — letting both endpoint sums range over all of V∖{v}, with ρ_t ≡ 1 and π_t ≡ 1 (uniform) — recovers Freeman betweenness."* Then describe BS honestly as a cut-restricted, frontier-weighted variant that **generalizes betweenness by adding a partition**, not as a functional that contains Freeman among the special values of its own parameters. (README line 117 already writes "V_t = V, sinks = V, ρ_t ≡ 1, π_t ≡ 1" — this is closer but still self-contradictory: if V_t = V the frontier V∖V_t is empty regardless of "sinks = V"; the partition must be *dropped*, not set to the whole vertex set.)

### 4.2 π_t(a) is never defined (SEVERITY: HIGH)

- Appears only inside the BS formula (line 72); the π on line 65 is the unrelated RL policy. Domain, normalization, and purpose unstated. It also silently breaks the Freeman reduction a second time (standard betweenness has no per-source multiplier).
- **Fix:** Define it, e.g. *"π_t(a) = normalized explored-node importance, a distribution over V_t defaulting to uniform 1/|V_t|,"* and add "π_t ≡ 1 (uniform)" to the corrected reduction condition — or delete it if redundant with ρ_t.

### 4.3 𝓡(·) (reachability operator) is never defined (SEVERITY: HIGH)

- The supervised training target y_t(v) = |𝓡(V_t∪{v})∩A| − |𝓡(V_t)∩A| (line 80; README line 131; PAPER_v2 line 120) is the crux of the C5 contribution, yet "reachable" is undefined (k-hop closure? PPR mass above threshold? TopK-retrievable-from?). Without it the target is not computable. README line 133 says "τ-hop reachable set" — so README is slightly better specified than POSITIONING, but τ is never pinned.
- **Fix:** Define 𝓡 explicitly, e.g. *"𝓡(S) = nodes reachable from S within the retrieval graph under the agent's h-hop k-NN expansion rule,"* and fix (h, k / τ) for the 2Wiki/MuSiQue supervision.

### 4.4 "Cumulative EIG = minimizing final answer entropy" conflates two quantities (SEVERITY: HIGH)

- **Text (lines 63–65):** presents Σ EIG = E_π[H(A|C_0) − H(A|C_H)] as an identity.
- **Problem:** Per-step EIG is an expectation over *not-yet-observed* content under the current belief; the telescoped sum on line 65 uses *realized* conditionals. Eq. (65) is a true **per-trajectory** telescoping (realized on both sides), but Σ(per-step EIG) equals it **only in expectation, and only when observations are drawn from the model's own predictive posterior** — i.e. under a calibration assumption, not an identity. Under the true corpus distribution a miscalibrated reader makes summed EIG a biased estimate.
- **Fix:** Split the claim: (i) keep eq. (65) as exact per-trajectory telescoping of *realized* entropy ("holds pathwise for any fixed π"); (ii) add the bridging line E_{X_a∼p(·|q,C_t)}[H(A|q,C_t) − H(A|q,C_{t+1})] = EIG(a|q,C_t), so Σ EIG equals the endpoint reduction **only** under the model's predictive posterior; (iii) flag explicitly that under the empirical corpus this is a calibration assumption.

### 4.5 Greedy per-step EIG is not justified by telescoping (SEVERITY: MEDIUM)

- Telescoping holds for any *fixed* π, but the docs use it to justify a **greedy** per-step rule (line 63/67). Conditional MI is not additive or submodular in general, so greedy can be arbitrarily suboptimal without an adaptive-submodularity condition. The doc's own line 67 invokes BatchBALD precisely because per-step EIG double-counts redundancy — an admission that Σ(marginal EIG) ≠ joint I(A; X_{a_1..a_H}), in tension with the clean framing.
- **Fix:** State the joint objective is I(A; observed set); note greedy is a heuristic with a (1−1/e) guarantee only under adaptive submodularity, and frame the per-step sum as the greedy approximation to the BatchBALD conditional selection already referenced.

### 4.6 κ, r(z|q), σ(a,z), ρ_t-as-distribution — undefined / overloaded symbols (SEVERITY: MEDIUM–LOW)

- **κ (line 76, MEDIUM):** used three times as a similarity/kernel but never introduced; the reader cannot tell if it is w_ij = max(0,cos) (line 53), a learned kernel, or a PPR/diffusion kernel — load-bearing because the surrogate is claimed to "approximate BS." Define at first use. (README line 123 *does* define κ as a Gaussian RBF kernel — so README is more complete than POSITIONING, but then uses a **different** kernel form than the max(0,cos) edge weight w, which should be reconciled.)
- **r(z|q) (line 71, MEDIUM):** labeled "relevance" but no form/range; ρ_t ≡ 1 silently requires r(z|q) = 1 for all z. Define concretely (e.g. max(0, cos(x_z,x_q))) and state the reduction requires both r ≡ 1 and (1 − max_u w_uz) ≡ 1.
- **σ(a,z), σ(a,z|v) (line 72, LOW):** standard shortest-path counts but never stated; a=z / σ=0 division-by-zero degenerate cases unhandled. Add one line defining them and the "omit σ(a,z)=0 terms" convention.
- **ρ_t as sampling distribution (line 76, LOW):** "z ∼ ρ_t" requires normalization over {z ∉ V_t}, but ρ_t is defined as an unnormalized product; the symbol is overloaded as both a BS weight and a density. Normalize or rename.

### 4.7 Discounting / horizon inconsistencies (SEVERITY: LOW)

- The MDP tuple carries a general γ (line 55) and R = EIG + β·BS − λ·cost (line 82), but the telescope (65) is undiscounted (γ=1) and covers only the EIG term; a discounted sum does not telescope for γ≠1, and BS/cost do not telescope at all. Also: line 63 specifies a VoI **stopping time** τ while eq. (65) uses a **fixed** horizon H, and "H" is overloaded as both horizon and entropy. **Fix:** state the telescope applies to the undiscounted EIG component only; replace fixed H with stopping time τ in the endpoint (E[H(A|C_0) − H(A|C_τ)]); rename the horizon symbol to avoid clashing with entropy H.

### 4.8 POMDP machinery over-dressed / mis-attributed (SEVERITY: LOW)

- The hidden state a⋆ is **static** (belief update is a pure Bayesian filter with **no transition kernel**, T = identity), so STP is a finite-horizon Bayesian **active-hypothesis-testing / experimental-design** problem — a degenerate POMDP — which is exactly why the objective coincides with sequential BALD/DAD (consistent with the C2 concession). Listing T in the tuple and repeatedly invoking "POMDP" claims more machinery than used. Separately, the "standard Åström/Kaelbling equivalence" (line 55; README line 79 "Åström 1965; Kaelbling et al. 1998") over-packages: the substantive result is Åström (1965) — belief is a sufficient statistic ⇒ belief-MDP; the "history-MDP" is trivial (histories are Markov by construction); Kaelbling et al. (1998) is the POMDP-**planning** survey, not the source of a distinct equivalence. **Fix:** annotate T = id (or drop it) and note the static-answer degeneracy; cite Åström for the sufficient-statistic result and Kaelbling as a planning reference only.

---

## 5. The survivor

**Verdict: the Bridge Score contribution SURVIVES the under-reach (pre-emption) attack — but narrowly, and the docs currently overstate the margin.**

The specific **triple conjunction** — (a) history-conditioned on the explored set V_t, (b) answer-frontier-conditioned via ρ_t, and (c) flow/breadth-across-the-cut betweenness — **plus** the reachable-gold-evidence-gain supervision target, is **not computed together by any verified prior work.** The kill attempts fail factor-by-factor:

- **QAFD-RAG (arXiv 2605.18775)** — the tightest new squeeze — delivers (b)+(c) (query-conditioned flow-across-a-cut with coverage guarantees) but its diffusion is **single-shot from query-anchor seeds with no visited set** → drops (a). (An earlier WebFetch summary calling it "iterative/history-conditioned" was model confabulation against a PDF whose math was in image streams; the arXiv abstract contradicts it.)
- **BridgeRAG (arXiv 2604.03384)** — bridge- and query-conditioned, but a **pairwise LLM relevance judge that explicitly avoids graph/flow methods** → drops (c) (same bucket as HopRAG).
- **ToG (2307.07697)** — history- and query-conditioned via beam search, but scores a pairwise relation×entity relevance product → drops (c).
- **Subset-betweenness (`networkx.betweenness_centrality_subset`) + relevance-weighted betweenness (arXiv 1905.03300)** — supply the **exact mechanical skeleton** (set S=V_t, T=frontier, weight by ρ_t) but are static, query-free-or-static-relevance network-analysis tools never applied to explored-vs-frontier retrieval cuts → neither computes the conjunction.

**But the survivor is not pre-empted, it is *pressured*:** the docs' §2/§3 basket is materially stale and misses the closest neighbors (QAFD-RAG, BridgeRAG, subset-betweenness). The honest reading is: **factor (a) — dynamic re-seeding from the growing V_t with a not-yet-explored novelty term — is the true differentiator vs. QAFD-RAG's single-shot query-seeded flow.** That must become an explicit ablation and falsifier, and the docs should pre-empt the reviewer who spots that BS_t is *mechanically* subset-betweenness with relevance weighting, locating novelty strictly in the dynamic V_t-reseeding + reachability target (not the formula shape). The overall repositioning (narrow findings-track paper) is reinforced, not weakened, by these discoveries.

---

## 6. Cross-doc issues

Verified directly against the source files:

1. **HIGH — Benchmark status contradiction.** README §5 titles it "The second artifact: a benchmark that credits detours," the TL;DR (line 17) lists it among what "STP actually proposes," line 145 calls it "reusable independently of STP," and POSITIONING §4-II lists it as a co-equal defensible contribution — but **PAPER_v2 §6 line 167** concedes: *"a benchmark that credits productive bridges remains future work"* and that detour-crediting has *"not been established."* One doc set delivers it; the other says it does not yet exist. This is the sharpest overclaim relative to the honest baseline.
2. **MEDIUM — Reward coefficient symbol.** README §4.4 line 135 and POSITIONING line 82 use **R = EIG + β·BS − λ·cost**; PAPER_v2 §4.7 line 127 uses **α·BS**, and PAPER_v2 is **internally inconsistent** (line 147 requires "full **β** beat both" ablations while defining the reward with α).
3. **MEDIUM — H1 threshold.** README line 153 and POSITIONING line 86 require the ≥3 F1 bridge win on **"≥2 of 3 primary datasets"**; PAPER_v2 H1 (line 137) **omits the multi-dataset clause** entirely. For a paper whose selling point is a frozen pre-registered protocol, an inconsistent pre-registration is substantive.
4. **LOW — H3 tolerance band.** README line 155 and PAPER_v2 line 137 say STP does not beat iterative-RAG **"by more than 1 F1"** at matched cost; POSITIONING line 86 says STP **"does not beat iterative-RAG"** with **no** tolerance band.
5. **LOW — Unsourced "beating PageRank."** PAPER_v2 §2 line 39 adds *"…beating PageRank"* to the Patankar antecedent; absent from POSITIONING Theme 6 and README line 133. It is an in-context-unsourced comparative result (and cuts toward over-crediting prior art).
6. **LOW — DeepPath cited under two ids.** README line 181 / POSITIONING Theme 4 & C3 use **arXiv 1707.06690**; PAPER_v2 §2 line 33 uses **aclanthology D17-1060**. Normalize to one.
7. **LOW — Inconsistent citation roster.** IGP (2601.17532), Cai et al., Chen 2009, and Search-o1 appear in POSITIONING but are **dropped from README §7 and PAPER_v2** — IGP especially, as a near antecedent to the conceded EIG objective, should appear in all three. Search-o1 is in README §7 / POSITIONING Theme 2 but omitted from PAPER_v2 §2 line 27.
8. **LOW — Author-attribution collision in PAPER_v2 only.** PAPER_v2 collapses HopRAG (line 31), PropRAG (line 29), MA-DPR (line 41) all to **"Liu et al., 2025"** and PathRAG/ReSearch to **"Chen et al., 2025"** — four distinct papers by different groups; at least the HopRAG/PropRAG/MA-DPR "Liu et al." triple is likely a mis-attribution. README/POSITIONING avoid this by omitting author names.

---

## 7. Prioritized correction list

**HIGH severity**

1. **Fix the Freeman reduction** everywhere it appears (POSITIONING line 74; README line 117; PAPER_v2 §4.5): replace "Setting V_t=V, ρ_t≡1" with "**Removing the explored/frontier partition** (both endpoints over V∖{v}), with ρ_t ≡ 1 and π_t ≡ 1 uniform, recovers Freeman betweenness." Verify the corrected form actually yields normalized betweenness before publishing.
2. **Recalibrate C7 from "feasibility refuted" to "geometrically constrained, not refuted."** Cite Robinson et al. only for token-input-subspace irregularity + prompt stability; move the actual navigation caution to Cai et al. (anisotropy) and Shao et al. (interpolation≠geodesic); delete "geodesics ill-defined" and the standalone "volume is semantically empty" attributions to Robinson. Keep claim (g) dropped, but as "unnecessary/unreliable vs. NN_k waypoints," not "impossible." (POSITIONING §1/§3/§5; README §4.3/§8; PAPER_v2 §2/§6.)
3. **Define π_t, 𝓡(·), and (h,k/τ)** in the BS formula and supervision target (POSITIONING lines 72/80; README lines 113/131–133; PAPER_v2 §4.5–4.6). The target must be computable.
4. **Split the "cumulative EIG = final entropy" claim** into a pathwise telescoping identity + an explicit calibration/predictive-posterior assumption (POSITIONING lines 63–65; README lines 97–101; PAPER_v2 §4.7).
5. **Resolve the benchmark-status contradiction.** Either downgrade README §5 title + TL;DR (line 17) + line 145 and POSITIONING §4-II to "**proposed** protocol / future work," matching PAPER_v2 §6 line 167 — or promote PAPER_v2 to match — but make all three agree. Recommend the humbler framing.

**MEDIUM severity**

6. **Normalize the reward coefficient to β** in PAPER_v2 §4.7 line 127 and fix the internal α/β clash at line 147.
7. **Add the "≥2 of 3 primary datasets" clause to PAPER_v2 H1** (line 137) to match README/POSITIONING.
8. **Do not present greedy per-step EIG as optimal via telescoping**; frame it as the BatchBALD greedy-conditional approximation to joint MI (POSITIONING lines 63/67).
9. **Define κ at first use and reconcile it** with the edge weight w = max(0,cos): POSITIONING uses undefined κ, README line 123 uses a Gaussian RBF — pick one and state whether/why it differs from w.
10. **Define r(z|q)** and state the reduction requires r ≡ 1 (POSITIONING line 71).
11. **Soften Self-Ask (C3)** to "linear sub-question decomposition, source of Bamboogle" and add "no single prior method computes the query- and explored-set-conditioned jump — hence the survivor is C5, not C3."
12. **Add QAFD-RAG, BridgeRAG, subset-betweenness + relevance-weighted betweenness** to §2/§3/§5, state plainly that BS_t is mechanically subset-betweenness with ρ_t weighting, and locate novelty in dynamic V_t-reseeding + reachability target. Add the falsifier: "if frozen query-only diffusion equals V_t-reseeded diffusion, factor (a) is inert and STP collapses to QAFD-RAG"; add BridgeRAG as a §6 baseline alongside HopRAG.

**LOW severity**

13. **Reconcile H3** tolerance: add or remove the "1 F1" band consistently across POSITIONING line 86 / README line 155 / PAPER_v2 line 137.
14. **Annotate T = id** (static answer) and note STP is a degenerate POMDP / Bayesian experimental-design problem; fix the Åström/Kaelbling packaging (POSITIONING line 55; README line 79).
15. **Fix discount/horizon:** restrict the telescope to undiscounted EIG, replace fixed H with stopping time τ, rename the horizon symbol to avoid clashing with entropy H.
16. **Add σ(a,z) definitions + σ=0 convention**, and normalize ρ_t (or rename it) when used as a sampling density.
17. **Remove/source the "beating PageRank" claim** (PAPER_v2 line 39); soften Patankar to note CPT rewards compressible (not novel) regions.
18. **Normalize DeepPath's id** (1707.06690) across all three docs.
19. **Unify the citation roster** — add IGP, Search-o1 (and, if kept, Cai et al./Chen 2009) to README §7 and PAPER_v2 §2.
20. **Fix PAPER_v2 author attributions** — split the "Liu et al., 2025" / "Chen et al., 2025" collisions (HopRAG vs PropRAG vs MA-DPR vs PathRAG vs ReSearch) against real author lists.
21. **Add one-clause caveats** for HNSW (search-time locating path, not a content trajectory), ITAL (MI-based annotation selection, not a ranking objective), MA-DPR (kNN graph approximates a continuous manifold; discreteness is one option), HopRAG (already carries a coarse visit-count history signal), and MuSiQue (multiple 2–4-hop shapes, not solely bridge).

**Inconclusive / not checked:** The existence and exact content of QAFD-RAG (2605.18775), BridgeRAG (2604.03384), and relevance-weighted betweenness (1905.03300) rest on the steelman's reported abstracts; I did not independently re-fetch them. Their ids post-date the stated knowledge horizon and should be verified before being added as citations. All other findings above are grounded in the supplied verification results and the directly re-read source docs.

Source files (all absolute): `/Users/pawel/SemanticTrajectoryPlanning/docs/POSITIONING.md`, `/Users/pawel/SemanticTrajectoryPlanning/README.md`, `/Users/pawel/SemanticTrajectoryPlanning/docs/PAPER_v2_SKELETON.md`, `/Users/pawel/SemanticTrajectoryPlanning/docs/README_v1_manifesto.md`.
