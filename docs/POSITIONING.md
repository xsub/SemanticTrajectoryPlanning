# How to Make STP a Serious Research Contribution

## 1. Honest verdict

Semantic Trajectory Planning, as currently framed, is **not a new paradigm — it is a re-vocabularization of an already-mature one**, and it must be repositioned drastically or it will be rejected on its own weakest claims. Every load-bearing idea — retrieval as a sequential, stateful, history-aware process (claim a/d); optimizing uncertainty/information-gain instead of similarity (b); deliberate "jumps" to distant regions and back (c); a learnable bridge metric (e); a policy taxonomy (f); operating in continuous embedding space (g) — has a named, verified, often years-old antecedent, and its most distinctive geometric bet (free navigation of a continuous semantic manifold) is **affirmatively contradicted** by primary geometry results, not merely unoriginal. The integrative framing (one unified exploration-policy object) and two narrow choices — EIG used as the *per-candidate ranking function* in the retrieval slot, and a *frontier/novelty-seeking* Bridge Score conditioned on the explored set and the answer-relevant frontier — are the only defensible openings, and both are incremental specializations encroached by 2025 work. STP can become a serious *narrow* contribution (a specific Bridge-Score functional + a benchmark that credits productive detours, validated by a selective per-cost win against HopRAG); it cannot honestly be sold as introducing the paradigm.

## 2. The landscape — what already exists, by theme

**Theme 1 — Iterative / active / adaptive RAG (sequential, uncertainty-driven retrieval).** Retrieval reframed as a stateful loop driven by uncertainty rather than one-shot TopK is the established premise here: FLARE (confidence-triggered re-retrieval, [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)), IRCoT (interleave retrieval with each CoT step, [arXiv:2212.10509](https://arxiv.org/abs/2212.10509)), Self-Ask (decompose into distant sub-questions and compose back, [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)), ITER-RETGEN ([arXiv:2305.15294](https://arxiv.org/abs/2305.15294)), Self-RAG ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511)), Adaptive-RAG (complexity-routed strategy spectrum, [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)), DRAGIN (token-entropy trigger + history-spanning query, [arXiv:2403.10081](https://arxiv.org/abs/2403.10081)), IterDRAG (inference-scaling of iterative retrieval, [arXiv:2410.04343](https://arxiv.org/abs/2410.04343)), and Stop-RAG (iterative RAG as a finite-horizon MDP with a value-based stopping controller, [arXiv:2510.14337](https://arxiv.org/abs/2510.14337)).

**Theme 2 — Agentic / RL search agents (retrieval as actions in a loop).** ReAct (thought–action–observation trajectories, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)), WebGPT (RL-trained browsing, [arXiv:2112.09332](https://arxiv.org/abs/2112.09332)), Search-o1 ([arXiv:2501.05366](https://arxiv.org/abs/2501.05366)), and the RL-search family Search-R1 ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516)), R1-Searcher ([arXiv:2503.05592](https://arxiv.org/abs/2503.05592)), ReSearch ([arXiv:2503.19470](https://arxiv.org/abs/2503.19470)), DeepResearcher ([arXiv:2504.03160](https://arxiv.org/abs/2504.03160)). The subfield is mature enough to have its own taxonomy survey ([arXiv:2510.16724](https://arxiv.org/abs/2510.16724)).

**Theme 3 — Tree / trajectory / MCTS search over retrieval+reasoning.** Tree of Thoughts ([arXiv:2305.10601](https://arxiv.org/abs/2305.10601)), RAP ([aclanthology.org/2023.emnlp-main.507](https://aclanthology.org/2023.emnlp-main.507/)), LATS ([arXiv:2310.04406](https://arxiv.org/abs/2310.04406)), MCTS-RAG (expand/rollout/backup with backtracking = "jump out, explore, return", [arXiv:2503.20757](https://arxiv.org/abs/2503.20757)), PropRAG (beam search over proposition paths, [arXiv:2504.18070](https://arxiv.org/abs/2504.18070)).

**Theme 4 — Graph RAG and multi-hop KG retrieval.** Think-on-Graph ([arXiv:2307.07697](https://arxiv.org/abs/2307.07697)) and ToG-2 (coarse-to-fine graph/text, [arXiv:2407.10805](https://arxiv.org/abs/2407.10805)), HippoRAG (PPR-spread retrieval reaching distant nodes, [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)), Microsoft GraphRAG (community hierarchy, [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)), PathRAG (relational paths as the retrieval unit, [arXiv:2502.14902](https://arxiv.org/abs/2502.14902)), LightRAG ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779)), HopRAG (LLM pseudo-query bridge edges + learned hop "helpfulness", [arXiv:2502.12442](https://arxiv.org/abs/2502.12442)), and the RL-walk lineage DeepPath ([arXiv:1707.06690](https://arxiv.org/abs/1707.06690)), MINERVA ([arXiv:1711.05851](https://arxiv.org/abs/1711.05851)), SR ([arXiv:2202.13296](https://arxiv.org/abs/2202.13296)), RAG-Gym ([arXiv:2502.13957](https://arxiv.org/abs/2502.13957)).

**Theme 5 — Active learning / Bayesian experimental design / EIG.** BALD (EIG-as-acquisition, [arXiv:1112.5745](https://arxiv.org/abs/1112.5745)), BatchBALD (joint redundancy-aware MI, [arXiv:1906.08158](https://arxiv.org/abs/1906.08158)), DAD (amortized sequential BED policy, [arXiv:2103.02438](https://arxiv.org/abs/2103.02438)), ITAL (EIG-as-retrieval-objective, [arXiv:1809.02337](https://arxiv.org/abs/1809.02337)), UoT ([arXiv:2402.03271](https://arxiv.org/abs/2402.03271)), grounded-retrieval EIG questioning ([arXiv:2311.08584](https://arxiv.org/abs/2311.08584), [arXiv:2406.17453](https://arxiv.org/abs/2406.17453)), and — most directly in RAG — InfoGain-RAG ([arXiv:2509.12765](https://arxiv.org/abs/2509.12765)) and IGP ([arXiv:2601.17532](https://arxiv.org/abs/2601.17532)). Curiosity/intrinsic-motivation: VIME ([arXiv:1605.09674](https://arxiv.org/abs/1605.09674)), count-based exploration ([NeurIPS 2016](https://papers.nips.cc/paper/6383-unifying-count-based-exploration-and-intrinsic-motivation)), ICM ([ICML 2017](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)).

**Theme 6 — Network science of bridges/brokerage.** Betweenness centrality (Freeman 1977, [jstor.org/stable/3033543](https://www.jstor.org/stable/3033543)), weak ties (Granovetter 1973, [jstor.org/stable/2776392](https://www.jstor.org/stable/2776392)), structural holes (Burt 1992), brokerage-role typology (Gould–Fernandez 1989, [jstor.org/stable/270949](https://www.jstor.org/stable/270949)), edge-betweenness communities (Girvan–Newman 2002, [pnas.org/doi/10.1073/pnas.122653799](https://www.pnas.org/doi/10.1073/pnas.122653799)), **bridging centrality / bridging coefficient = "probability of leaving the local cluster"** (Hwang et al. 2008, [dl.acm.org/doi/10.1145/1401890.1401934](https://dl.acm.org/doi/10.1145/1401890.1401934)), literature-based discovery / ABC bridge chains (Swanson 1986), pivotal-paper betweenness (Chen 2009, [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1751157709000200)), and a learnable graph-curiosity GNN that scores moves into novel graph regions (Patankar et al. 2023, [arXiv:2307.04962](https://arxiv.org/abs/2307.04962)).

**Theme 7 — Embedding-space geometry (is continuous navigation feasible?).** Token embeddings provably violate the manifold hypothesis (Robinson et al., NeurIPS 2025, [arXiv:2504.01002](https://arxiv.org/abs/2504.01002)); interpolation ≠ geodesic on curved latent spaces (Shao et al., [CVPR 2018 W](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Shao_The_Riemannian_Geometry_CVPR_2018_paper.pdf)); anisotropy leaves most volume semantically empty (Cai et al. ICLR 2021, [openreview](https://openreview.net/pdf?id=xYGNO86OWDH)); HyDE shows synthetic points are transient queries that snap back to nodes ([arXiv:2212.10496](https://arxiv.org/abs/2212.10496)); MA-DPR computes geodesics over a discrete k-NN graph ([arXiv:2509.13562](https://arxiv.org/abs/2509.13562)); HNSW is already greedy coarse-to-fine multi-step traversal ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)); BeamDR does sequential dense-space evidence chaining ([arXiv:2104.05883](https://arxiv.org/abs/2104.05883)); Das et al. reformulate the dense query vector across steps ([arXiv:1905.05733](https://arxiv.org/abs/1905.05733)).

**Theme 8 — Multi-hop benchmarks + trajectory metrics.** HotpotQA ([arXiv:1809.09600](https://arxiv.org/abs/1809.09600)), 2WikiMultiHopQA ([aclanthology.org/2020.coling-main.580](https://aclanthology.org/2020.coling-main.580/)), MuSiQue (adversarially filtered so disconnected retrieval can't shortcut, [arXiv:2108.00573](https://arxiv.org/abs/2108.00573)), StrategyQA ([arXiv:2101.02235](https://arxiv.org/abs/2101.02235)), FanOutQA ([arXiv:2402.14116](https://arxiv.org/abs/2402.14116)), FRAMES (2–15 articles, end-to-end, [arXiv:2409.12941](https://arxiv.org/abs/2409.12941)), Bamboogle ([arXiv:2210.03350](https://arxiv.org/abs/2210.03350)).

## 3. Claim-by-claim novelty audit (C1–C7)

C1 = the umbrella reframing (retrieval as sequential trajectory planning instead of static TopK).

| Claim | Verdict | Killer citation |
|---|---|---|
| **C1** Reframe retrieval as sequential trajectory planning vs static TopK | **Covered** | ReAct ([2210.03629](https://arxiv.org/abs/2210.03629)) + IRCoT ([2212.10509](https://arxiv.org/abs/2212.10509)); HNSW shows even "TopK" is already a multi-step graph walk ([1603.09320](https://arxiv.org/abs/1603.09320)) |
| **C2** EIG / uncertainty reduction (not similarity) as the retrieval objective | **Covered** | BALD ([1112.5745](https://arxiv.org/abs/1112.5745)) → ITAL in retrieval ([1809.02337](https://arxiv.org/abs/1809.02337)) → InfoGain-RAG, "Document Information Gain" reranking ([2509.12765](https://arxiv.org/abs/2509.12765)) |
| **C3** Controlled Semantic Jumps (leave cluster, visit distant region, return) | **Covered** (most pre-empted) | Granovetter weak ties ([2776392](https://www.jstor.org/stable/2776392)) + Self-Ask ([2210.03350](https://arxiv.org/abs/2210.03350)) + DeepPath diversity reward ([1707.06690](https://arxiv.org/abs/1707.06690)) |
| **C4** Stateful, history-aware MDP with "Exploration Memory" | **Covered** | Stop-RAG (iterative RAG = finite-horizon MDP, [2510.14337](https://arxiv.org/abs/2510.14337)); MINERVA (RNN path-history MDP, [1711.05851](https://arxiv.org/abs/1711.05851)) |
| **C5** Semantic Bridges + learnable frontier-seeking Bridge Score | **Covered as stated; *narrow specialization* survives** | Hwang bridging coefficient ([1401890.1401934](https://dl.acm.org/doi/10.1145/1401890.1401934)); Patankar GNN graph-curiosity ([2307.04962](https://arxiv.org/abs/2307.04962)); HopRAG learned hop helpfulness ([2502.12442](https://arxiv.org/abs/2502.12442)) |
| **C6** Taxonomy of exploration policies | **Covered** | Agentic-RAG survey ([2501.09136](https://arxiv.org/abs/2501.09136)) + RL-agentic-search survey ([2510.16724](https://arxiv.org/abs/2510.16724)) already taxonomize the same axes |
| **C7** Continuous-embedding navigation (vs discrete graph) | **Covered *and* feasibility refuted** | Pre-empted by BeamDR ([2104.05883](https://arxiv.org/abs/2104.05883)); refuted by Robinson et al., "Token Embeddings Violate the Manifold Hypothesis" ([2504.01002](https://arxiv.org/abs/2504.01002)) |

The only thing that "survives" — and only as an **incremental specialization, not a standalone claim** — is the precise functional form of C5: a Bridge Score conditioned on the explored set **and** weighted by the answer-relevant frontier, distinct from (i) query-free brokerage centrality, (ii) answer-relevance edge scoring (HopRAG), and (iii) query-free curiosity bonuses.

## 4. The defensible narrow contribution

STP should claim **exactly one mechanism and one artifact, nothing broader**:

**(I) A state-conditioned, answer-frontier-weighted Bridge Score** — a betweenness-style flow quantity across the *explored cut*, weighted by an answer-frontier potential $\rho_t$, with a tractable diffusion/kernel surrogate and a supervised target of *reachable-gold-evidence gain conditioned on the explored set*. Its defensibility rests on a single triple distinction no cited work computes together: it is simultaneously (a) **history-conditioned** (depends on the visited set $V_t$, unlike Freeman/Hwang centrality), (b) **query-conditioned toward the answer frontier** (unlike VIME/ICM/count-based curiosity), and (c) **flow/breadth-based across the cut** (unlike HopRAG's pairwise relevance edge score). Drop any one factor and you recover a named baseline — which is precisely why all three together are the contribution.

**(II) A benchmark protocol that credits productive detours** — a frozen, pre-registered "bridge-jump" labeling of multi-hop items (a gold hop whose next passage is *not* in the TopK neighbors of the accumulated context), so that off-path-but-enabling retrieval is rewarded rather than penalized as distractor noise — which no existing multi-hop benchmark does.

Why this is the right altitude: it is the one region the adversarial sweep could not pre-empt verbatim, it is **falsifiable** (it predicts a *selective* per-cost win on bridge items and a *null* on easy items), and it is honest about resting on a discrete substrate. Everything else (trajectory, EIG-as-objective, MDP, taxonomy, continuous navigation) must be explicitly conceded to prior art in the related-work section.

## 5. Formal core

**Substrate decision (load-bearing).** Commit to a **discrete corpus graph** $G=(V,E,w)$ (HNSW k-NN and/or KG), $w_{ij}=\max(0,\cos(x_i,x_j))$. Continuous vectors are admitted **only as transient query waypoints** resolved by $\mathrm{NN}_k(w)$ (HyDE regime); off-node points are never retrievable states. Claim (g) is dropped — it is pre-empted (BeamDR/Das/DeepPath) *and* geometrically refuted (Robinson et al., NeurIPS 2025): if the space is not a manifold, geodesic "trajectories" through interpolated points are ill-defined and most of the volume is semantically empty.

**The MDP.** STP is a finite-horizon **POMDP** with hidden answer $a^\star$, equivalently a **belief-MDP** on $b_t=p(a^\star\mid h_t)$, equivalently a **history-MDP** on the interaction history $h_t$ (the three are the standard Åström/Kaelbling equivalence; "Exploration Memory" $\psi(h_t)$ and belief $b_t$ are deterministic features of $h_t$, not new objects). Tuple $\langle\mathcal S,\mathcal A,\Omega,T,O,R,\gamma,H,b_0\rangle$. Unified action object (the integrative packaging): $\mathcal A=\{\texttt{STOP}\}\cup\{\texttt{RETRIEVE}(u)\}$, $u$ a graph neighbor (discrete) or a waypoint resolved by $\mathrm{NN}_k$. Deterministic belief update is Bayes:

$$b_{t+1}(a)=\frac{p(o_t\mid a)\,b_t(a)}{\sum_{a'}p(o_t\mid a')\,b_t(a')}.$$

**Information gain (the C2 objective, conceded as BALD-with-$A$-for-$\theta$).** Target = the *answer* $A$, not model parameters. Per-candidate acquisition score = expected reduction in answer entropy = mutual information:

$$\mathrm{EIG}(a\mid q,C_t)=H(A\mid q,C_t)-\mathbb{E}_{X_a}\!\big[H(A\mid q,C_t\cup\{c_a\})\big]=I(A;X_a\mid q,C_t).$$

The trajectory objective telescopes — maximizing cumulative EIG = minimizing final answer entropy, with a value-of-information stopping rule (stop when best marginal EIG $<\lambda\cdot\text{cost}$):

$$\max_\pi\;\mathbb{E}_\pi\!\Big[\textstyle\sum_t\big(H(A\mid q,C_t)-H(A\mid q,C_{t+1})\big)\Big]=\mathbb{E}_\pi\big[H(A\mid q,C_0)-H(A\mid q,C_H)\big].$$

Tractability: discretize to an ANN shortlist (A1, never rank raw continuous points); estimate $H(A\mid\cdot)$ by **semantic entropy** (sample + cluster by entailment, A2); use a first-token-logit proxy or an **amortized embedding-only surrogate** (DAD-style, A3) so EIG runs at retrieval latency; use **greedy-conditional** (BatchBALD) selection to kill redundancy (A4) — which makes "diversity/jumps" fall out of the objective rather than being bolted on.

**Bridge Score (the C5 contribution).** Answer-frontier potential and explored-cut betweenness:

$$\rho_t(z)=\underbrace{r(z\mid q)}_{\text{relevance}}\cdot\underbrace{(1-\textstyle\max_{u\in V_t}w_{uz})}_{\text{not-yet-explored}},\qquad
\mathrm{BS}_t(v)=\sum_{a\in V_t}\sum_{z\notin V_t}\rho_t(z)\,\pi_t(a)\,\frac{\sigma(a,z\mid v)}{\sigma(a,z)}.$$

Setting $V_t=V$, $\rho_t\equiv1$ recovers Freeman betweenness exactly — so $\mathrm{BS}_t$ is its *conditional, frontier-weighted* generalization. Tractable embedding surrogate (PPR seeded from the explored set, $O(k)$ per candidate):

$$\widehat{\mathrm{BS}}_t(v)\propto\underbrace{\big(1-\max_{u\in V_t}\kappa(x_v,x_u)\big)}_{\text{novelty vs memory}}\cdot\underbrace{\mathbb{E}_{z\sim\rho_t}[\kappa(x_v,x_z)]}_{\text{answer-frontier proximity}}\cdot\underbrace{H(\kappa(x_v,\cdot))}_{\text{connective breadth}}.$$

Learnable head $\mathrm{BS}_\theta(v\mid s_t)$ trained — crucially — on **reachable-gold-evidence gain**, not answer correctness (the latter is Stop-RAG/Search-R1, pre-empted):

$$y_t(v)=\big|\mathcal R(V_t\cup\{v\})\cap A\big|-\big|\mathcal R(V_t)\cap A\big|,$$

i.e. $v$ is rewarded for unlocking previously-unreachable gold evidence **even when $v$ itself contributes nothing to the answer**. Supervision is free from 2Wiki/MuSiQue evidence-path annotations. Total reward: $R=\mathrm{EIG}+\beta\cdot\mathrm{BS}-\lambda\cdot\text{cost}$.

## 6. The minimal falsifiable experiment (condensed)

**Hypotheses (pre-registered).** *H1 (per-cost bridge win):* on bridge-jump items, full STP beats the best baseline **including HopRAG** by ≥3 F1 at **equal-or-lower** token budget (paired bootstrap, $\alpha=0.05$, ≥2 of 3 primary datasets). *H2 (mechanism):* STP's **bridge-hop recall** gap exceeds its non-bridge-hop recall gap. *H3 (specificity — the discriminating clause):* on **non-bridge** items STP does **not** beat iterative-RAG at matched cost. A real win is *selective*.

**Bridge-jump labeling (frozen, published before runs).** Walk each gold chain; a hop is a *bridge hop* if gold$_i\notin\mathrm{TopK}(C_{<i})$ for $K\in\{5,10,20\}$. Bridge-jump item = ≥1 bridge hop.

**Datasets.** Primary: **MuSiQue, 2WikiMultiHopQA, FRAMES**. Stress probe: Bamboogle. **Control (non-bridge, for H3): HotpotQA-distractor** — never primary (it is shortcuttable).

**Baselines (all matched encoder/reader/corpus/budget).** TopK, oracle-budget TopK, IRCoT, FLARE, Self-Ask, Adaptive-RAG, GraphRAG, **HopRAG (the real adversary)**, IterDRAG — **plus B9 = STP minus Bridge Score (EIG+memory only), run as a first-class baseline.**

**Metrics, three co-equal axes.** Answer EM/F1 reported *separately* on bridge vs non-bridge (never pooled); hop-resolved retrieval recall incl. bridge-hop recall + evidence-path F1; **cost** (retrieval calls, LLM passes, tokens, p50/p95 latency). **Pre-registered rule: every EIG inner-loop LLM call is charged at full price — no free EIG.** Headline = F1-vs-token-budget Pareto on bridge items.

**Falsify if any of:** the bridge win vanishes once EIG cost is charged (it's just IterDRAG); STP ≈ HopRAG at matched cost (Bridge Score adds nothing); B9 ≈ full STP (Bridge Score inert); STP wins *uniformly* incl. non-bridge (generic compute, H3 violated); continuous-waypoint variant underperforms its discrete-resolution version (claim g refuted in-house — report it).

**Pilot gate (days, NumPy/NetworkX, no LLM/GPU).** A semantic-trajectory simulator over a stochastic-block-model concept graph with *planted bridge edges* (low cross-weight), node2vec embeddings, gold chains forced to cross ≥1 bridge, **oracle** EIG (belief = answer-nodes consistent with evidence) and **oracle** Bridge Score. Compare cosine-greedy / EIG-greedy / EIG+BS / curiosity-only / relevance-only while sweeping bridge count and cross-weight. Predicted signature: cosine collapses, EIG-greedy degrades (no drive to *leave* a cluster), STP holds — *and the gap appears only when bridges exist*. If EIG-greedy matches STP even with oracle access, the program is dead before any LLM spend.

## 7. Staged roadmap

1. **Manifesto (informal, 1–2 wk).** State the reframing as *integrative*, not foundational. Concede C1–C4, C6, C7 to prior art up front. Stake the flag only on the Bridge-Score functional and detour-crediting evaluation. No claims of a new paradigm.
2. **Defensible whitepaper (3–4 wk).** The §5 formalism verbatim, with the related-work table (§2/§3) conceding every pre-empted claim with citation. Includes the §6 pilot results as the existence proof of the mechanism under oracle conditions. Deliverable: a clean POMDP/belief-MDP statement + the three-factor Bridge Score + supervised reachability target, positioned strictly against HopRAG, Stop-RAG, BALD/ITAL, and the two surveys.
3. **Pilot (parallel with whitepaper).** Run the synthetic simulator; publish the planted-bridge sweep. This both gates the program and calibrates the bridge-difficulty regime the real benchmark over-samples. Pre-register H1–H3 and the cost-accounting rule now.
4. **Full paper + prototype (8–12 wk).** Implement on the discrete substrate (HNSW + diffusion surrogate + amortized EIG head + learned Bridge Score trained on 2Wiki/MuSiQue evidence paths). Run §6 against all baselines incl. HopRAG and the B9 ablation, with EIG cost charged. Publishable **iff** the win is selective, per-cost, and ablation-localized to the $\rho_t$-frontier term. If falsified, report the negative result honestly (it is still a contribution: "frontier Bridge Score does not beat HopRAG at matched cost").

## 8. Likely reviewer objections and rebuttals

**O1: "This is just iterative/agentic RAG with new words."** *Concede entirely for C1–C4/C6/C7.* Rebuttal: the contribution is *not* the reframing; it is the specific state-conditioned, answer-frontier-weighted Bridge Score (§5) and the detour-crediting benchmark (§4-II), which the related-work table shows no cited method computes. Everything else is positioned as integrative packaging, explicitly.

**O2: "EIG-over-similarity is BALD/ITAL/InfoGain-RAG."** *Concede.* Rebuttal: STP does not claim EIG-as-objective; it uses EIG only as a conceded prior, and its sole EIG-specific choice (per-candidate ranking on an ANN shortlist with an amortized surrogate) is presented as engineering, not novelty.

**O3: "The Bridge Score is Hwang's bridging coefficient / curiosity bonus / HopRAG."** Rebuttal: each of those drops one of the three required factors — Hwang is query-free and history-free; curiosity is query-free; HopRAG drops the frontier $(1-\text{cov}_t)$ factor and the flow/breadth structure. The falsifiable claim is that the *conjunction* (the BS-relevance-only and BS-novelty-only ablations both underperform full $\beta$) is load-bearing. If the experiment shows otherwise, the claim is withdrawn.

**O4: "Continuous-space navigation is incoherent."** *Full concession.* Rebuttal: claim (g) is dropped; the substrate is discrete; continuous vectors are HyDE-style transient queries. We cite Robinson et al. (NeurIPS 2025) ourselves and report any in-house continuous-vs-discrete ablation as confirming the discrete substrate.

**O5: "You only win because you spend more compute (cf. IterDRAG)."** Rebuttal: this is exactly why cost is a co-equal pre-registered axis, every EIG inner-loop call is charged, and the headline is a per-cost Pareto on bridge items with a non-bridge null (H3). If the win disappears at matched cost, we report it as falsification.

**O6: "Your benchmark is constructed to favor your method."** Rebuttal: the bridge-jump split is frozen and published *before* any system runs, defined by a baseline encoder's TopK failure (not by STP), with HotpotQA as an independent non-bridge control and the bridge-fraction-per-dataset reported as a sanity check.

**O7 (concession, not rebuttal): "Even the surviving contribution is incremental."** Agreed. STP should be submitted as a focused short paper / findings-track contribution claiming a narrow, validated specialization — not as a paradigm paper. That honest scoping is the difference between a serious contribution and a desk reject.