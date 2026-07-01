# Semantic Trajectory Planning (STP)

### A Frontier-Weighted Bridge Score for Multi-Hop Retrieval — with an honest account of what is, and is not, new

**Author:** Paweł Suchanecki · Independent Researcher
**Status:** Whitepaper v2 (positioning draft). Supersedes the v1 manifesto ([`docs/README_v1_manifesto.md`](docs/README_v1_manifesto.md)).

---

> **Read this first.** The first version of this document framed STP as a *paradigm shift* — retrieval reformulated from nearest-neighbor lookup into trajectory planning over semantic space. After an adversarial literature audit (81 verified papers across 8 subfields; see [`docs/POSITIONING.md`](docs/POSITIONING.md)), that framing does not hold: **the paradigm it announced already exists and predates this work by years.** This version concedes that openly and narrows STP to the one component the audit could not pre-empt. STP is presented here as an **integrative synthesis plus one incremental, falsifiable functional** — not as a new paradigm. If you came for the manifesto, it is preserved for the record; this is the version meant to survive scrutiny.

---

## TL;DR

- **What's already true (and not ours):** retrieval-augmented generation has already moved away from one-shot top-K. Iterative, active, agentic, and RL-trained retrieval reframe it as a sequential, history-aware, often uncertainty-driven decision process — many works formalize it explicitly as a Markov decision process. STP claims **none** of this as novel.
- **The one thing STP actually proposes:** a **frontier-weighted Bridge Score** — a *history-conditioned, answer-frontier-weighted* betweenness across the cut between what the agent has already explored and the answer's currently-unreachable region — with a tractable diffusion surrogate and a supervised target of *reachable-gold-evidence gain*. Plus a companion **benchmark protocol that credits productive detours** instead of penalizing them as distractor noise.
- **Why it might be real:** drop any one of its three conditioning factors and it collapses into a named baseline (betweenness centrality, a curiosity bonus, or HopRAG-style relevance scoring). The claim is that the *conjunction* is load-bearing — and that is empirically falsifiable.
- **Why it might not be:** it is an incremental specialization, it rests on an expensive information-gain objective, and the win — if any — must be *selective* (only on genuine bridge tasks) and *cost-matched*. The experiment is designed to falsify it cleanly.
- **Substrate:** a **discrete corpus graph**, not a continuous manifold. The "navigate continuous embedding space" idea is dropped — it is both pre-empted and geometrically refuted.

---

## 1. Motivation: the bridge hop

Current retrieval is excellent at finding what is *similar*. But similarity is not usefulness. Many reasoning tasks require chaining facts across **low-similarity edges**, where the next required passage is not among the nearest neighbors of what you have already gathered.

Concretely: a multi-hop query whose gold evidence chain contains at least one hop where `gold_i ∉ TopK(accumulated_context)`. A human analogue is the chain

```
build systems → dependency graphs → SAT solving → graph theory → scheduling
```

where each link is conceptually adjacent but no single similarity query spans the chain. Network science has held for fifty years that exactly these long-range **bridge ties** carry the novel, non-redundant information (Granovetter, 1973).

This failure mode is narrow and specific — and it is precisely what compositional multi-hop benchmarks (MuSiQue, 2WikiMultiHopQA) are built to require. **That single failure mode, not "retrieval in general," is what STP targets.**

---

## 2. What is borrowed vs. what is new

This is the integrity centerpiece of the document. Everything in the left column is conceded to prior art with citations in §7.

| STP idea (v1) | Status | It was already done by |
|---|---|---|
| Retrieval as a sequential **trajectory** instead of static top-K | **Borrowed** | ReAct, IRCoT; even HNSW top-K is itself a multi-step graph walk |
| **Expected information gain / uncertainty reduction** as the objective | **Borrowed** | BALD → ITAL (in retrieval) → InfoGain-RAG ("document information gain") |
| **Controlled semantic jumps** (leave a cluster, visit a distant region, return) | **Borrowed** (most pre-empted of all) | Granovetter weak ties; Self-Ask; DeepPath's diversity reward; MCTS-RAG backtracking |
| Retrieval as a **stateful, history-aware MDP** with "exploration memory" | **Borrowed** | Stop-RAG (iterative RAG = finite-horizon MDP); MINERVA (RNN path-history walk, 2018) |
| **Semantic bridges** + a learnable **Bridge Score** | **Borrowed *as stated*; one narrow specialization survives** | Freeman betweenness; Hwang bridging coefficient; HopRAG; Patankar graph-curiosity GNN |
| **Taxonomy** of exploration policies | **Borrowed** | Agentic-RAG survey; RL-agentic-search survey already taxonomize the same axes |
| **Continuous navigation** directly in embedding space | **Dropped** — pre-empted *and* geometrically refuted | BeamDR (pre-empt); Robinson et al. 2025, "Token Embeddings Violate the Manifold Hypothesis" (refute) |

**Net:** of seven distinguishing claims, **zero survive verbatim.** What survives — and only as an incremental specialization — is the precise *functional form* of the Bridge Score, defined next.

---

## 3. Substrate decision: a discrete corpus graph

Every downstream construct lives on this substrate, so the choice is load-bearing and we make it explicitly.

Let the corpus be chunks $\{c_i\}$ with frozen embeddings $x_i \in \mathbb{R}^d$. Build a symmetric $k$-NN similarity graph $G=(V,E,w)$ with $w_{ij}=\max(0,\cos(x_i,x_j))$, optionally augmented with KG relations or LLM-generated pseudo-query edges. **The only retrievable states are corpus nodes.** A continuous vector $v \in \mathbb{R}^d$ is admitted *only* as a transient query waypoint, immediately resolved to its neighborhood $\mathrm{NN}_k(v)$; it is never itself a returnable item.

This is not a convenience — it is a concession to primary results:

- Token-embedding subspaces **provably are not manifolds**, and their geodesics are unstable and need not track semantic distance (Robinson et al., 2025).
- Linear interpolation diverges from geodesics on curved latent spaces absent a pullback metric (Shao et al., 2018).
- Synthetic intermediate points function only as *queries* that snap back to corpus nodes (HyDE).
- Even "manifold-aware" retrieval computes geodesics over a **discrete** $k$-NN graph (MA-DPR), and production ANN (HNSW) is already a multi-step coarse-to-fine graph traversal.

So "continuous operation" in STP means *continuously move the query waypoint, discretely resolve every step to corpus nodes* — which collapses to a discrete belief-MDP at each step. We make **no** claim of free navigation over a smooth semantic space.

---

## 4. The framework

### 4.1 A belief-MDP (conceded structure)

STP is a finite-horizon **POMDP** whose hidden variable is the latent answer $a^\star$, equivalently a **belief-MDP** over $b_t = p(a^\star \mid h_t)$, where $h_t$ is the interaction history. The "exploration memory" $\psi(h_t) = (V_t, F_t, c_t)$ — visited embeddings, failed/low-yield paths, coverage pseudo-counts — is a *deterministic feature* of $h_t$, not a new object. The POMDP-over-$a^\star$ / belief-MDP-over-$b_t$ / history-MDP-over-$h_t$ equivalence is standard (Åström 1965; Kaelbling et al. 1998). **STP introduces none of this structure**; it only chooses the objective (§4.3) and the feature map $\psi$.

A single action unifies *when / what-where / stop*:

$$\mathcal{A} = \{\texttt{STOP}\} \cup \{\texttt{RETRIEVE}(u) : u \in \mathcal{U}\}$$

where $u$ is a graph neighbor of the current node (discrete move) or a waypoint resolved by $\mathrm{NN}_k$. A large-displacement action is a *controlled jump*; a "return" points $u$ into the visited set $V_t$. This unification is **integrative packaging** — each component is individually pre-empted (DRAGIN = when, HopRAG = where, Stop-RAG = stop).

On `RETRIEVE(u)` the observation $o_t = \mathrm{NN}_k(u)$ is deterministic given the fixed corpus, and the belief updates by Bayes:

$$b_{t+1}(a) = \frac{p(o_t \mid a)\, b_t(a)}{\sum_{a'} p(o_t \mid a')\, b_t(a')}.$$

### 4.2 Expected information gain about the answer (conceded objective)

The acquisition value of a candidate action is the mutual information between the answer and the prospective observation:

$$\mathrm{EIG}(u \mid h_t) = I(a^\star;\, o \mid h_t, u) = \mathbb{H}[b_t] - \mathbb{E}_{o}\big[\mathbb{H}[\,\mathcal{B}(b_t, u, o)\,]\big].$$

This is **exactly the BALD acquisition function with the answer $a^\star$ substituted for model parameters $\theta$**, and its use as a retrieval objective is established (ITAL; InfoGain-RAG). We claim no novelty for it. The trajectory objective telescopes to total uncertainty reduction under a budget,

$$\max_\pi\ \mathbb{E}_\pi\Big[\textstyle\sum_t \big(\mathbb{H}[b_t] - \mathbb{H}[b_{t+1}]\big)\Big] = \mathbb{E}_\pi\big[\mathbb{H}[b_0] - \mathbb{H}[b_H]\big],$$

which is sequential Bayesian experimental design (DAD) and yields a value-of-information **stopping rule** (stop when the best marginal EIG falls below retrieval cost), as in Stop-RAG.

For free-text answers $\mathbb{H}[b_t]$ is estimated by **semantic entropy** (sample, cluster by entailment, take entropy over clusters; Kuhn et al. 2023); set-level EIG uses greedy conditional gain to kill redundancy (BatchBALD) — so "diversity" falls *out of* the objective rather than being bolted on.

### 4.3 The Bridge Score — the contribution

Define the **answer-frontier potential** $\rho_t : V \to [0,1]$:

$$\rho_t(v) = \underbrace{r(v \mid q)}_{\text{relevance to goal}} \cdot \underbrace{\big(1 - \mathrm{cov}_t(v)\big)}_{\text{not-yet-explored}}, \qquad \mathrm{cov}_t(v) = \max_{u \in V_t} w_{uv}.$$

The Bridge Score of a candidate $v \notin V_t$ is a **conditional, frontier-weighted flow betweenness across the explored cut**. With $\sigma(a,z)$ the number of shortest paths $a \rightsquigarrow z$ and $\sigma(a,z \mid v)$ those through $v$:

$$\boxed{\ \mathrm{BS}_t(v) = \sum_{a \in V_t} \sum_{z \in V \setminus V_t} \rho_t(z)\, \pi_t(a)\, \frac{\sigma(a, z \mid v)}{\sigma(a, z)}\ }$$

where $\pi_t$ weights explored anchors by recency/relevance. The reductions make the relationship to prior art *precise and falsifiable*:

- $V_t = V$, sinks $= V$, $\rho_t \equiv 1$, $\pi_t \equiv 1$ recovers **Freeman betweenness** (1977) exactly — the unconditional, query-free, history-free special case.
- The through-edge ratio gives the demand-weighted analogue of **edge betweenness** (Girvan–Newman 2002).
- It is **not Hwang's bridging coefficient** (2008): that is query-free, history-free, on the static topology. $\mathrm{BS}_t$ is conditioned on $V_t$ (history-aware) and weighted by $\rho_t$ (query- and frontier-aware). Hwang scores how bridge-y a node is *in the fixed graph*; $\mathrm{BS}_t$ scores how much a node bridges *the agent's current memory* to *the answer's currently-unreachable region*.

### 4.4 Tractable surrogate and learnable head

Exact $\mathrm{BS}_t$ is $O(|V|\,|E|)$. Replace shortest-path flow with a diffusion surrogate (Personalized PageRank seeded from the *explored* set $V_t$ — whereas HippoRAG seeds from query entities; that is the history-aware twist). In pure embedding terms with kernel $\kappa(x,y) = e^{-\lVert x-y\rVert^2 / 2\tau}$:

$$\widehat{\mathrm{BS}}_t(v) \propto \underbrace{\Big(1 - \max_{u \in V_t} \kappa(x_v, x_u)\Big)}_{\text{novelty vs. memory}} \cdot \underbrace{\mathbb{E}_{z \sim \rho_t}[\kappa(x_v, x_z)]}_{\text{frontier proximity}} \cdot \underbrace{\mathbb{H}(\kappa(x_v, \cdot))}_{\text{connective breadth}}.$$

**All three factors are load-bearing.** Drop *novelty* → a relevance reranker. Drop *frontier proximity* → a query-free curiosity bonus. Drop *breadth* → an MMR-style diversity term. The product, conditioned on $V_t$ and weighted by $\rho_t$, is the object with no verbatim antecedent.

The learnable head is trained on a **frontier-expansion target — not answer-correctness**:

$$y_t(v) = \big|\mathcal{R}(V_t \cup \{v\}) \cap A\big| - \big|\mathcal{R}(V_t) \cap A\big|,$$

where $A$ is the gold supporting-fact set (annotated in 2WikiMultiHopQA, MuSiQue) and $\mathcal{R}(\cdot)$ is the $\tau$-hop reachable set. A node that is *not itself gold evidence* but **unlocks previously-unreachable gold evidence** gets high $y_t$ here and near-zero under answer-relevance training — the distinction from HopRAG's "helpfulness" and from outcome value functions (Stop-RAG, Search-R1). The GNN-plus-intrinsic-reward machinery itself is pre-empted (Patankar et al. 2023); the only new thing is that the reward is *query-conditioned reachable-answer-frontier gain over the explored set*, not query-free topological surprise.

Total reward: $R = \mathrm{EIG} + \beta\cdot\mathrm{BS} - \lambda\cdot\mathrm{cost}$. The v1 "policy taxonomy" is recovered as regions of $(\lambda, \beta, \text{action-set structure})$ and carries no independent formal content.

---

## 5. The second artifact: a benchmark that credits detours

No existing multi-hop benchmark rewards a retrieval that is *off-path but enabling*; off-path passages are treated as distractor noise. STP proposes a **frozen, pre-registered bridge-jump labeling** layered onto existing datasets:

> Walk each gold chain hop-by-hop. At hop $i$, retrieve top-$K$ from the accumulated context $C_{<i}$ and check whether gold passage $i$ is present, for $K \in \{5, 10, 20\}$. A **bridge hop** is a gold hop *absent* from top-$K$; a **bridge-jump item** has $\geq 1$ bridge hop.

This split is defined by a *baseline encoder's* top-$K$ failure — not by STP — and is published before any system runs. It is what makes the central hypothesis falsifiable, and it is reusable independently of STP.

---

## 6. How we will know if it is real

The whole point of v2 is that the contribution is **falsifiable**. Pre-registered hypotheses:

- **H1 (per-cost bridge win).** On bridge-jump items at a fixed token budget, full STP beats the best baseline *including HopRAG* by ≥3 answer-F1 at equal-or-lower cost (one-sided paired bootstrap, $\alpha = 0.05$, on ≥2 of 3 primary datasets).
- **H2 (mechanism).** STP's recall of the across-low-similarity gold passage exceeds every baseline, with the gap **larger on bridge hops than on non-bridge hops**.
- **H3 (specificity — the discriminating clause).** On **non-bridge** items, STP does **not** beat iterative-RAG by more than 1 F1 at matched cost. A real win is *selective*.

**Datasets.** Primary: MuSiQue, 2WikiMultiHopQA, FRAMES. Stress: Bamboogle. Non-bridge control (H3 only): HotpotQA-distractor.

**Baselines** (shared encoder/corpus/reader/budget): single-shot top-K, cost-matched top-K ceiling, IRCoT, FLARE, Self-Ask, Adaptive-RAG, GraphRAG, **HopRAG (the real adversary)**, IterDRAG — **plus the critical ablation B9 = STP minus Bridge Score (EIG + memory only)**. If B9 matches full STP on bridge items, the contribution is empirically inert.

**Cost rule (pre-registered, adversarial to STP by design):** every LLM call used to estimate a candidate's EIG is charged at full price. There is no free EIG. The headline result is the **F1-versus-token-budget Pareto frontier on bridge items**.

**The claim is FALSIFIED if any of:** the bridge win vanishes once EIG cost is charged (it was just inference scaling); STP ≈ HopRAG at matched cost (Bridge Score adds nothing); B9 ≈ full STP (Bridge Score inert); STP wins *uniformly* incl. non-bridge (generic compute, H3 violated).

**Pilot gate (days, NumPy/NetworkX, no LLM/GPU).** A semantic-trajectory simulator over a stochastic-block-model concept graph with *planted* bridge edges, node2vec embeddings, gold chains forced to cross ≥1 bridge, and **oracle** EIG + Bridge Score. Predicted signature: cosine-greedy collapses, EIG-greedy degrades (no drive to *leave* an exhausted cluster), STP holds — *and the gap appears only when bridges exist*. **If oracle-EIG-greedy matches oracle-STP, the program is dead before any LLM spend.**

---

## 7. Related work

The premise — retrieval should be sequential, history-aware, and driven by information need rather than raw similarity — is the **mainstream position**, not a frontier claim. STP starts from this consensus.

**Iterative / active / adaptive RAG.** FLARE ([2305.06983](https://arxiv.org/abs/2305.06983)), IRCoT ([2212.10509](https://arxiv.org/abs/2212.10509)), Self-Ask ([2210.03350](https://arxiv.org/abs/2210.03350)), ITER-RETGEN ([2305.15294](https://arxiv.org/abs/2305.15294)), Self-RAG ([2310.11511](https://arxiv.org/abs/2310.11511)), Adaptive-RAG ([2403.14403](https://arxiv.org/abs/2403.14403)), DRAGIN ([2403.10081](https://arxiv.org/abs/2403.10081)), IterDRAG ([2410.04343](https://arxiv.org/abs/2410.04343)), Stop-RAG ([2510.14337](https://arxiv.org/abs/2510.14337)).

**Agentic / RL search agents.** ReAct ([2210.03629](https://arxiv.org/abs/2210.03629)), WebGPT ([2112.09332](https://arxiv.org/abs/2112.09332)), Search-o1 ([2501.05366](https://arxiv.org/abs/2501.05366)), Search-R1 ([2503.09516](https://arxiv.org/abs/2503.09516)), R1-Searcher ([2503.05592](https://arxiv.org/abs/2503.05592)), ReSearch ([2503.19470](https://arxiv.org/abs/2503.19470)), DeepResearcher ([2504.03160](https://arxiv.org/abs/2504.03160)), RL-agentic-search survey ([2510.16724](https://arxiv.org/abs/2510.16724)).

**Tree / beam / MCTS over retrieval+reasoning.** Tree of Thoughts ([2305.10601](https://arxiv.org/abs/2305.10601)), RAP ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.507/)), LATS ([2310.04406](https://arxiv.org/abs/2310.04406)), MCTS-RAG ([2503.20757](https://arxiv.org/abs/2503.20757)), PropRAG ([2504.18070](https://arxiv.org/abs/2504.18070)).

**Graph RAG / multi-hop KG.** Think-on-Graph ([2307.07697](https://arxiv.org/abs/2307.07697)) & ToG-2 ([2407.10805](https://arxiv.org/abs/2407.10805)), HippoRAG ([2405.14831](https://arxiv.org/abs/2405.14831)), GraphRAG ([2404.16130](https://arxiv.org/abs/2404.16130)), PathRAG ([2502.14902](https://arxiv.org/abs/2502.14902)), LightRAG ([2410.05779](https://arxiv.org/abs/2410.05779)), **HopRAG ([2502.12442](https://arxiv.org/abs/2502.12442)) — the nearest competitor.**

**RL/MDP retrieval & KG path-walking.** Nogueira & Cho ([D17-1061](https://aclanthology.org/D17-1061/)), DeepPath ([1707.06690](https://arxiv.org/abs/1707.06690)), MINERVA ([1711.05851](https://arxiv.org/abs/1711.05851)), SR ([2202.13296](https://arxiv.org/abs/2202.13296)), Multi-step Retriever-Reader ([1905.05733](https://arxiv.org/abs/1905.05733)), Learning to Retrieve Reasoning Paths ([ICLR 2020](https://openreview.net/forum?id=SJgVHkrYDH)), RAG-Gym ([2502.13957](https://arxiv.org/abs/2502.13957)).

**Active learning / EIG.** BALD ([1112.5745](https://arxiv.org/abs/1112.5745)), BatchBALD ([1906.08158](https://arxiv.org/abs/1906.08158)), DAD ([2103.02438](https://arxiv.org/abs/2103.02438)), ITAL ([1809.02337](https://arxiv.org/abs/1809.02337)), InfoGain-RAG ([2509.12765](https://arxiv.org/abs/2509.12765)), UoT ([2402.03271](https://arxiv.org/abs/2402.03271)), semantic entropy ([Kuhn et al. 2023](https://openreview.net/forum?id=VD-AYtP0dve)).

**Network science of bridges.** Freeman betweenness ([1977](https://www.jstor.org/stable/3033543)), Granovetter weak ties ([1973](https://www.jstor.org/stable/2776392)), Burt structural holes (1992), Gould–Fernandez brokerage ([1989](https://www.jstor.org/stable/270949)), Girvan–Newman edge betweenness ([2002](https://www.pnas.org/doi/10.1073/pnas.122653799)), Hwang bridging centrality ([KDD 2008](https://dl.acm.org/doi/10.1145/1401890.1401934)), Swanson literature-based discovery (1986), Patankar graph-curiosity GNN ([2307.04962](https://arxiv.org/abs/2307.04962)).

**Embedding-space geometry (the substrate evidence).** Robinson et al., "Token Embeddings Violate the Manifold Hypothesis" ([2504.01002](https://arxiv.org/abs/2504.01002)), Shao et al. ([CVPR-W 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Shao_The_Riemannian_Geometry_CVPR_2018_paper.pdf)), HyDE ([2212.10496](https://arxiv.org/abs/2212.10496)), HNSW ([1603.09320](https://arxiv.org/abs/1603.09320)), MA-DPR ([2509.13562](https://arxiv.org/abs/2509.13562)), BeamDR ([2104.05883](https://arxiv.org/abs/2104.05883)).

**Multi-hop benchmarks.** HotpotQA ([1809.09600](https://arxiv.org/abs/1809.09600)), 2WikiMultiHopQA ([COLING 2020](https://aclanthology.org/2020.coling-main.580/)), MuSiQue ([2108.00573](https://arxiv.org/abs/2108.00573)), Bamboogle ([2210.03350](https://arxiv.org/abs/2210.03350)), StrategyQA ([2101.02235](https://arxiv.org/abs/2101.02235)), FanOutQA ([2402.14116](https://arxiv.org/abs/2402.14116)), FRAMES ([2409.12941](https://arxiv.org/abs/2409.12941)).

---

## 8. Limitations (stated, not hidden)

- **Most STP mechanisms are pre-empted, and we concede them.** Sequential trajectories, history-aware state, EIG-as-objective, controlled jumps, retrieval-as-MDP, the policy taxonomy, and continuous-space operation are each instantiated by named prior work, several predating this by up to eight years. The spatial vocabulary ("trajectory," "exploration memory," "semantic jump," "bridge") is largely a re-labeling.
- **The contribution is narrow and may be inert.** It is one functional; its machinery (betweenness, PPR, GNN intrinsic rewards, EIG) is entirely borrowed. The experiment is built so the EIG-only and BS-relevance-only ablations can falsify it.
- **Continuous navigation is dropped, not solved.** Token embeddings are provably not a manifold; STP says nothing about smooth semantic navigation.
- **EIG is expensive and the cost accounting is adversarial by design.** Without an amortized surrogate, STP may simply be priced out of the per-cost frontier it must dominate. The honest outcome may be that EIG-as-ranking is not competitive at retrieval latency.
- **Benchmark/supervision limits.** Bridge labeling depends on encoder and $K$; reachability targets depend on imperfect gold-evidence annotation.

**Net position:** STP is an integrative synthesis of established iterative/agentic-RAG and RL graph-exploration machinery for the multi-hop setting, **plus one incremental functional — the frontier-weighted Bridge Score — whose value is an empirical question this document poses but does not yet answer.** It is not a new retrieval paradigm, and is not presented as one.

---

## 9. Roadmap

1. **This whitepaper (done).** Honest positioning; concede the pre-empted claims; stake the flag only on the Bridge Score functional and the detour-crediting evaluation.
2. **Pilot (days).** Build the synthetic semantic-trajectory simulator (§6) and run the planted-bridge sweep under oracle EIG/Bridge Score. This *gates* the whole program before any LLM spend.
3. **Defensible short paper.** The §4 formalism + §7 related-work table + pilot results as existence proof, positioned strictly against HopRAG, Stop-RAG, BALD/ITAL.
4. **Full evaluation (if the pilot passes).** Implement on the discrete substrate; run §6 against all baselines incl. HopRAG and B9 with EIG cost charged. Publishable **iff** the win is selective, per-cost, and ablation-localized to the frontier term. **If falsified, report the negative result** — "a frontier Bridge Score does not beat HopRAG at matched cost" is itself a contribution.

---

## Repository contents

| Path | What it is |
|---|---|
| [`README.md`](README.md) | This whitepaper (v2) |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | Full "how to make STP serious" report: landscape, claim-by-claim novelty audit, formal core, experiment design, reviewer objections + rebuttals |
| [`docs/PAPER_v2_SKELETON.md`](docs/PAPER_v2_SKELETON.md) | Peer-review-grade paper skeleton (abstract → related work → method → experiments → limitations) for the escalation path |
| [`docs/README_v1_manifesto.md`](docs/README_v1_manifesto.md) | The original v1 manifesto, preserved for the record |

---

*Honesty note: the literature audit and formalization backing this document were produced with assistance from an AI research workflow; every citation above was verified to be a real, existing publication. If you find a citation that does not check out, please open an issue — that is exactly the kind of error this rewrite exists to eliminate.*
