# SENTINEL — Architectural Decomposition & System Design (Pre-Implementation)

**Self-Evolving Neural Threat Intelligence through Neuroadaptive Evolutionary Learning**
Design document v0.1 · Defensive-security research · derived from *SENTINEL Draft v3.0 (June 2026)*

> **Status:** Design only. No implementation code is specified to be *run* here; this document
> fixes the architecture, resolves the paper's under-specified details, and justifies every
> consequential choice at the level a top-tier reviewer (IEEE S&P / USENIX / CCS / NDSS /
> NeurIPS / ICLR) would demand before accepting the methodology.

---

## 0. Scope, ethics, and non-negotiable invariants

SENTINEL is **strictly defensive**. The following invariants are *architectural constraints*,
not policies — they are enforced by type system, capability gating, and tests, so that a
violation is a build/test failure rather than a runtime hope.

| Invariant | Enforcement mechanism |
|---|---|
| **I1 — No attack synthesis.** The system never generates, mutates, or optimizes attacks. | Adversarial inputs only enter through a read-only `CorpusLoader`. There is no code path from any learned component to an attack-emitting output. Evolution operates on the *defense genome* type only. |
| **I2 — Evolution confined to defensive modules.** | The `Genome` type can only contain `DefensiveModule` instances. Objectives, oversight, and the base agent are immutable references not present in the mutable genome. Mutation operators are a closed set (`AddValidator`, `TuneValidator`, `AddDetector`, `TuneDetector`, `AddPolicyConstraint`, `Reroute`). |
| **I3 — Human gate before retention.** | The evolution loop emits *proposals*; a proposal is `PENDING` until a `HumanGateDecision` record is attached. Deployment reads only `APPROVED` genomes. CI test asserts no autonomous transition `PENDING → DEPLOYED`. |
| **I4 — Oversight immutable.** | The Sentinel layer, the human gate, and the fitness evaluator are outside the genome and cannot be referenced as mutation targets (enforced by `MutationTarget` enum + registry whitelist). |
| **I5 — Probes are public-corpus only.** | `CorpusLoader` records a SHA-256 provenance hash per probe and rejects any probe lacking a registered source. |

These map directly to the paper's Risk Register rows 1–2 ("Dual-use drift", "Self-modification
escapes defensive subsystem"), which it flags non-negotiable.

---

## 1. Architecture Review

### 1.1 Layered decomposition

SENTINEL = FGAE substrate (3 layers) + a **Sentinel security layer** that occupies FGAE's
critic slot, plus a meta-defense learner and a defensive-evolution engine. The paper's Table 3
gives four layers; we decompose into nine concrete subsystems for implementation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR  (async pipeline; per-cycle budget ≈ 6,780 tok)              │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────┐   the substrate (pluggable; FGAE reference impl)
│ BASE AGENT    │   served quantized LLM via ModelBackend abstraction
│ (Base Layer)  │   → produces a task answer / tool-call plan
└───────────────┘
   │  every input, retrieved doc, tool output, memory write
   ▼
┌───────────────────────────────────────────────────────────────────────┐
│ SENTINEL LAYER  (PRIMARY)   "is this a threat, and of what kind?"       │
│  Stage 1  RuleScreen        — fast, cheap, high-recall structural gate  │
│  Stage 2  AnomalyScreen     — embedding-space OOD / novelty             │
│  Stage 3  NeuralClassifier  — OWASP-class head + calibrated confidence  │
│  Stage 4  SignatureExtractor— behavioral signature vector (6 axes)      │
└───────────────────────────────────────────────────────────────────────┘
   │  ThreatEvent{class, conf, signature, context, source}
   ▼
┌───────────────┐        ┌───────────────────────────────────────────────┐
│ THREAT GRAPH  │◀──────▶│ META-DEFENSE LAYER                             │
│ (memory)      │  query │  threat repr → best countermeasure            │
│ nodes/edges   │        │  contextual bandit / Thompson sampling        │
└───────────────┘        └───────────────────────────────────────────────┘
   │  history signals (recurrence, bypass, migration)
   ▼
┌───────────────────────────────────────────────────────────────────────┐
│ EVOLUTION ENGINE  (NOVELTY)  defense-genome search (NSGA-II + QD)       │
│  propose mutation → evaluate on held-out probes → score fitness         │
│  → HUMAN GATE → retain only if Pareto-improving & utility preserved     │
└───────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ EVAL ENGINE   │   │ METRICS ENGINE│   │ VIZ / STATS   │
│ adversarial + │   │ 4-dimension   │   │ figures +     │
│ clean grids   │   │ metric catalog│   │ ANOVA/bootstrap│
└───────────────┘   └───────────────┘   └───────────────┘
```

**Subsystem responsibilities (one line each):**

1. **Orchestrator** — async per-cycle pipeline; enforces token budget; wires layers; deterministic seeding.
2. **Base Agent** — task solving via `ModelBackend`; *unchanged from FGAE*; never trusts input as instruction.
3. **Sentinel Layer** — 4-stage cascade producing `ThreatEvent`s (the primary contribution).
4. **Threat Graph** — typed property graph: structured threat/defense memory + similarity/recurrence/migration queries.
5. **Meta-Defense** — learns `threat → countermeasure` value; selects highest-EV defense (no hardcoded map).
6. **Evolution Engine** — multi-objective search over the defensive genome; human-gated.
7. **Eval Engine** — runs the 5×3×10 grid (defenses × models × threats) + clean grid; produces raw outcomes.
8. **Metrics Engine** — computes the full Tier-1/2/3 catalog across 4 research dimensions.
9. **Stats/Viz** — bootstrap CIs, two-way ANOVA, Tukey HSD, multiple-comparison correction, publication figures.

### 1.2 Data flow (per cycle)

```
RawInput (task | retrieved doc | tool output | memory write)
  └─▶ Sentinel.screen(input, ctx)
        Stage1 RuleScreen ──(structural flags)──┐
        Stage2 AnomalyScreen ──(novelty score)──┤
        Stage3 NeuralClassifier ──(class, p)────┤──▶ ThreatEvent | Benign
        Stage4 SignatureExtractor ──(sig ∈ R^d)─┘
  ThreatEvent ─▶ ThreatGraph.add(event)            (log: class, sig, ctx, source)
  ThreatEvent ─▶ MetaDefense.select(event, graph)  ─▶ DefensePlan
  DefensePlan ─▶ BaseAgent.resolve_under_defense(input, plan) ─▶ DefendedOutput
  Outcome{blocked?, leaked?, drifted?} ─▶ ThreatGraph.record_outcome(edge: defeated_by|bypassed)
  (periodically) EvolutionEngine.maybe_propose(graph) ─▶ Proposal ─▶ HumanGate ─▶ Genome'
```

The token envelope mirrors paper Table 5: Base solve (600/800) → Sentinel screen (1,400/300)
→ Countermeasure select (1,700/80) → Defended re-solve (900/1,000) ≈ **6,780 tok/cycle**.
The Sentinel screen *substitutes for* FGAE's reasoning critic, so per-cycle cost is unchanged;
added cost comes only from running each task under both clean and adversarial conditions.

### 1.3 State machines

**(a) Attack detection** (per input through the cascade):
```
NEW ─▶ RULE_SCREEN ─[clean & low-novelty]─▶ BENIGN(accept)
   RULE_SCREEN ─[flagged]─▶ ANOMALY_SCREEN
   ANOMALY_SCREEN ─[in-distribution & no rule flag]─▶ BENIGN(accept)
   ANOMALY_SCREEN ─[novel | flagged]─▶ CLASSIFY
   CLASSIFY ─[p < τ_low]─▶ UNCERTAIN ─▶ (conservative defense; log for review)
   CLASSIFY ─[p ≥ τ_high]─▶ THREAT(class k) ─▶ SIGNATURE ─▶ EMIT ThreatEvent
   any ─▶ ERROR ─▶ FAIL_CLOSED (treat as threat, log)   # security default
```
Design rule: **fail-closed**. Ambiguity escalates toward defense, never toward acceptance —
a missed attack is the costliest failure (paper Tier-1 rationale for recall).

**(b) Defense selection** (meta-defense):
```
THREAT_EVENT ─▶ LOOKUP(graph: prior outcomes for class+signature-neighborhood)
  ─[known effective defense, high posterior]─▶ EXPLOIT(select best)
  ─[uncertain / cold-start]─▶ EXPLORE(Thompson sample over defense arms)
  SELECTED ─▶ APPLY ─▶ OBSERVE(outcome) ─▶ UPDATE(posterior) ─▶ DONE
```

**(c) Architecture evolution** (genome lifecycle):
```
STABLE ─[graph shows module persistently bypassed/missing]─▶ PROPOSE(mutation)
  PROPOSE ─▶ EVALUATE(held-out probes + clean tasks)
  EVALUATE ─[ΔPareto not improving OR utility drop>5%]─▶ REJECT ─▶ STABLE
  EVALUATE ─[Pareto-improving & utility ok]─▶ CANDIDATE
  CANDIDATE ─▶ HUMAN_GATE
     HUMAN_GATE ─[reject]─▶ ARCHIVE ─▶ STABLE
     HUMAN_GATE ─[approve]─▶ DEPLOY ─▶ STABLE'(new genome)
```
Invariant: there is **no edge** `CANDIDATE → DEPLOY` that bypasses `HUMAN_GATE` (I3).

### 1.4 Threat lifecycle (end-to-end, paper §1.2 mapping)

```
Input → Detection(S1–S3) → Classification(class+conf) → Signature(S4)
      → Logging(ThreatGraph node + edges) → Defense Selection(meta-defense)
      → Evaluation(outcome: blocked/leaked/drifted; clean-task utility)
      → Evolution(if pattern persists: propose defensive module)
      → Human Gate → Deployment(updated genome) → [loop back; behavior re-measured]
```
The lifecycle is *the same* learn-from-signal loop FGAE applies to reasoning failures; the only
substitution is the signal type (adversarial threat vs. reasoning failure).

---

## 2. Research Gap Analysis

The proposal is a *research plan*, so many implementation-level decisions are unspecified. For
each gap: the likely intent, the chosen implementation, and the scientific justification.

| # | Under-specified in paper | Inferred intent | Decision | Justification / rejected alternatives |
|---|---|---|---|---|
| G1 | **Classifier architecture** ("neural threat classifier") | A learned, paraphrase-robust, per-class detector that does not train the base LLM | **Frozen sentence-embedding encoder + lightweight calibrated heads** (logistic / small MLP) per OWASP class, one-vs-rest, with temperature-scaled probabilities. | Paper mandates *no base-model training* (§5 cost) and *behavioral, not keyword* detection (§2). Fine-tuning a 14B is out of budget. Embedding+head is cheap, swappable, and gives calibrated confidence for the gate. *Rejected:* full fine-tune (budget/Invariant against modifying base), pure LLM-as-judge (cost + nondeterminism; we keep it as an optional Stage-3 fallback head, not the primary). |
| G2 | **Embedding/anomaly method** | Detect novel attacks unlike anything seen | **Mahalanobis distance in encoder space + Isolation Forest ensemble** over the benign+seen-threat manifold; score = max-normalized novelty. | Two complementary detectors: Mahalanobis (parametric, good for unimodal class manifolds) and IsolationForest (non-parametric, multimodal). The paper's Threat Novelty Score explicitly lists both. *Rejected:* single kNN distance (sensitive to corpus density); deep OOD nets (data-hungry, unjustifiable at this scale). |
| G3 | **Signature vector definition** | A formal behavioral signature, not a string | **Hybrid 6-axis interpretable vector ⊕ learned residual embedding** (see §3.2). Axes: semantic-intent, objective-drift, privilege-escalation, instruction-hierarchy-violation, trust-boundary-crossing, tool-abuse. | Interpretable axes are required to *study threat behavior* (the stated primary concern) and to make migration/drift analysis meaningful; the learned residual preserves discriminative power. *Rejected:* pure learned embedding (opaque — undermines the behavior-science contribution); pure hand-features (brittle, low recall). |
| G4 | **Threat graph storage** | A queryable, scalable threat/defense memory | **In-process typed property graph (NetworkX-compatible API) with a pluggable persistent backend (SQLite+JSON now; Neo4j/Kùzu adapter interface).** Vector index (FAISS/hnswlib) for signature similarity. | Scale here is ≤ ~10⁵ nodes (40k cycles); an embedded graph + ANN index is sufficient and reproducible. The adapter interface satisfies "design for scalability" without premature DB ops. *Rejected:* mandatory Neo4j (repro/CI friction for a <$10 study). |
| G5 | **Meta-defense learning algorithm** | Learn threat→countermeasure without hardcoding | **Contextual multi-armed bandit (LinUCB / Thompson sampling)** over a fixed library of defensive *strategies*, context = signature vector. | Online, sample-efficient, naturally balances explore/exploit, gives per-strategy posteriors used by evolution. Matches the paper's "highest-expected-value defense". *Rejected:* full RL (sample-hungry, unstable at 40k cycles); static lookup table (explicitly disallowed). |
| G6 | **Evolution algorithm** | Search defensive architectures | **NSGA-II (multi-objective GA) as the spine, with a MAP-Elites quality-diversity archive** for behavioral coverage; bandit-derived priors seed mutation. | Defense design is inherently multi-objective (security vs. utility vs. latency/cost) → Pareto methods dominate scalarization; QD prevents collapse to a single defense and directly yields the "which defenses emerge / reuse" science. See §3.5 for the full mathematical comparison. *Rejected:* plain GA/weighted-sum (loses Pareto front), Bayesian opt (poor in combinatorial/structured genome spaces), neuroevolution (no weights to evolve here — we evolve *composition*, not networks). |
| G7 | **Fitness formulation** | A rigorous, multi-objective security fitness | **Constrained multi-objective:** maximize vector (1−ASR, recall, precision) subject to FPR ≤ ε, utility-drop ≤ 5%, latency/cost budget; aggregated for reporting via the paper's `SecurityFitness = (1−ASR)/(cost×latency)` as a *scalar reporting metric*, but selection uses Pareto dominance. See §3.6. | Scalarizing first hides tradeoffs reviewers care about; constraints encode the hard utility gate (paper risk row 3). *Rejected:* single weighted scalar as the selection signal (weight choice becomes an attack on the result's validity). |
| G8 | **Transfer-learning strategy** | Cross-threat & cross-model generalization | Cross-threat: leave-one-class-out; defenses learned on seen classes evaluated on held-out class (ΔASR). Cross-model: freeze evolved genome from model A, evaluate on B/C/32B. | Directly operationalizes H3 and the model-agnostic claim with clean train/test separation. |
| G9 | **"Successful attack" operational definition** | Per-class success criterion for ASR | Per-class **programmatic success oracle** (e.g., LLM07 = system-prompt substring/semantic leak; ASI03 = out-of-scope credential/tool invoked in the sandbox; LLM02 = secret regex/canary exfiltrated). Canary-token methodology. | ASR is the flagship metric; it must be *objective and per-class*, not a global judge. Canaries make leakage detection deterministic. *Rejected:* single global LLM judge (nondeterministic, unciteable). |
| G10 | **Multi-turn / context-length / poisoning harnesses** | Beyond single-turn probes | Dedicated harnesses: conversation simulator (2/5/10/20 turns), context padding (1k–64k), retrieval-poisoning injector, persistent-memory poisoner, tool sandbox. | The user brief and reviewer expectations require these robustness studies; single-turn is insufficient for a security venue. |
| G11 | **Statistical design** | Publication-grade validation | Pre-registered hypotheses (H1–H4), two-way ANOVA (model×condition), Tukey HSD, 10k-bootstrap CIs, Cohen's d / Hedges g / η², Holm–Bonferroni + BH correction, power analysis (≥0.8). | Listed in brief; encoded as a reusable `stats` module so *no metric is reported without a test*. |

---

## 3. Per-component design (theory → decision → alternatives → security implications)

### 3.1 Sentinel detection cascade (PRIMARY)

**Theory.** A cascade trades cost for recall by ordering screens cheap→expensive and routing
only suspicious inputs deeper. We bias every stage toward **recall** (fail-closed) because the
asymmetric loss of a missed attack dominates a false positive (paper Tier-1).

- **Stage 1 — RuleScreen.** *Structural, not keyword.* Flags computed from structure: presence
  of imperative verbs *in untrusted spans*, instruction-reframing patterns (role redefinition,
  "ignore/override"), delimiter/format breaks, scope tokens in tool params, canary proximity.
  These are **features**, not a blocklist — they feed later stages and never *alone* accept/reject
  (avoids the paraphrase-evasion failure the paper calls out). Theoretical basis: instruction/data
  channel confusion (Greshake 2023; Perez & Ribeiro 2022).
- **Stage 2 — AnomalyScreen.** Mahalanobis + IsolationForest novelty in encoder space (G2). Catches
  paraphrases and zero-days that route around Stage 1.
- **Stage 3 — NeuralClassifier.** Frozen encoder + calibrated per-class heads (G1). Outputs OWASP
  class + temperature-scaled confidence; the confidence gates downstream action (τ_low/τ_high).
- **Stage 4 — SignatureExtractor.** Emits the behavioral signature (§3.2).

**Security implications.** Cascade ordering means an attacker who evades Stage 1 still faces
anomaly + neural screens; calibration prevents over-confident wrong classes from auto-accepting.
**Rejected:** single monolithic LLM judge (cost, nondeterminism, no calibration).

### 3.2 Behavioral signature (hybrid, 6 interpretable axes + residual)

`signature = [a1..a6] ⊕ z` where each `a_i ∈ [0,1]` is an interpretable behavioral axis and
`z ∈ R^m` is a learned residual embedding (projection of encoder features orthogonalized against
the axes). Axes (paper §2 "behavioral signature the agent learns"):

1. **semantic-intent** distance from benign task intent;
2. **objective-drift** (ASI01) — divergence of inferred goal from original task;
3. **privilege-escalation** (ASI03/LLM06) — requested scope minus granted scope;
4. **instruction-hierarchy-violation** (LLM01) — untrusted content asserting system authority;
5. **trust-boundary-crossing** (LLM04/ASI06) — retrieved/stored content contradicting trusted baseline;
6. **tool-abuse** (ASI02) — out-of-policy tool/params.

**Why hybrid (G3).** The axes make *Attack Migration*, *Signature Drift*, and *Threat Recurrence*
scientifically interpretable (you can say *attacks migrated from hierarchy-violation to tool-abuse*);
the residual keeps classification accuracy competitive with a black-box embedding.

### 3.3 Threat graph

Typed property graph. **Nodes:** `Threat`, `Defense`, `Module`, `Outcome`, `AttackClass`.
**Edges:** `defeated_by`, `resembles` (signature ANN), `migrated_to` (temporal class shift),
`bypassed`, `evolved_into`. **Queries:** signature similarity (ANN), recurrence (per-class
frequency over time), migration (temporal Δ-distribution + KL), defense-reuse (defense→#classes).
Backend pluggable (G4). Vector index for `resembles`.

### 3.4 Meta-defense (contextual bandit)

Arms = defensive strategies {input-sanitization, instruction-data separation, privilege-narrowing,
retrieval-grounding, memory-quarantine, output-validation}. Context = signature vector. Reward =
1 if attack neutralized AND clean-task utility preserved, else 0 (with shaped penalty for utility
loss). LinUCB/Thompson balances explore/exploit; posteriors are persisted to the graph and exported
as priors to the evolution engine (G5). **No hardcoded threat→defense map.**

### 3.5 Evolution engine — algorithm selection (mathematical justification)

We evolve the **defensive composition** (which validators/detectors/policies are present and how
routed), i.e. a structured, mixed discrete/continuous genome with multiple competing objectives.

- **Genetic Algorithm (scalar):** simple, but collapsing objectives to one weighted sum makes the
  result hostage to weight choice and hides the security–utility tradeoff. ✗ as selection signal.
- **Bayesian Optimization:** excellent for low-dim continuous black-box; weak on structured/combinatorial
  genomes and multi-objective fronts without heavy machinery. ✗ as spine.
- **Multi-Armed Bandit:** ideal for *strategy selection* (we use it for meta-defense), but not for
  searching *architectures* (no notion of recombination/structure). ✓ as sub-component, ✗ as spine.
- **Evolution Strategies / Neuroevolution:** designed to evolve *real-valued weight vectors / networks*.
  We are not training networks — there are no weights to evolve. ✗ (category mismatch).
- **Quality-Diversity (MAP-Elites):** maintains an archive of diverse high-performers across a
  behavior space → directly produces "which defenses emerge", reuse, and coverage science. ✓ as archive.
- **NSGA-II (multi-objective GA):** Pareto-ranking + crowding distance gives the full security–utility
  front reviewers want, with O(MN²) non-dominated sorting per generation (M objectives, N pop). ✓ spine.

**Decision (G6):** **NSGA-II spine + MAP-Elites archive**, bandit posteriors as mutation priors.
NSGA-II yields the Pareto front (selection); MAP-Elites yields behavioral diversity (science).
Mutations restricted to the closed defensive set (I2). Every candidate is held-out-evaluated and
human-gated (I3).

### 3.6 Security fitness (constrained multi-objective)

Selection uses **Pareto dominance** over objective vector
`f = (1−ASR, recall, precision)` **subject to constraints**
`FPR ≤ ε_fp`, `clean_utility_drop ≤ 0.05`, `latency ≤ L_max`, `token_cost ≤ C_max`.
For *reporting* (single-number comparisons, page-1 framing) we also compute the paper's
`SecurityFitness = (1−ASR)/(cost×latency)` and `DefenseEfficiency = ΔSecurity/ΔComplexity`.
Constraints encode the hard utility gate (reject any change dropping clean accuracy >5%, paper
risk row 3). **Rejected:** scalar weighted sum as the *selection* signal (G7).

---

## 4. Experimental & statistical framework

- **Conditions (paper Table 8):** Vanilla, Static Filter, Reflection-Defense, Meta-Defense, Full SENTINEL — each adds exactly one mechanism so pairwise diffs attribute protection.
- **Grid:** 5 defenses × 6 models × 10 threat classes × 60 probes, 3 seeds. The roster spans five vendor lineages (Meta, Microsoft, DeepSeek, Mistral, Alibaba), a within-family scale axis (Llama-3.1-8B ↔ Llama-3.3-70B), and a reasoning-vs-instruct contrast at matched 14B scale (Phi-4 ↔ R1-Distill-Qwen-14B). Clean grid reuses GSM8K/MATH-500/HumanEval/MBPP/HotpotQA for utility retention.
- **Metrics (4 dimensions):** Security effectiveness (ASR, ASR-AULC flagship, time-to-hardening, recall, FPR, P/R/F1/macro/weighted); Threat behavior (migration KL + matrix, recurrence curves, signature drift cosine/euclid + cluster evolution, Shannon diversity, novelty via Mahalanobis/IsolationForest); Defense evolution (security fitness, defense efficiency, stability index, convergence rate, leave-one-out module utility, reuse ratio, cross-threat transfer); Utility (security-utility tradeoff, clean-task accuracy).
- **Robustness:** seen/unseen/zero-day, paraphrase (≥100 variants/attack), multi-turn (2/5/10/20), context length (1k–64k), retrieval poisoning, memory poisoning, tool abuse (param injection / scope escalation / chaining) — all in sandbox.
- **Ablations:** Full → −Evolution → −Meta-Defense → −Threat-Graph → −Signatures → −Classifier → rule-only → neural-only.
- **Statistics:** two-way ANOVA (model×condition), Tukey HSD post-hoc, 10k bootstrap CIs, Cohen's d / Hedges g / η², Holm–Bonferroni + Benjamini–Hochberg, power ≥ 0.8 with sample-size derivation. Sensitivity sweeps over thresholds, fitness weights, evolution rate, graph/memory size, embedding dim, corpus size.
- **Reproducibility:** fixed seeds, Hydra config snapshots, experiment manifests, dataset & model hashes, hardware/runtime metadata captured per run.

---

## 5. Repository layout & tech stack

```
SENTINEL/
├─ pyproject.toml            # Python 3.12+, ruff, mypy(strict), pytest
├─ docs/                     # this design + per-module specs + ADRs
├─ conf/                     # Hydra/OmegaConf configs (models, threats, experiments)
├─ src/sentinel/
│  ├─ core/                  # types (pydantic), invariants, seeding, logging
│  ├─ substrate/             # FGAE interface + reference impl (pluggable)
│  ├─ models/                # ModelBackend abstraction: mock | vLLM | openai-compat
│  ├─ sentinel_layer/        # rule_screen, anomaly_screen, classifier, signature
│  ├─ graph/                 # threat graph + backends + ANN index
│  ├─ meta_defense/          # bandit strategy selector + defense library
│  ├─ evolution/             # genome, mutations, nsga2, map_elites, human_gate
│  ├─ defenses/              # concrete DefensiveModule implementations
│  ├─ corpora/               # read-only loaders (OWASP/ATLAS/red-team), provenance
│  ├─ eval/                  # success oracles, grids, robustness harnesses
│  ├─ metrics/               # 4-dimension metric catalog
│  ├─ stats/                 # bootstrap, anova, posthoc, effect size, corrections, power
│  └─ viz/                   # publication figures
├─ tests/                    # unit + integration + invariant/safety tests
└─ experiments/              # manifests, run scripts, results, figures
```
**Stack:** Python 3.12, Pydantic v2 (typed records), OmegaConf/Hydra (config), `structlog`
(structured logging), `asyncio` (pipeline), numpy/scipy/statsmodels/pingouin (stats),
scikit-learn (IsolationForest, calibration), sentence-transformers (frozen encoder),
hnswlib/FAISS (ANN), networkx (+SQLite/Kùzu adapter), matplotlib (figures), pytest + CI, Docker.
**Model-agnostic:** swapping any of the six roster models (Llama-3.1-8B ↔ Phi-4 ↔ DeepSeek-R1-Distill ↔ Mistral-Small-24B ↔ Qwen3-32B ↔ Llama-3.3-70B) is a *config* change only;
there is deliberately **no mock backend** — the GPU-independent logic is covered by the unit suite instead.

---

## 6. Build plan (mapped to the paper's 16-week phases)

| Phase | Paper weeks | Deliverable | Hard gate |
|---|---|---|---|
| P0 Scaffold | (pre) | repo, types, invariants, config, CI, MockBackend | invariant tests green |
| P1 Foundation | 1–2 | substrate iface + ref impl, corpus loaders, probe set | reasoning loop live; probes loaded w/ provenance |
| P2 Sentinel | 3–5 | 4-stage cascade + signatures | detection recall ≥ 0.80; class F1 ≥ 0.75 |
| P3 Meta-Defense | 6–7 | bandit selector + threat graph | first-pass neutralization ≥ 60% |
| P4 Evolution | 8–9 | genome + NSGA-II/QD + human gate | working hardening loop on 1 class |
| P5 Integration | 10–11 | capability-retention harness | clean-task drop < 5% |
| P6 Run | 12–14 | adversarial+clean grids, ablation, transfer | raw results all conditions |
| P7 Analyze | 15 | stats, ASR curves, figures | significance tests; page-1 figure |
| P8 Write | 16 | manuscript scaffold + gated, defensive-only repo | conference draft |

Each phase ends with **self-critique (reviewer mode)** per the brief.

---

## 7. Self-critique (anticipating reviewers trying to reject)

- **"ASR oracle is gameable / subjective."** → Per-class programmatic oracles + canary tokens (G9); publish oracle code.
- **"Evolution gains are noise."** → NSGA-II selection + held-out eval + 3 seeds + bootstrap CIs + ANOVA/Tukey; report Pareto front not a single number.
- **"Detector overfits probe phrasings."** → ≥100 paraphrases/attack, held-out classes, signature-drift test, zero-day set.
- **"FGAE substrate is doing the work, not Sentinel."** → ablation `−Sentinel`/`−classifier`/`rule-only` isolates the security layer's marginal contribution.
- **"Bandit/evolution are over-engineered for 10 classes."** → ablations `−Meta-Defense`/`−Evolution` quantify whether they earn their complexity; if null, publish as the paper's pre-committed diagnostic finding.
- **"Single annotator / no human-gate realism."** → log every gate decision with rationale; report inter-gate consistency.
- **"Cross-model claim is weak with few models."** → six models over five vendor lineages, two-way ANOVA model×condition, a within-family scale contrast (Llama 8B↔70B) separating scale from family, a matched-scale reasoning contrast (Phi-4 ↔ R1-distill), + cross-model transfer matrix.

---

## 8. Open decision requiring your input

One choice materially shapes implementation and isn't resolvable from the paper alone:
**how aggressively to target real GPU inference vs. a CPU-runnable reference.** Default plan:
build fully model-agnostic with a `MockBackend` + deterministic stub encoder so the *entire system,
tests, stats, and figures run on CPU with no API cost*, and add vLLM/OpenAI-compatible adapters so
the real 14B/32B runs drop in via config when you have the A100. This keeps the <$10 / reproducible
claim intact and lets development proceed without GPU access.
