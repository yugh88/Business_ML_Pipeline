# 17 — V3 strategy review (dossier + our evidence) and frozen V3 design

Inputs: `amazon_ml_challenge_top_repos_research_dossier-1.md`, full-data v2 validation (`analysis/aws_v2/`), dev experiments
(`analysis/out/72–76`). Rule: only ideas verified on OUR data enter V3.

## Baseline (v2, full-partition, frozen decision, fold 7 selection)
Holdout macro-F0.5 0.9788 / 0.9788 / 0.9790 (folds 0/8/9), P 0.994, R 0.956, singletons 0.957, 4.46 cand/S1 (train),
5.02 cand/S1 (test). Submission secured in `output/` (official validator PASS).
Loss decomposition (holdout): 1.3% positives lost before the model (blocking+pruning), 3.2% rejected by the model
(52% of those = empty pool address, mostly undecidable when >1 S1 share the name). Oracle: fix all in-candidate FN
0.9900, remove all FP 0.9850. p2 is calibrated (bin-wise observed rate ≈ mean p).

## Dossier ideas vs our evidence
| idea | our status | evidence | V3 |
|---|---|---|---|
| reverse target→S1 retrieval | core since v2 | recall@10 0.977–0.980 (07) | keep |
| multi-channel retrieval + rank/score features | 4 channels, scores+ranks+block flag | v2 | keep; agreement count = deterministic fn of existing cols → redundant for GBDT |
| sibling / collective features | stage-2 coll + consensus | +0.0005 full data | keep |
| house-number geometry | **new**: change type (truncation/digit/other) + log numeric diff | dev +0.0024 (0.9707→0.9731); truncation 16–22% positive vs 0.5–3.7% other | **add** |
| injected/missing name tokens | **new**: extra/missing counts, generic-only flags, repeated token | dev +0.0055 (0.9731→0.9786), FP 320→172 | **add** |
| address-token diffs | tested | dev −0.0003 | reject |
| XGB + LGBM blend | measured in-run | local test aborted by laptop memory guard | **in V3: LGB vs XGB vs mean, chosen on fold 7** |
| margin / exclusivity / ambiguity abstention | tested full data | ≤ +0.0001 over threshold; G (expected-F, includes exclusivity) +0.0005 | keep G only |
| singleton gate | G allows k=0 | singleton acc 0.957 | keep |
| hard-negative loop | training set = retrieved candidates only (all hard); stage 2 on hardest ~10/S1 | — | covered |
| per-segment thresholds, calibration | tested | no gain; already calibrated | reject |
| multilingual embeddings | cross-script errors already rare (0.05% zero overlap after dict) | — | reject |
| stage-3, bigger trees | tested | 0.9706 / 0.9708 (no gain) | reject |

Dossier headline scores are not comparable: the 0.9994 ablation reports candidate recall 1.0000 (true pairs guaranteed
in the evaluated set — not a real blocking pipeline); prem970's figures conflict (0.99813 vs 0.99684). The closest
comparable design (Ayan: reverse retrieval + collective + expected-F) reports 0.98703 — consistent with our trajectory.
Public LB #1 per dossier snapshot: 0.98696.

## Frozen V3 design
v2 retrieval/features reused (S3 copy) → **augment** stage adds 7 features (`featx.py`) → stage 1 (3-way cross-fit, all
features) → collective + consensus → stage 2: LightGBM and XGBoost on identical features (+ mean blend) → evaluate A–G
per variant on fold 7 → local joint (τ, policy) tuning with the 0.0001-F0.5 tolerance favoring smaller candidate sets →
final TSVs + official validation. Instance: on-demand r6i.2xlarge (8 vCPU/64 GB; 12 on-demand vCPU free while v1 runs).

Expected bottleneck: empty-address shared-name copies (irreducible) and near-twin number/name perturbations; V3 targets
the latter. Expected full-data holdout ≈ 0.984–0.987 if dev gains transfer (dev +0.008 over v2 feature set).

---

## UPDATE — hidden leaderboard: v2 scored **0.936** (local holdout 0.979). LB top now ≈ 0.98997.

### Label-free train/test shift analysis (analysis/out/80–83)
| check | train | test | reading |
|---|---|---|---|
| pool records per S1 | 4.68 | 5.75 (all countries) | +1.1 pool records per S1 |
| S1 with an ambiguous candidate (p2 0.2–0.8) | 16–19% | 33–35% (US, India, France alike) | ~2× — NOT France-specific |
| candidates with p2 ≥ 0.5 per S1 | 3.38 (truth 3.46) | 3.70–3.80 | ~0.3 extra confident candidates per S1 |
| pool predicted/true matched share | 73.2% (truth) | 62.9% (v2 pred) | test ~40% unmatched vs 27% |
| EM prior-shift estimate (assumes calibration holds) | 0.766 | US 0.789, India 0.749, France 0.679 | calibration breaks under this shift → EM uninformative |
| **unmatched records with an unmatched look-alike sibling** (same 1st name word + house no. + street) | **18%** | **63.5%** | test distractors come in CLUSTERS |
| France sample (manual) | — | predictions mostly correct; empty-address copies rejected | France alone cannot explain 0.043 |

**Diagnosis:** test pool contains *orphan clusters* — several copies of businesses that are absent from test S1. In train,
distractors are isolated records. v2's sibling-consensus / collective features learned "agreement with other copies ⇒
match"; an orphan cluster agrees with itself, so when it resembles a present S1 (chain name, same building, neighbouring
number) v2 accepts it ⇒ false merges on ~a third of S1 (≈ consistent with 0.979 → 0.936). Our holdout never contained
orphans, so it could not see this. Other shifts (France formats, abbreviations, scripts, missingness) exist but are small
(analysis/01, 12 F5) and do not explain a uniform, all-country drop.

### V3 design change (evidence-backed)
* **New stage `orphanize`**: drop 16% of train S1 entities (reproducible hash) but keep their S2/S3 copies as labeled
  negatives → train unmatched share ≈ 40% (test-like); recompute every S1-set-dependent feature (reverse rank, best /
  2nd-best reverse score, margins). Training AND validation run on this orphan-matched train, so the holdout now measures
  the test-like regime (and the chosen threshold/policy is calibrated to it).
* Keep: augment (+7 features), stage 1/collective/stage 2 (LGB vs XGB vs blend), policies A–G, local τ tuning.
* Expected: orphan-matched holdout lower than 0.979 (it is harder) but a much smaller LB gap. Success criterion: LB ≫ 0.936.
