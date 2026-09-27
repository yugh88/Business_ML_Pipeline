# STATE (single source of truth — update after every major stage)

_Last update: 2026-09-25, session 3 — pipeline built, awaiting AWS credential confirmation._

## Completed (evidence in analysis/00–10, manual_checks.md; raw numbers analysis/out/)
- Parquet conversion + normalized parquet (`work/*_norm.parquet`, norm v2) for train/test.
- EDA, GT, noise, missingness, difficulty (01,03,04,05,06). Blocking study (07). Features (08). Model E1–E6 (09).
- Dev set: 43,888 eval S1 (train S1 hash%50==0), 2.67M candidates (`work/dev_cands.parquet`, `dev_feats.parquet`).

## Competition rule update (organizers, 2026-09-25)
candidate_pairs.tsv is reviewed with its code for FINAL ranking: a SMALLER candidate set per S1 ranks higher (beyond LB).
→ Treat candidates/S1 as a co-objective. Ours: retrieval (~150/S1) → learned stage-1 pruning (p1 ≥ τ) → final stage-2 matcher
scores only the pruned set = candidate_pairs.tsv. Choose τ on fold 7 as the smallest set within ~0.0003 F0.5 of best
(dev: τ=1e-4 → 8.1/S1 keeps 99.992% positives; 1e-3 → 6.1/S1, 99.95%; 1e-2 → 4.7/S1, 99.67%). Report cands/S1 everywhere.

## Key discoveries
- Each pool record matches ≤1 S1; 5.58% S1 singletons; mean 3.46 matches/S1; 26% pool = unmatched distractors.
- ~50% of S1 share their core name → name alone is ambiguous; address + competition decide.
- No postal codes; France (test only) has no state map; state features worth ~0.0005.
- Legal-suffix removal collides as much as it helps (keep as feature, not key).
- True copies have perturbed house numbers; generator also creates near-twin distractors (±1 letter / ±house no.).
- Devanagari shop-word pool names: 0/15,355 matched (distractor signature); 2× more in test.

## Session 3 results (dev set, holdout macro-F0.5; details 12, 13, 14)
- Stage-2 collective features 0.9625 → 0.9678; + sibling consensus (house number/street/anchor) → **0.9707**
  (P 0.991, R 0.948). Stage-3 iteration, bigger trees, per-segment thresholds: no gain.
- Key discovery: true copies agree with EACH OTHER on perturbed house numbers (69% vs 0.27% for negatives).
- Remaining loss (oracle): FN +0.0177 (empty-address +0.0062, mostly shared-name → irreducible), FP +0.0078.
- Normalization v3 frozen (13); production config frozen (14); AWS pipeline `aws/` smoke-tested locally (validator PASS).

## Current best (pre-session-3 summary)
- Blocking: reverse TF-IDF pool→S1 (name+addr, cap 30k–100k) ∪ rev char-3gram ∪ fwd na@10 ∪ core2 blocks.
  Config A 98.4% recall ≤84 cand/S1; D 99.1% ≤174 cand/S1 (eval sample, approx).
- Model: LightGBM ~60 features (+number edit distance). Dev holdout macro-F0.5 **0.9642** (P 0.987, R 0.932,
  singleton acc 0.95). Perfect-matcher ceiling on same candidates 0.996.
- Threshold: plateau 0.65–0.80, chosen 0.72 (dev).

## Gap to benchmark (0.9833)
- Dev 0.964 is on a 2% S1 sample where the global 1-to-many assignment could NOT be evaluated.
- Losses: recall (empty-address positives 52% accepted; "normal" positives 95%), near-twin FPs, wrong-S1 FPs.

## Unresolved (high value)
1. Full-partition end-to-end validation (global best-S1 constraint, abstention, twins) — coded in aws/evaluate.py, runs on AWS.
2. Whether production-scale training (~50× more S1) closes part of the gap (learning curve suggests ≈ +0.003–0.006).
3. France quality (no labels) — sanity checks in validate stage.
(Resolved this session: collective features, pool-duplicate evidence, normalization v3, distractor structure.)

## AWS status
User approved using the account with dedicated isolation. Bucket `s3://<your-bucket>` (private, public
access blocked); role `er-pipeline-role` (S3 access to that bucket only); SG `er-pipeline-sg` (no inbound).
vCPU quota = 5 (on-demand & spot) → running on **r6i.xlarge** (4 vCPU/32 GB), instance i-022377ac3ded4e869, launched
2026-09-25 ~13:24, self-terminates on exit. Stage-1 cascade (p1 ≥ 1e-4) added: final candidates ≈8/S1.
Monitor: `BUCKET=<your-bucket> aws/launch.sh logs`. Resume: `launch.sh launch` again (checkpoints in S3).

## Budget ledger (user cap: $70–80 total; ask before exceeding)
| run | instance | $/h | started (local) | status |
|---|---|---|---|---|
| v1 baseline (config F, 2-way cross-fit) | r6i.xlarge on-demand i-022377ac3ded4e869 | 0.26 | 13:24 | running |
| v2 (cap 100k, 3-way cross-fit, consensus feats, policy G) | r6i.8xlarge spot i-063f0c5c3eca1bb22, prefix entity-resolution-v2 | ~0.7 | 13:47 | spot-reclaimed 17:12 after features (checkpointed) |
| v2 resume #1 | r6a.8xlarge spot i-08b5ac239b1245170 | ~0.53 | 17:15 | spot-reclaimed within minutes (no work) |
| v2 resume #2 (config_v2b) | r6i.2xlarge ON-DEMAND i-081f4c28aca2b2435 | 0.52 | 17:25 | running |
EC2 quotas now: on-demand standard 16 vCPU, spot 32 vCPU. SageMaker ml.* job quotas 0 (adjustable) — not needed.

## Session 3b status (2026-09-26)
- v2 submitted: hidden LB **0.936** vs local 0.979 → diagnosed orphan-cluster shift (analysis/17 UPDATE).
- v1 aborted (memguard, 32 GB) in stage1 — superseded by v2, not relaunched.
- V3 running: r6i.4xlarge on-demand i-0cec25c722ec66d75, prefix entity-resolution-v3, config_v3.yaml
  (augment +7 feats, orphanize 16%, LGB/XGB/blend, policies A–G). Monitor: scratchpad mon3.sh.
- After V3: fetch p2 → local tune_decision with ORPHAN_FRAC=0.16 → finalize → validator → report LB-ready TSVs.

## Directive (user, 2026-09-26): performance > cost/runtime
Use full data, no unjustified subsampling; size epochs/capacity/data by learning curves; do not reject transformers/deeper
models for cost alone — estimate complementary signal first; report test-like (orphan) AND ordinary views.
Prepared: stage2_nn fusion (config_v3nn.yaml: 2 capacities, ≤8 epochs + early stop, train/held-out curves, 25/50/100% data
curve, full candidate set). tune_decision reports test-like + ordinary holdout. GPU quotas = 0 in all regions checked
(request L-DB2E81BA → 8 in ap-south-1 to unlock g5/g4dn for char-CNN / transformer cross-encoder).

## Next action
Monitor v1+v2 (scratchpad mon.sh) → compare holdouts → next experiment or freeze → fetch →
15_final_validation.md, 16_final_submission_report.md.

## V3 results (2026-09-26 ~21:30 UTC)
Test-like (orphan) holdout f0/f8/f9: XGB+G 0.98455/0.98455/0.98462 (τ 1e-4, 10.1 cand/S1); XGB+A@0.70 τ0.05 0.98404/0.98419/0.98422
(3.84 cand/S1). Ordinary view (orphans removed) only +0.0007 higher → no overfit to the simulation. LGB 0.9821; blend 0.9834.
V3 test: 3.45 matches/S1 (v2 3.60). Files: output_v3_auto/ (pipeline pick, validator PASS). Expected LB ≈ 0.97 (range 0.955–0.984).
Tuning rule fixed: tolerance only on candidate size; best policy wins at chosen τ (was costing ~0.0004).
User decision: will submit V4 only. V4 = i-042c7c291860319be (xgb2 up to 8000 rounds + fusion NN ×2), ETA ~09:00 UTC.

## 2026-09-26 ~05:25 UTC
- V3 submitted: LB **0.966** (v2 0.936; rank ~500; #1 0.98999). Orphan fix confirmed (+0.030).
- Remaining gap 0.018: test unmatched records are 4x more 'tempting' (22.8% with best p in 0.05–0.7) than random
  simulated orphans (5.9%) → test orphans are near-twins of PRESENT S1 (analysis/out/91). 7.7% of train S1 have such a
  look-alike (out/92).
- V5 launched (spot r6i.4xlarge i-0cb30822e61fddb92, prefix entity-resolution-v5, config_v5.yaml): hard-orphan
  simulation = drop 159,669 look-alike S1 + 193,565 random (16%, list work/v5_dropped_s1.parquet), 7 new features,
  LGB + XGB(1500) + XGB2(≤8000). NN off pending V4 evidence. Monitor: scratchpad mon5.sh. ETA ~10:30 UTC.
- V4 (i-042c7c291860319be) continues: large NN not better than standard so far (0.0260 vs 0.0254).

## 2026-09-26 ~09:30 UTC — V5 done
- V5 (hard look-alike orphans) holdout 0.9906 on ITS population (optimistic: look-alike S1 removed from evaluation).
- Label-free test checks (out/93): S1_ambig V3→V5 France .272→.188, India .245→.187, US .189→.121; contested ~0.002→~0.0001;
  p≥.7/S1 unchanged 3.44. Test unmatched still 18.9% "tempting" vs 6.0% for V5 sim hard orphans → simulation not fully test-like.
- FINAL V5 TSVs: output_v5/ (validator PASS; 3.84 cand/S1; 3.41 matches/S1; decision analysis/aws_v5/decision_v5.json).
  Also output_v5_auto/ (≈10 cand/S1), output_v4a/, output_v3_auto/ (LB 0.966), output/ (v2, LB 0.936).
- Expected LB V5 ≈ 0.970–0.980 (> V3 0.966). V4 NN evaluation ~11:30 UTC decides V6 = V5 + NN.
- Repo for teammates: https://github.com/yugh88/Business_ML_Pipeline (V3 pipeline + configs v3nn/v5, single README).

## 2026-09-26 ~10:20 UTC — V5 LB 0.967; data deep-dive → V6
- V5 public LB 0.967 (V3 0.966): hard-orphan simulation barely helped.
- Rejected hypothesis: test orphans = copies of train S1 (only 2.7% key overlap, analysis/out/94).
- FOUND (out/95–100): test unmatched clusters are SIBLING-COMPANY families ("X Holdings/Group/Exports/International…").
  In train, pool name = S1 core + extra word ∈ {group, holdings, enterprises, exports, ventures, uptown, west, …}
  has ~0% match rate (279 such words, n≥100), while {center, services, dr/mr/sri, council, trust…} are noise (91–99%).
  V5 still accepted ~26k US/India + ~50k France (unseen French words) such pairs on test; holdout has ~20× fewer.
  GEN set in featx.py wrongly lumps group/enterprises with noise words.
- V6 = V5 + rule "reject if any extra word is a train-derived descriptor (rate<5%) or French analog":
  holdout +0.0006–0.0007 (P .9966→.9972, R unchanged); test removed 83,613 pairs (FR 38.8k, IN 38.0k, US 6.8k).
  Files: output_v6/ and ~/Desktop/V6_submission/matching_results.tsv (validator PASS). Rule: analysis/aws_v5/v6_descriptor_rule.json.
- Next (V7): make it a cross-fitted token-rate FEATURE (extra/missing words), fix GEN set, retrain on hard-orphan data.

## Session 4 (2026-09-26) — root cause of the LB gap (label-free forensics, analysis/out/113–137)
- Density-reweighted holdout (113) does NOT reproduce LB → gap is not decoy density alone.
- Clean categories (same name+same number) occur at identical per-S1 rates in train and test → test copy generation = train.
- **Twin signature**: generator twins have house number = S1's + 1..9 (never minus; 119). True copies' ±k perturbations are
  symmetric. Test predictions: +k 43.8/1k vs −k 18.6/1k (US) → ≈25/1k twin false merges; pairwise p1 is symmetric (19/15.5).
- Test twins come as multi-record clusters (size-2 two-source groups 14× train, 121); train twins are single records.
  Stage-2 collective/consensus features promote clusters: "p2x2 accepts but pairwise p1 rejects" = 80/1k US, 135/1k India
  in test vs 36/69 in train (134) → estimated 7–10% of test S1 carry a false merge ≈ the 0.991 → 0.967 gap.
- **France (124–126)**: groupe/france/developpement/fils/associes are COPY NOISE when swapped in (85–90% keep exact number) →
  V6's French analog list was wrong (removed ~39k likely-true pairs). French siblings = holding/participations/
  international/distribution/SNC (≈10% same number, +k 20–60%).
- **V7** = V5 − {+k twin with same-source own-number support, +k groups in both sources, US/India train descriptors,
  French sibling descriptors}: holdout 0.99057 → 0.99097; removes 31/59/19 pairs per 1k S1 (US/India/France). output_v7/.
- V7b (+ reject promoted changed-number records): holdout −0.0011, extra test effect uncertain → not recommended.
- Holdout loss after V7 (137): FN 0.0063 (empty-address copies not retrieved / p1<tau), FP 0.0027.
- Next: V8 = retrain stage 2 with test-like twin/sibling CLUSTER augmentation (hard negatives) so collective features
  stop promoting clusters; recall work on empty-address copies.

## V8 run (launched 2026-09-26 ~18:17 local, instance i-0b3524f53e71a5191 (relaunched 18:21 with word-class features extra_desc/extra_noise; first instance stopped during bootstrap), r6i.4xlarge on-demand $1.04/h, prefix entity-resolution-v8)
- V4 (NN) terminated on user approval (it consumed the same collective features → cannot fix the twin-cluster shift).
- Train input = V5 input + 327,157 synthetic hard negatives (scripts/140_make_aug.py): twin entities (house number +{1,2,3,4,5,7,9})
  and descriptor siblings, 1–3 records each (sizes 74k/90k/25k entities), cloned from real copies + one light noise op.
- New features: featx num_sdiff (signed number diff), num_plusk (twin offset flag), extra_desc / extra_noise (injected word is a train sibling descriptor / only copy-noise words, word_classes.json; GEN fixed: group/enterprises removed); collective k1_s1num_same_src,
  k1_s1num_other_src, k1_num_same_src, k1_num_other_src (per-source version vs twin cluster).
- Reuses V5 test features (features_test/retrieve_test/union_test checkpoints copied; augment recomputes EXTRA on them).
  Train re-blocked/featurized from scratch; same V5 orphan drop list; config_v8.yaml (workers 14 for 16 vCPU).
- Mini smoke test passed end-to-end (validator PASS); stage-2 importance: num_plusk #2, k1_s1num_same_src #4.
- Budget: Cost Explorer (lagged) usage ≈ $20.4 through 2026-09-26 (EC2 compute $14.0, S3 $3.1); + V4 tail + V8 ≈ $35–40 total.
- **Address fingerprint (153)**: a candidate whose RAW address string equals another strong same-source candidate's is a
  copy of the same entity (per-source version). Train: true copies 22% share, siblings/distractors 0.1–1% (same-number
  siblings 0.8%). France: same-number category swaps (club↔ecole↔amicale…) share 20% = same as copy-noise words (20.3%)
  → they are COPIES (keep; do NOT add a French category-swap rule); French sibling words (holding/participations/…) 1% → siblings.
  Candidate feature for V9 (collective): addr_fingerprint_shared (+ name fingerprint), and FN rescue for rejected candidates.
- Fingerprint by swapped-word position (157): train last-word swaps, same number: true 24.8% share vs false 1.8%;
  France category-word (last) swaps 19.6% ≈ French noise words 20.4% ≈ other 20.2% → copies; V5 accepts 78% (≈ right).
  FN rescue via fingerprint (154) does NOT work (co-located businesses share addresses; precision 12–44%).
- **Hindi / Indian scripts (160)**: ~23% of India S2/S3 records contain Indian-script text (Devanagari 10% of names; also
  Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Gurmukhi), same proportions in train and test; S1 is always Latin.
  They are true copies at the same rate as Latin records (73%). normlib transliterates them (train-learned token dictionary +
  anyascii). V5 holdout: Devanagari-name records recall 0.991 / precision 0.998, other Indic 0.989 / 0.998, Latin 0.971 / 0.995.
  Test (V7): predicted-match rate identical across scripts (57–58%) → no Hindi-specific problem; not a cause of the LB gap.
- **Diagnostic suite (170, V5 baseline)**: holdout loss FN 0.006 / FP 0.0035 (singleton FP 0.0013). FN by stage (3 folds):
  not retrieved 11.7k (68% empty-address), cascade p1<tau 7.6k (62% empty), scored low 8.1k, decision cut 7.3k.
  V5 calibration: +k over-predicted (bin .7: p .75 vs true .59), −k under-predicted (bin .1: p .15 vs true .51) — no signed
  offset in V5 (V8 adds num_sdiff). Test margins: tight decisions (rejected within 0.2 of weakest accepted) 1.6–1.8% of S1 vs
  0.4% train; weak accepts (<0.9) US 4.6% / India 7.9% / France 10.3% vs train 3.3% → shift = less confident test decisions.
- No ID / row-order leak between copies of the same entity within a source (row / id distances ≈ random).
- **Stage-3 (171) on V5 scores** (LightGBM on OOF p2 folds 1–3 + fingerprint / offset / group features, G policy tuned on
  fold 7): holdout 0.99057→0.99084 / 0.99053→0.99089 / 0.99057→0.99090 (+0.0003; fingerprints +0.0002 of it).
  Plan: apply the same stage-3 on V8 scores (local, no AWS) → V9 candidate.
- **Full stage-3 (172, V5 scores)**: + promotion (p2−p1), sibling/noise word flags, name-edit counts, p2-weighted house-number
  groups by source (own_same_src / own_other_src / grp_w / grp_nsrc / n_numgroups): holdout 0.99057→0.99207,
  0.99053→0.99195, 0.99057→0.99206 (+0.0015). Top gain: own_other_src (S1-number support from the OTHER source), p2, p1,
  own_same_src, fp_tok_src, promo. scripts/173_stage3_apply.py = train (OOF folds 1–4) + fold-7 policy + holdout + test
  submission + label-free test checks. V9 = V8 + stage-3 (output_v9).
- **V5 + stage-3 applied to test (173 → output_v5s3, validator PASS)**: holdout 0.99199/0.99189/0.99196; test twin excess
  (+k − −k per 1k): US 25→14, India 18.7→8.6, France 15.8→9.0; promoted US 80→73, India 135→119, France 111→144 (↑, likely
  French category-swap copies). Fallback candidate if V8 fails.
- **Stage-3 variants (174)**: folds 1–6 best (0.99209/0.99194/0.99207); bigger trees / name fingerprint no gain; 2nd pass worse.
- **V5+stage-3 diagnostics (170 v5s3)**: FP loss 0.0035→0.0029, FN 0.0060→0.0051; calibration fixed for ±k; test US excess
  79→61/1k (+k N0 19.2→11.6). Remaining local FN: not retrieved 19.3k/3 folds (66% empty-address), scored low 6.0k,
  decision cut 4.3k. Remaining FP: same-number 2.0k, empty 1.5k, d100+ 1.0k.
- **Rules on top of stage-3 (158)**: R4/R3 ≈ 0 (stage-3's x_desc handles descriptors); R1R2 holdout −0.00027, test removes
  US 12.1 / India 9.7 / France 9.9 per 1k; symmetry says ~8.6/1k of those are twin FPs → net positive on test for V5s3.
  For V9 decide from V8's own ±k symmetry.
- **S2↔S3 relational features (176, V5 scores; user-requested test)**: pool-pool name/address similarity to other-source /
  same-source candidates weighted by their OOF p2 and ownership. Holdout all +0.00006/+0.00002/−0.00003 (inconsistent);
  orphan-exposed segment −0.00107/−0.00068 on 2 of 3 folds; test diagnostics WORSE (+k US 32.3→34.2, India 48.2→49.2,
  France 14.4→15.9; promoted up 5–8/1k) → NOT added (twin clusters are mutually similar across sources = same trap as
  collective features). Re-test on V8 scores (synthetic twins in training) before final decision.
- Stage-3 m_max now GLOBAL over the split (was fold-local): same holdout, test promoted US 73→69/1k.
- V8: union train 14:30 UTC — 338.6M pairs, 153 cands/S1, blocking recall 0.99083 (incl. synthetic pool).
- **Orphan-exposed S1 (177, stage-3 on V5)**: 4.5% of S1, F 0.968, loss 81% FP; FPs = copies of the dropped look-alike
  (2,122 of 2,377), 88% appended to S1s that already have ≥2 strong accepted copies; stage-3 over-confident there (median p3
  0.905; 48% < 0.9 vs 0.7% of TPs). Examples: identical chain name in another city, alias-only name on the same street,
  category swap at a neighbour number. No clean policy fix; S2↔S3 similarity did not help this segment. Residual.
- **Stage-3 with all stage-2 scores (178)**: holdout +0.00006/+0.00011/+0.00009 (consistent) but test checks slightly worse
  (US +k 32.3→34.4, promoted 69→73/1k; within noise). V9 builds both (output_v9 multi, output_v9b single) → choose on V8.
- **Interim comparison (test label-free, per 1k S1; twin excess = +k − −k)**: V5 US 25.2 / IN 18.7 / FR 15.8, promoted 80/135/111;
  V7 2.2 / 0.5 / −0.6, promoted 55/98/97 (holdout 0.9910); V5s3 13.6 / 8.5 / 8.6, promoted 73/118/142 (holdout 0.9921);
  **V5s3r = V5s3 − twin rules R1/R2 (+R3)**: 2.7 / −0.4 / −0.05, promoted 65/115/136, holdout ≈0.9918, validator PASS →
  output_v5s3r (Desktop V5s3r_submission). Best interim candidate before V9.
- **LB: V5s3r = 0.973** (V5 0.967; local ≈0.9918 → gap 0.019, was 0.024). Twin fix confirmed on hidden test (±k symmetry ≈0).
  Remaining gap must come from other excess categories: empty-number identical names (US 10.6 / IN 8.5 per 1k), d100+ identical
  names (9.9 / 5.0), same-number name-changed N5/N6/N7 (~12 US), France. V8/V9 must shrink these (not only ±k).
- **V5→V5s3r attribution**: removed 53.8/1k (+k twins 16, descriptors 25), added 27.1/1k (16.7 name-changed records). Sample of
  added: ~9/14 copies, ~4/14 FPs (twin +4, 41A→42A, category swap + sector change, brand replaced) → additions ≈70% precise on test
  (below F0.5 break-even ~77%): neutral-to-slightly-negative; removals clearly positive.
- **Stage-3 veto-only (179, V5)**: base∩stage3 holdout 0.99147/0.99145/0.9914 (+0.0009 vs base, −0.0006 vs full stage-3), no test
  additions. V9 builds veto variant too (output_v9veto); choose on V8 diagnostics.

## V9 = FINAL CANDIDATE (2026-09-27 00:22 IST) — output_v9/ , Desktop V9_submission/ (validator PASS incl. --check-ids)
- V8 base (synthetic twin/sibling clusters + twin/word/per-source features), tuned tau 0.05 p2x2 G(a=1.0, lam=0.05) cap 1:
  holdout aug 0.9897/0.98964/0.98951, clean 0.99108/0.99106/0.99095. 3.81 cand/S1.
- V9 = V8 + stage-3 (multi-score, folds 1–6, G a=1.25 lam=0): holdout aug 0.99027/0.99032/0.99015, clean 0.9915/0.9916/0.99144.
- Test checks vs V5s3r (LB 0.973; common V5-p1 reference): twin excess US −0.9 / IN 2.7 / FR −0.5 (no rules needed);
  promoted 49/95/124 vs 65/115/136. Removes vs V5s3r: same-number name-changed 11.9, identical-name far numbers 7.3,
  empty-number 11.2 per 1k (the suspected residual-gap categories). Stage-3 additions over V8 sampled ~80–90% copies.
- Variants not chosen: V9b (single-score; lower holdout), V9veto (base∩stage-3; lower holdout, additions shown net-positive),
  twin rules on V9 (symmetry already ~0), R3 (16 French pairs only).
- AWS V8 instance terminated after the run.

## Generator reverse-engineering audit (2026-09-27, scripts 194–199) — STOPPED: no large lever left
- **Copy-channel noise (195)**: typos / OCR digits / injected accents / punctuation / address reordering are NOT copy-only:
  look-alike negatives carry them too. Hardest class (first word replaced, same address): true 0.455 with noise vs 0.443 without.
  V9 mean p3 matches holdout truth within ~0.01 in every class x evidence bucket → no unexploited signal. Refuted.
- **Teammate "adaptive normalization" (196)**: test-derived stop list (df >= 0.3%) + frequent word-swap groups. Holdout ablation
  (accept what the adaptive normalizer would collapse): 0.99150/0.99160/0.99144 → 0.99028/0.99046/0.99026 (−0.0012, all folds);
  swaps −0.0002. US adds TP 206 / FP 830, India TP 304 / FP 1782 → harmful for US/India. France (198): real gap for dotted legal
  forms (S.A.S., S.A., E.U.R.L.), SCI, EI, OCR '5arl' — rejected pairs have copy-like address fingerprints (~2k pairs ≈ 8 per 1k FR S1),
  but 93–96% of those S1 already have other accepted copies → ≈ +0.0001 LB. groupe/developpement/france extras, club↔ecole swaps,
  −sci: rejected sets are distractor-like (fingerprint 0.01–0.06) → V9 correct; a stop list would add false merges.
- **India initials (199)**: 191's low fingerprint was an artifact (true India initials copies share 0.003 on holdout); density ≈ train.
- **Order/ID leak (197)**: none — true pairs' ID gaps / row gaps distributed like random pairs (corr ≈ 0).
- **S1 structure**: copies per S1 (truth ≈ 6% singletons, mean ≈ 3.5) matches V9 test k-distribution (France k=0 5.3% vs 6.0%).
- Remaining measurable levers: French aliases (+0.0002–0.0005, sign uncertain for brands), French legal forms (+0.0001),
  US/India look-alike density excess (≤ +0.004 if removed perfectly; needs V10-style retrain). Nothing applied; V9 stays best.

## Full-pipeline error budget + V10 (2026-09-27, scripts 200–211; outputs analysis/out/200–211)
- **Holdout budget (V9 clean 0.99152, loss 0.00848)**: true pairs not retrieved (p1<1e-4) 0.625% → +0.0019 if perfectly decided;
  stage-1 cascade drops 0.357% → +0.0011 (realized with G only +0.00006, tune report); decision FN 0.764% → +0.0025
  (p3<0.5 7.3k, G set-size cut 7.2k); FP → +0.0030 (natural distractors 0.0019, orphan copies 0.0010, other present S1: 6 pairs).
  Perfect decoding on V9's candidates = **0.99705** (τ-set) / 0.99812 (p1≥1e-4). Missed pairs are NOT concentrated on S1 with
  no found copy (2.3%; full-miss S1 0.6/1k → 0.0006). Transitive pool-pool retrieval via accepted copies: +0.00001 (exact
  name+addr) / precision 0.5% (raw address). Isotonic recalibration −0.00007; exclusivity loses 6 pairs → decoding saturated.
- **Test budget (LB simulator 203/204)**: V9 p3 as truth reproduces v2 exactly (0.9358 vs 0.936) but over-predicts V9 by ~0.009
  ("blind" loss shared by all versions); step realizations v2→V3 86%, V3→V5 23%, V5→V5s3r 76%, V5s3r→V9 69%. Truth models
  mixing calibrated pairwise p1 (α≈0.25–0.5) fit the recent LB deltas; α=1 (trust stage-3) is excluded by both.
- **Mix decoder** (G on α·p3+(1−α)·iso(p1)): holdout cost US/India −0.00005/−0.00028 (α .75), −0.0010/−0.0022 (α .5).
  Holdout shows α .5 removes 5.2/1k TRUE +1..9 US copies (stage-1 twin features) → +k "twin" reading was wrong.
  Simulator-free holdout-referenced check (211): α .75 +0.00036 LB (US+India), α .6 +0.00006, α .5 −0.00041. France: fingerprint
  says mix removes copy-like no_overlap pairs and adds non-copy identical pairs → France kept = V9.
- **V10 = V9 with US/India decoded at α_d=0.75, France = V9 rows**: output_v10/, ~/Desktop/V10_submission/ (validator PASS,
  --check-ids PASS). Changes: US −4,948/+2,688 pairs, India −7,653/+4,567. Expected LB ≈ +0.0004 (range −0.0002…+0.001).

## V10 (2026-09-27) — test-density look-alike retrain (user-approved)
- Teammate/user ideas checked (214/215): graph closure for single-source S1 hurts holdout (−0.0002…−0.0018, adds 33–66%-precision
  look-alikes); BM25/4-gram/FAISS retrieval has no headroom (misses: 63% empty address, identical-name misses 99% shared by ≥5 S1,
  only 868/11,814 alias-type); pretrained multilingual embeddings add nothing beyond V9 (212). Per-country stage-3 models:
  ensemble(global+country) +0.00006 US / +0.00011 India holdout, all folds positive -> used (scripts/173c_stage3_country.py).
- Augmentation (216, sanity 217): 554k synthetic look-alike records / 330k entities, per-country rates from the test excess profile
  (213): fwr, swap, light, numx(+10..99/far/trunc), namenum, empty + V8 twins (India rate 6%->2.2%) and siblings. Fixed a V8 sampler
  bug (fixed-seed group shuffle picked copies from one source; now per-record random, 48/52 S2/S3 like real copies).
- French dotted legal-form fix = post-hoc decision rule (220; AWS normalization unchanged so reused V8 test features stay consistent):
  same legal form after collapsing dots, same/missing house number, same address -> 931 pairs / 891 French S1 on V10-v9 dry run.
- ABLATION launched 07:35 UTC: 30% S1 subsample (work/abl, 218), arm A = V8 look-alikes (prefix entity-resolution-v10a,
  i-05ab25c5a52ecc19d), arm B = V10 look-alikes (entity-resolution-v10b, i-0f4a57a696c7d66f4), spot r6i.4xlarge, xgb2 off,
  V8 test features reused (code identical to V8). Evaluator: scripts/219_ablation_eval.py.
- PRE-REGISTERED pass criteria (B vs A): (1) holdout-referenced LB value of B's test changes > 0 (US+India); (2) excess accepted
  (test acc − holdout TP) falls in the targeted categories; (3) clean holdout cost ≤ ~0.001 per country; (4) ±k symmetry / promoted
  not worse. Pass -> full run (setup_v10.sh, config_v10.yaml, work/v10_input) on-demand r6i.4xlarge (~7.5 h, ~$8).
- **ABLATION RESULT (09:10 UTC): arm B (V10 look-alikes) FAILS** the pre-registered criteria (219 + same-variant check):
  holdout-referenced LB value −0.0027 (US −0.0021, India −0.0040); targeted categories mostly worse (number-changed twins,
  India swaps/identical groups +0.5…+2.7 excess/1k; only US first-word-replaced −1.9, trunc −0.75, empty −0.56 improve);
  test twin excess worse under every scorer (US +7.2…+9.4 vs A +3.0…+3.9; India +18.5…+20.0 vs +17.9…+18.2); clean holdout
  −0.0005…−0.0014. Likely cause: India twin rate cut + twin records spread across sources weakened twin learning.
  -> full V10 retrain NOT launched. Ablation cost ≈ $1.5 (2 spot r6i.4xlarge ~1.5 h). S3 prefixes entity-resolution-v10a/b
  (≈34 GB each) can be deleted.
- Teammate files audited (224–227): T1 (V9 +35.8k +k twins/−18.7k cap-7) predicted 0.964–0.966; T2 (+7.5k identical same-number)
  ~0.976; T4 (+12.9k, superset of T2) ~0.974 (0.972–0.976; hard ceiling 0.980 only if every addition were true). Labelled replays of
  their addition types: 1–9% precision; French additions 0% address fingerprint. Nothing usable for our submission.
- FINAL LOCAL BUILD (V11, no AWS): V9 scores + per-country stage-3 ensemble (173c, AUG=v8 -> output_v11s3) + alpha 0.75 tempering
  (US/India, re-validated) + French dotted legal-form rule (220). output_v8/candidate_pairs.tsv restored from identical V9 copy.
- **V11 = FINAL (2026-09-27 ~10:00 UTC)**: output_v11/, ~/Desktop/V11_submission/ (validator PASS incl. --check-ids).
  V9 stage-2 scores + per-country stage-3 ensemble (output_v11s3; fair clean-fold-7-tuned comparison vs V9: 0.99164 vs 0.99157,
  + on all folds, US +0.00005 India +0.00011) with aug-fold-7 decision (a=1.0, lam=0) + alpha 0.75 pairwise tempering US/India
  (holdout-referenced value +0.00051 US / +0.00035 India country F; clean holdout cost −0.0001 / −0.00044) + French dotted
  legal-form rule (+1,066 pairs / 1,018 S1). Test: accepted US 3371 / IN 3342 / FR 3360 per 1k, twin excess −1.2 / +3.1 / −0.4,
  promoted 44.6 / 69.1 / 67.8 (V9 49 / 95 / 124). Holdout ladder: V11 0.99148, −FP +0.0028, +decision FN +0.0028, cascade 0.0011,
  retrieval 0.0019. Simulator vs V9: +0.0009 (alpha .5) / +0.0021 (alpha 0) / −0.0004 (alpha 1, V9-biased) -> expected LB ≈ 0.978.
- **V12 = FINAL (noise pass, 231–233)**: output_v12/, ~/Desktop/V12_submission/ (validator PASS incl. --check-ids) = V11 + 1,531 pairs from
  the only noise fixes with 100% labelled-holdout precision: R2 house-number formats lost in parsing (N°, bis/ter, #, letter suffix; raw
  number == S1 number, identical name, same street) US +221 / IN +16 / FR +1,129; R3b OCR-digit-only name variants, same address
  US +37 / IN +91 / FR +37. French additions share raw address with accepted copies 0.150 (copies 0.19-0.22, look-alikes ~0).
  Tested and REJECTED (holdout precision 8-49%, F negative): order-invariant numbers (India reordered-address gap is not recoverable
  by rule), typo names, empty-address unique names, removing hidden-number twins. Expected LB ≈ 0.978 (0.9765-0.979).
- Teammate method (reported LB 0.987: DeBERTa/RoBERTa cross-encoders + CatBoost/LGBM, threshold >0.92, France normalization):
  (234) a high threshold on OUR p3 is harmful (t 0.92: holdout US −0.0047 / India −0.0054; holdout-referenced test value −0.003;
  French removed pairs share addresses 0.185 ≈ copies) -> their gain is model-side, not the threshold. French postal codes exist in
  only 0.4-0.5% of addresses (irrelevant). Accents/legal forms/abbreviations/country partition already handled.
- **France-specific conservative rule (236/237, label-free)**: French identical-name same-address accepted pairs with low p3 share raw
  addresses far less than true copies of the same set size (bias-checked on labelled US/India: low-p3 true copies share 0.7x of
  high-p3 true copies; false share 0-4%) -> est. 18-28% true for p3<0.8 in S1 with k>=3 and >=1 strong copy.
  **V13 = V12 − 3,492 such French pairs (2,559 S1, none emptied)**: output_v13/, ~/Desktop/V13_submission/ (validator PASS),
  expected +0.0003 LB over V12 (robust across bias corrections 0.6-0.85). t<0.9 variant rejected (worst case near break-even).
- Cross-encoder pilot (235, local MPS, frozen multilingual embeddings, bs 32) running = cheap validation gate before any full CE run.
- **Cross-encoder (CE) integration -> V14 (2026-09-27, scripts 238-242, local MPS, $0)**: pilot CE (235: multilingual MiniLM-L12,
  200k pairs from train folds 1-6, frozen embeddings, max 64 tokens) scored the uncertain band (0.02<=p3<=0.98, p1>=0.05) of V11s3
  folds 0/7/8/9 + test (238). Stacker (LightGBM: logit p3, CE logit, logit p1, class, offset, country) fit on clean fold 7 (239):
  band log-loss 0.270 -> 0.171, errors 14,104 -> 9,141 (folds 0/8/9). **Unchanged V11 decoding on clean holdout 0/8/9:
  US 0.99248 -> 0.99385 (+0.00137), India 0.98938 -> 0.99224 (+0.00286)**; FP 4,923 -> 4,018, TP +7,864.
  Test (240): adds/removes per 1k US 13.4/8.2, India 21.6/16.8 (holdout 15.2/3.4, 20.4/6.7); holdout-referenced value US +0.0013..+0.0022,
  India +0.0021..+0.0039 (country F); removals +k-dominated (twins). **France NOT applied**: CE (never trained on French data) would
  add 62/1k and remove 27/1k (3-4x US/India), dominated by swap_extra/desc classes; for identical@same (the only class where the
  fingerprint estimator validates on labelled holdout: est 0.90 vs actual 0.96 for adds) French CE adds look like look-alikes
  (est 7-12% true, break-even 70%). **V14 = V13 + CE changes for US/India only**: output_v14/ (variants output_v14_fr{pooled,country}
  kept for reference), ~/Desktop/V14_submission/ (official validator PASS --check-ids; 5,811,617 pairs; preds subset of candidates;
  candidate_pairs.tsv identical to V9-V13). **Expected LB vs V13: +0.0015..+0.0027 -> ~0.980-0.981.**
- idea.txt audit (user-supplied "0.994 blueprint"), holdout evidence: (1) hard-twin generator = V8 aug (done) / V10 (failed ablation
  -0.0027); twins are +k only, not +-k. (2) cross-encoder = the one supported pillar (V14). (3) zero-tolerance door-number veto
  (|d| 1..50): holdout accepted TPs with number offsets +-1..99 are 226/1k S1 (India) / 71/1k (US) vs FPs 1.7 / 0.7 -> veto costs
  >= -0.007 India / -0.003 US F even counting only +-1..9. (4) singleton gate (241, LightGBM on per-S1 score features, fold-7 fit):
  upper bound +0.0013, realised US -0.00006 / India +0.00003 (gated 510 S1, only 244 truly singleton). (5) global bipartite
  matching: 0 of 7.64M labelled pairs share a record; decoder already enforces exclusivity; LAP without S1 capacity decomposes to
  per-record argmax = our m_max rule; exclusivity FNs = 6 in 556k S1 -> gain ~0.
- AWS GPU quotas still 0 (ap-south-1: on-demand G/VT 0, spot G/VT 0; no request filed). CE v2 (243) training locally on MPS:
  all 314k hard + 200k easy pairs, 96 tokens (64 truncated 18% of pairs), 2 epochs, ~2 h, then band scoring -> ce2_band_*.parquet.
- **LB: V14 = 0.980** (submitted 2026-09-27 evening; V9 0.977). Prediction was +0.0015..+0.0027 over V13 (V11-V13 never submitted;
  chain V9 -> V14 predicted ~+0.003) -> CE path confirmed on the leaderboard.
