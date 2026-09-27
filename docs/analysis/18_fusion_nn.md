# 18 — Fusion neural model (deep-learning stream) — design, cost, GPU status

## Why (and why not a plain FNN / big transformer)
Advice from a previous winner: understand the data, pick models that fit it, try own FNN / multimodal fusion.
Their 2025 task was price regression from product text (DeBERTa + features, cross-attention fusion). Ours is pairwise
matching of short noisy strings where the decisive evidence is character-level (dropped digits, injected tokens,
synthetic brand names, transliteration) plus structural signals (orphan clusters, competition). On dev, a plain FNN on
our tabular features scored 0.953 vs LightGBM 0.964 → a plain FNN is not the right model. The transferable idea is the
**two-stream fusion**: a text encoder that reads the raw pair + the engineered-feature stream, fused by attention.

## Architecture (`aws/scripts/stage2_nn.py`, ~9.2M params, trained from scratch, BSD-3 PyTorch)
* Text streams (name pair, address pair): learned **character-trigram embedding bag** (exact 41³ trigram vocabulary,
  masked mean+max pooling) per record → pair interaction [u, v, |u−v|, u·v] → 128-d.
* Feature stream: MLP over the ~95 standardized engineered + collective + consensus features and stage-1 p1.
* Fusion: the 3 streams as tokens → multi-head self-attention (4 heads) + LayerNorm → MLP head → match logit.
* Training: identical K-way cross-fitting as stage 2, orphan-simulated train (analysis/17), pairs with p1 ≥ 0.01
  (≈5/S1 — everything the final decision can accept), 2 epochs, AdamW, early selection by held-out log-loss.
* Outputs: p2nn and p2nnb = mean(p2_LGB, p2nn); evaluate / tune_decision pick among {p2, p2x, p2blend, p2nn, p2nnb} on
  fold 7 only; holdouts 0/8/9 reported. Kept only if it beats LightGBM on the orphan-matched holdout.

## Compute
| encoder | train rows/s (8 CPU threads) | inference rows/s | full run on r6i.4xlarge |
|---|---|---|---|
| char-CNN (first version) | ~740 | ~1,370 | infeasible on CPU (~5 h inference alone) |
| **trigram embedding bag (chosen)** | ~14,000 | ~34,000 | ≈ 2 h (train 3 models × 2 epochs + score ~20M pairs), ≈ $2 |
Vectorized encoding (Arrow buffers + LUT): 1M strings in 0.6 s.

## GPU status (checked read-only 2026-09-26)
G/VT on-demand and spot quotas = **0** in ap-south-1, us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1,
ap-southeast-1, ap-northeast-1 (standard CPU quota is 5 outside ap-south-1). To enable GPU: request
"Running On-Demand G and VT instances" (L-DB2E81BA) → 8 in ap-south-1 (g5.2xlarge A10G $1.46/h or g4dn.2xlarge T4
$0.83/h). GPU would make the char-CNN and a small transformer cross-encoder practical.

## Where a transformer could still help (only after evidence)
A small (≤100M, Apache/MIT) cross-encoder re-scoring only the **uncertain** pairs (p2 in 0.05–0.95, ≈10% of the
pruned set) — worthwhile only if V3 error analysis shows remaining errors are textual. Structural errors (orphan
clusters, empty-address shared names) are handled by data simulation / decision logic, not by bigger text models.

## Run plan
After V3 finishes: download V3 p2/validation, then relaunch the v3 prefix with `config_v3nn.yaml`, clearing only the
stage2_nn → validate checkpoints (features/coll/p2 reused; no re-copy). Then local τ/policy tuning over all variants.
