# 11 — AWS production plan

## Why AWS
Measured locally (single process): reverse TF-IDF ≈ 4e7 sparse multiply-adds/s; features ≈ 90k pairs/s but polars
peaked 9 GB at 2.7M pairs; one local job already pushed macOS into 5.5 GB swap. Full train+test production
(~20M pool queries, ~300M candidate pairs) is HEAVY → AWS only. Mac = code, samples, monitoring.

## Architecture (simplest reliable)
One EC2 instance (no SageMaker: no managed training benefit for LightGBM/sparse CPU work), S3 for all state.
`aws/run_pipeline.py` executes idempotent stages; each stage writes Parquet shards atomically (`*.tmp` → rename →
upload to S3) and skips shards already present in S3 → restart-safe.

S3 layout `s3://<bucket>/entity-resolution/{input,normalized,indexes,candidates,features,checkpoints,models,validation,submissions,logs}/`.
Upload: normalized parquet only (~1.3 GB) + raw test S1/S2/S3 parquet for ID validation (~0.5 GB).

## Instance choice
| option | vCPU | RAM | on-demand $/h (us-east-1) | fit |
|---|---|---|---|---|
| m7i.4xlarge | 16 | 64 GB | ~0.81 | OK for 8–10 workers |
| **r6i.4xlarge** | 16 | 128 GB | ~1.01 | **chosen**: 12–14 retrieval workers × ≤6 GB + headroom 25% |
| m6i.8xlarge | 32 | 128 GB | ~1.54 | faster, only if runtime > 4 h |
Spot pricing (~0.3–0.4 $/h) is used if available in the region; checkpointing makes interruption cheap.
Disk: 200 GB gp3 EBS (~$0.02/h). No GPU.

## Work estimate (from measured per-query work: US 2.5e4, India 4.3e4 ops/query at cap 30k)
| stage | ops / volume | 14 workers | notes |
|---|---|---|---|
| reverse na cap 30k, train+test (~20M queries) | ~6.7e11 | ~20–30 min | per country, 50k-query shards |
| reverse c3 cap 5k, train+test | ~1.8e11 | ~8 min | |
| forward na cap 20k (S1 4M queries) | ~0.7e11 | ~3 min | |
| core2 blocks ≤200 (DuckDB) | — | ~3 min | |
| features (train subset + full test, ~250M pairs) | — | ~30–45 min | 1M-pair shards, 6 workers × ~4 GB |
| stage-2 aggregates + LightGBM train (≤40M rows) | — | ~30 min | 16 threads |
| full-partition validation + test inference + submission | — | ~20 min | |
| optional cap-100k upgrade | ×3 retrieval | +1 h | only if validation shows gain |

Expected wall time ≈ 2.5–4 h → **compute ≈ $3–5 on-demand (≈$1.5 spot)**; S3 + EBS + transfer < $1.
Leaves room for 2–3 re-runs within the $20 soft cap.

## Memory safety on AWS
Chunked sparse retrieval (`max_out` bound per chunk), one index per country loaded once per worker (fork after load,
copy-on-write), `memguard` watchdog in each worker (abort at <25% system RAM available), worker count benchmarked on
one shard first (start 8 → scale to 14 only if RSS allows), shards released after write.

## Validation on AWS (required before test)
Full train production blocking; hold out fold 0 (10% of S1 by hash, all its candidates and competing S1 intact)
end-to-end; compare decision layers A–F (threshold, best-S1 constraint, abstention, margin, twin propagation).

## Credentials needed (when the time comes)
An IAM identity (profile or env vars — never pasted into files) with: `sts:GetCallerIdentity`; S3 create/list/put/get on
one bucket; EC2 run/describe/terminate instances, create/describe key pair or SSM, security group, and an instance profile
with S3 access to the same bucket (`iam:PassRole`). Lightweight check first: `aws sts get-caller-identity`, `aws s3 ls`,
`aws ec2 describe-instance-types --instance-types r6i.4xlarge`.

## SageMaker vs EC2 — checked 2026-09-25 (read-only quota queries, ap-south-1)
| | SageMaker Processing / Training (this account) | EC2 (running) |
|---|---|---|
| usable instance quota | Processing: only ml.t3.medium ×4 (2 vCPU/4 GB), ml.t3.large ×4 (8 GB), ml.t3.xlarge ×2 (4 vCPU/16 GB, burstable). **All m5/m6i/r5/r6i/c5/c6i processing and training quotas = 0** (incl. spot). | on-demand & spot standard vCPU quota 5 → r6i.xlarge (4 vCPU, 32 GB, non-burstable) |
| fits workload? | No: largest usable box has 16 GB and burstable CPU; features/stage-1 need ~20–25 GB | Yes (smoke-tested; memguard + RLIMIT) |
| price | ml.* ≈ 15–25% above the equivalent EC2 price; ml.t3.xlarge would run far longer (CPU credits) | ~$0.27/h |
| checkpointing | built-in S3 input/output channels, but the pipeline already checkpoints every stage to S3 | same S3 checkpoints; relaunch resumes |
| reliability | managed job retries; needs a container/image, job definitions per stage | single instance, self-terminates; failure → relaunch resumes from last finished stage |
| effort | port 12 stages to Processing/Training jobs + quota increase request (manual approval, hours–days) | done, running |

**Decision: continue on EC2.** SageMaker brings no memory/CPU/price advantage for sparse retrieval + LightGBM and is
blocked by zero quotas. Revisit only if a quota increase (e.g. ml.r5.4xlarge processing) is approved and a re-run is needed.
