"""Ablation inputs (30% S1 subsample, identical real data in both arms):
  train_s1 / train_gt : S1 with hash % 10 < 3 (all folds represented), their GT rows
  pool (real)         : every true copy of a kept S1 + 30% of pure distractors (records that are nobody's copy)
  arm A (control)     : + V8 synthetic look-alikes of kept S1 (aug_v8_*)
  arm B (treatment)   : + V10 synthetic look-alikes of kept S1 (aug_v10_*)
Writes work/abl/{train_s1,train_gt,train_s2_A,train_s3_A,train_s2_B,train_s3_B}.parquet"""
import sys, os
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
os.makedirs("work/abl", exist_ok=True)
s1 = pl.read_parquet("work/train_s1.parquet")
sub = s1.filter((pl.col("entity_id").hash(4242) % 10) < 3)
sub.write_parquet("work/abl/train_s1.parquet")
keep = sub.select(pl.col("entity_id").alias("s1"))
gt = pl.read_parquet("work/train_gt.parquet")
gt.join(keep.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="semi").write_parquet("work/abl/train_gt.parquet")
pairs = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"])
copies_kept = pairs.join(keep, on="s1", how="semi").select(pl.col("m").alias("entity_id"))
all_copies = pairs.select(pl.col("m").alias("entity_id")).unique()
for s in ("s2", "s3"):
    real = pl.read_parquet(f"work/train_{s}.parquet")
    pure = real.join(all_copies, on="entity_id", how="anti")
    pure = pure.filter((pl.col("entity_id").hash(4243) % 10) < 3)
    base = pl.concat([real.join(copies_kept, on="entity_id", how="semi"), pure])
    for arm, aug in (("A", "v8"), ("B", "v10")):
        meta = pl.read_parquet(f"work/aug_{aug}_meta.parquet", columns=["entity_id", "s1"]).join(keep, on="s1", how="semi")
        syn = pl.read_parquet(f"work/aug_{aug}_{s}.parquet").join(meta.select("entity_id"), on="entity_id", how="semi")
        out = pl.concat([base, syn])
        out.write_parquet(f"work/abl/train_{s}_{arm}.parquet")
        print(f"{s} arm {arm}: real copies {real.join(copies_kept, on='entity_id', how='semi').height} + pure distractors {pure.height} + synthetic {syn.height} = {out.height}", flush=True)
print("S1 kept", sub.height, "of", s1.height, "| GT rows", pl.read_parquet("work/abl/train_gt.parquet").height)
