"""LOCAL smoke-test data: tiny train/test subsets (S1 sample + their true pool + random distractors), incl. France."""
import os, sys, polars as pl
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
gt = pl.read_parquet(f"{src}/train_gt.parquet")
s1 = pl.scan_parquet(f"{src}/train_s1.parquet").filter(pl.col("entity_id").hash(1) % 1000 < 3).collect()
g = gt.join(s1.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id", how="semi")
ids = set(x for v in g["matched_entity_ids"].to_list() if v for x in v.split(","))
for s in (2, 3):
    pl.scan_parquet(f"{src}/train_s{s}.parquet").filter(pl.col("entity_id").is_in(list(ids)) | (pl.col("entity_id").hash(2) % 1000 < 1)).collect().write_parquet(f"{dst}/train_s{s}.parquet", row_group_size=3000)
s1.write_parquet(f"{dst}/train_s1.parquet", row_group_size=3000); g.write_parquet(f"{dst}/train_gt.parquet")
pl.scan_parquet(f"{src}/test_s1.parquet").filter(pl.col("entity_id").hash(1) % 1000 < 3).collect().write_parquet(f"{dst}/test_s1.parquet", row_group_size=3000)
for s in (2, 3):
    pl.scan_parquet(f"{src}/test_s{s}.parquet").filter(pl.col("entity_id").hash(2) % 1000 < 4).collect().write_parquet(f"{dst}/test_s{s}.parquet", row_group_size=2000)
print({f: pl.read_parquet(f"{dst}/{f}").height for f in os.listdir(dst)})
