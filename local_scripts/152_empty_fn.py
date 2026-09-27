"""Empty-address / no-number copies that V5 misses (largest FN bucket, analysis/out/137): why are they missed and
could a name-only rule recover them safely? Uses train (labels) only."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
f = 0
gt = duckdb.connect().execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().filter(pl.col("fold") == f)
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
pairs = pl.read_parquet("work/train_pairs.parquet").join(gt.select("s1"), on="s1", how="semi")
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2", "a2_empty").collect()
pr = decide(prep(d, dec["tau"], "p2x2"), dec).with_columns(pl.lit(True).alias("pred"))
fn = pairs.join(pr, on=["s1", "m"], how="anti").join(d.select("s1", "m", "p1", "p2x2", "a2_empty"), on=["s1", "m"], how="left")
ids = pl.concat([fn.select(pl.col("s1").alias("entity_id")), fn.select(pl.col("m").alias("entity_id"))]).unique()
nrm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "ncore", "a") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
fn = fn.join(nrm.rename({"entity_id": "s1", "ncore": "c1", "a": "a1"}), on="s1").join(nrm.rename({"entity_id": "m", "ncore": "c2", "a": "a2"}), on="m")
fn = fn.with_columns((pl.col("a2") == "").alias("addr_empty"), (pl.col("c1") == pl.col("c2")).alias("same_core"),
                     pl.when(pl.col("p1").is_null()).then(pl.lit("not_cand")).when(pl.col("p1") < dec["tau"]).then(pl.lit("p1<tau")).otherwise(pl.lit("scored")).alias("why"))
print("FN pairs fold 0:", fn.height, " S1:", gt.height)
print(fn.group_by("addr_empty", "why").agg(pl.len(), pl.col("same_core").mean().round(3).alias("same_core")).sort("len", descending=True).rows())
# name uniqueness among S1 (same country): is the S1 core name shared by other S1?
s1n = pl.read_parquet("work/train_s1_norm.parquet", columns=["entity_id", "country", "ncore"])
cnt = s1n.group_by("country", "ncore").len().rename({"len": "s1_same_core"})
e = fn.filter(pl.col("addr_empty")).join(s1n.rename({"entity_id": "s1"}).select("s1", "country"), on="s1").join(cnt, left_on=["country", "c1"], right_on=["country", "ncore"], how="left")
print("empty-address FN: S1 core shared by k S1s:", e.group_by(pl.col("s1_same_core").clip(1, 5)).len().sort("s1_same_core").rows())
print("examples:"); print(e.select("c1", "c2", "a1", "why", "s1_same_core").head(12).rows())
