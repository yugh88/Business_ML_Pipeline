"""NEW signal for empty-address copies: pool->pool exact raw-name link.
An unmatched pool record r (esp. empty address) whose EXACT raw business_name equals the raw name of accepted records of
exactly ONE S1 (same country) is proposed as a copy of that S1. Precision/recall on train holdout (V5 decision)."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
pairs = pl.read_parquet("work/train_pairs.parquet").select("s1", "m").with_columns(pl.lit(True).alias("y"))
# all train pool records with raw name / empty-address flag
pool = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet").select("entity_id", "business_name", "business_address", "country") for s in (2, 3)]).collect()
nrm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "a") for s in (2, 3)]).collect()
pool = pool.join(nrm, on="entity_id").with_columns((pl.col("a") == "").alias("aempty")).drop("a", "business_address")
print("pool:", pool.height, "empty-address:", int(pool["aempty"].sum()))
# accepted pairs over ALL folds (V5 decision) — the anchors
acc_parts = []
for f in range(10):
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    acc_parts.append(decide(prep(d, dec["tau"], "p2x2"), dec).select("s1", "m").with_columns(pl.lit(f).alias("fold")))
acc = pl.concat(acc_parts); del acc_parts
print("accepted pairs:", acc.height)
A = acc.join(pool.rename({"entity_id": "m"}), on="m")                      # s1, m, fold, business_name, country, aempty
# anchors: raw name -> set of S1 (only non-empty-address accepted records act as anchors)
anch = A.filter(~pl.col("aempty")).group_by("country", "business_name").agg(pl.col("s1").unique().alias("s1s"), pl.len().alias("n_anchor"))
anch = anch.filter(pl.col("s1s").list.len() == 1).with_columns(pl.col("s1s").list.first().alias("s1_link")).drop("s1s")
unmatched = pool.join(acc.select(pl.col("m").alias("entity_id")).unique(), on="entity_id", how="anti")
prop = unmatched.join(anch, on=["country", "business_name"]).select(pl.col("entity_id").alias("m"), pl.col("s1_link").alias("s1"), "aempty", "n_anchor")
prop = prop.join(pairs, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(False))
print("proposed links (unmatched pool record -> single S1 via exact raw name):", prop.height)
print(prop.group_by("aempty").agg(pl.len(), pl.col("y").mean().round(4).alias("precision")).rows())
print("by #anchors:", prop.group_by(pl.col("n_anchor").clip(1, 4)).agg(pl.len(), pl.col("y").mean().round(4)).sort("n_anchor").rows())
# holdout F0.5 with the links added (folds 0/8/9)
for f in [0, 8, 9]:
    s1f = gt.filter(pl.col("fold") == f)
    base = acc.filter(pl.col("fold") == f).select("s1", "m")
    add = prop.join(s1f.select("s1"), on="s1", how="semi").select("s1", "m")
    def sc(p):
        x = p.join(pairs, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(False)).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
        g = s1f.join(x, on="s1", how="left").fill_null(0)
        return f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean()
    print(f"fold {f}: V5 {sc(base):.5f} -> +links {sc(pl.concat([base, add]).unique()):.5f}  (+{add.height} links, precision {add.join(pairs, on=['s1','m'], how='left')['y'].fill_null(False).mean():.4f})")
