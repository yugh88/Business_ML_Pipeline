"""Per-S1 configuration of the PREDICTED set by house-number offsets: which offset kinds are present.
Train holdout (V5 decision; per-config F0.5 and share) vs test (V5 file). Excess configs in test = where FPs live."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
def config(pr):
    k = pl.when(pl.col("off") == "same").then(pl.lit("S")).when(pl.col("off") == "+1..9").then(pl.lit("P")).when(pl.col("off") == "-1..9").then(pl.lit("M")) \
          .when(pl.col("off") == "empty").then(pl.lit("e")).otherwise(pl.lit("o"))
    pr = pr.with_columns(k.alias("k"))
    return pr.group_by("s1").agg(pl.col("k").unique().sort().str.join("").alias("cfg"), pl.len().alias("npred"))
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
pr = decide(prep(d, dec["tau"], "p2x2"), dec).select("s1", "m").join(d.select("s1", "m", "y"), on=["s1", "m"]); del d
pr = attach(pr, "train")
per = pr.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
c = config(pr)
g = gt.filter(pl.col("fold").is_in([0, 8, 9])).join(t1, on="s1").join(per, on="s1", how="left").join(c, on="s1", how="left").with_columns(pl.col("tp").fill_null(0), pl.col("fp").fill_null(0), pl.col("cfg").fill_null("<none>"))
g = g.with_columns(pl.Series("f", f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float))))
tr = g.group_by("country", "cfg").agg(pl.len().alias("n_tr"), pl.col("f").mean().round(4).alias("F_tr"), (1 - pl.col("f")).sum().alias("loss_tr"), (pl.col("n") == 0).mean().round(3).alias("singleton_share"))
tot = g.group_by("country").len().rename({"len": "N"})
tr = tr.join(tot, on="country").with_columns((1000 * pl.col("n_tr") / pl.col("N")).round(2).alias("per1k_tr"), (1000 * pl.col("loss_tr") / pl.col("N")).round(3).alias("loss_per1k_tr"))
p5 = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v5/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
ct = config(attach(p5.join(s1c, on="s1"), "test"))
te = s1c.join(ct, on="s1", how="left").with_columns(pl.col("cfg").fill_null("<none>")).group_by("country", "cfg").len().rename({"len": "n_te"})
te = te.join(s1c.group_by("country").len().rename({"len": "N_te"}), on="country").with_columns((1000 * pl.col("n_te") / pl.col("N_te")).round(2).alias("per1k_te"))
j = te.join(tr.select("country", "cfg", "per1k_tr", "F_tr", "loss_per1k_tr", "singleton_share"), on=["country", "cfg"], how="left").with_columns((pl.col("per1k_te") - pl.col("per1k_tr")).round(2).alias("excess"))
for cc in ["US", "India", "France"]:
    ref = j.filter(pl.col("country") == cc)
    if cc == "France":
        ref = ref.drop("per1k_tr", "F_tr", "loss_per1k_tr", "singleton_share", "excess").join(tr.filter(pl.col("country") == "US").select("cfg", "per1k_tr", "F_tr", "singleton_share"), on="cfg", how="left").with_columns((pl.col("per1k_te") - pl.col("per1k_tr")).round(2).alias("excess"))
    print(f"==== {cc} (France compared to US train)")
    print(ref.sort("per1k_te", descending=True).head(22))
j.write_parquet("work/s1_config.parquet")
