"""Holdout (labelled) check of the +k logic: for V9 and the mix decoders, accepted TP / FP by house-number offset class,
per 1k holdout S1 (US / India). Tells whether the mix's +k removals hit twins (FP) or true copies with number noise."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
d9 = json.load(open("output_v9/decision_s3.json")); TAU = d9["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
x = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti")
x = x.with_columns(pl.Series("ip1", iso.predict(x["p1"].to_numpy())).cast(pl.Float32))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
nm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", pl.col("nums").str.split(" ").list.first().alias("h")) for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
x = x.join(nm.rename({"entity_id": "s1", "h": "h1"}), on="s1").join(nm.rename({"entity_id": "m", "h": "h2"}), on="m").join(t1, on="s1")
x = x.with_columns((pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)).alias("d"))
x = x.with_columns(pl.when(pl.col("d").is_between(1, 9)).then(pl.lit("+1..9")).when(pl.col("d").is_between(-9, -1)).then(pl.lit("-1..9"))
                     .when(pl.col("d").is_between(10, 99)).then(pl.lit("+10..99")).when(pl.col("d").is_between(-99, -10)).then(pl.lit("-10..99")).otherwise(pl.lit("rest")).alias("off"))
ns = dict(x.group_by("country").agg(pl.col("s1").n_unique()).rows())
out = []
for ad in (1.0, 0.75, 0.5):
    q = x.with_columns((ad * pl.col("p3") + (1 - ad) * pl.col("ip1")).alias("qa"))
    acc = decide(prep(q.select("s1", "m", "p1", "qa"), TAU, "qa"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    z = x.join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    t = z.group_by("country", "off").agg((pl.col("y").sum()).alias("true_cands"), (pl.col("acc") & pl.col("y")).sum().alias("TP"), (pl.col("acc") & ~pl.col("y")).sum().alias("FP"))
    t = t.with_columns(*[(pl.col(c) / pl.col("country").replace_strict(ns) * 1000).round(2) for c in ("true_cands", "TP", "FP")], pl.lit(ad).alias("alpha_d"))
    out.append(t)
r = pl.concat(out).filter(pl.col("off") != "rest").sort("country", "off", "alpha_d", descending=[False, False, True])
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_hide_dataframe_shape(True)
print("per 1k holdout S1: true candidates, accepted TP and FP by offset class")
print(r.select("country", "off", "alpha_d", "true_cands", "TP", "FP"))
