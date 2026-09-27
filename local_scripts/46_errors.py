"""Error analysis of the dev LightGBM model on the holdout split (threshold 0.72)."""
import sys; sys.path.insert(0, "scripts"); import memguard
import polars as pl, numpy as np, lightgbm as lgb, json
exec(open("scripts/43_models.py").read().split("def X(df")[0].split("R = {}")[1])  # re-use feature lists & split setup
def X(df, cols): return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
m = lgb.Booster(model_file="work/dev_lgb_all.txt")
ho = d.filter(pl.col("split") == "hold"); ho = ho.with_columns(pl.Series("p", m.predict(X(ho, ALL))))
T = 0.72
ho = ho.with_columns((pl.col("p") >= T).alias("pred"))
seg = pl.when(pl.col("a2_empty")).then(pl.lit("addr_empty")).when(pl.col("brand2")).then(pl.lit("brand")).when(pl.col("dom2")).then(pl.lit("domain")) \
        .when(pl.col("indic2")).then(pl.lit("indic")).when(pl.col("n_tsr") < 50).then(pl.lit("weakname")).when(pl.col("a_tsr") < 70).then(pl.lit("weakaddr")).otherwise(pl.lit("normal"))
ho = ho.with_columns(seg.alias("seg"))
print("## positives: recall by segment (candidates only)")
print(ho.filter(pl.col("y")).group_by("seg").agg(pl.len().alias("n"), pl.col("pred").mean().alias("recall"), pl.col("p").median().alias("med_p")).sort("n", descending=True))
print("## false positives by segment")
fp = ho.filter(~pl.col("y") & pl.col("pred"))
print(fp.group_by("seg").agg(pl.len().alias("n_fp"), pl.col("p").median()).sort("n_fp", descending=True), "total FP", fp.height, "TP", ho.filter(pl.col("y") & pl.col("pred")).height)
# per-S1: upper bound if the model were perfect on candidates
g = gt.filter(pl.col("split") == "hold")
cand_tp = ho.filter(pl.col("y")).group_by("s1").len().rename({"len": "c_tp"})
ub = g.join(cand_tp, on="s1", how="left").fill_null(0).with_columns(pl.when(pl.col("n") == 0).then(1.0).when(pl.col("c_tp") == 0).then(0.0)
     .otherwise(1.25 * pl.col("c_tp") / pl.col("n") / (0.25 + pl.col("c_tp") / pl.col("n"))).alias("f"))
print("## macro-F0.5 upper bound given blocking (perfect matcher):", ub["f"].mean())
# singletons with false merges
single = g.filter(pl.col("n") == 0).select("s1")
sfp = fp.join(single, on="s1", how="semi")
print("## singleton S1 with >=1 predicted match:", sfp["s1"].n_unique(), "/", single.height, " segs:", sfp.group_by("seg").len().rows())
# is the FP pool record matched to another S1 in GT? (true twin elsewhere) vs a pure distractor
allp = pl.read_parquet("work/train_pairs.parquet").select("m", pl.col("s1").alias("true_s1"))
fpx = fp.join(allp, on="m", how="left").with_columns(pl.col("true_s1").is_null().alias("pure_distractor"))
print("## FP pool record is an unmatched distractor:", fpx["pure_distractor"].mean(), " (else it belongs to another S1)")
fpx.select("s1", "m", "true_s1", "p", "seg", "n_tsr", "a_tsr", "rrank", "r_margin_best", "name_freq_s1").sort("p", descending=True).write_parquet("work/dev_holdout_fp.parquet")
ho.filter(pl.col("y") & ~pl.col("pred")).select("s1", "m", "p", "seg", "n_tsr", "a_tsr", "rrank", "r_margin_best").write_parquet("work/dev_holdout_fn.parquet")
print(memguard.report())
