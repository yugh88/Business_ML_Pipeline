"""Light test (band only, runs beside MPS training): do per-S1 / per-record CONTEXT features improve the CE stacker?
 ce_rank (rank of the pair's CE among the S1's band pairs), ce_gap (CE - best other CE of the S1), n_band, p3_max_s1 / n_strong_s1
 (all tau-set pairs of the S1), m_comp (best p3 of the same pool record for another S1), same-source strong copies.
Fit on fold 7, evaluate band log-loss / errors on folds 0/8/9. CE_SET env as in 244."""
import sys, os
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
CES = os.environ.get("CE_SET", "ce").split(",")
CLS = ["identical@same", "light_edit@same", "swap_extra@same", "first_word_replaced@same", "no_overlap@same", "desc@same",
       "identical@num_changed", "identical@empty", "identical@other", "other"]
OFF = ["same", "no_num", "+1..9", "-1..9", "+10..99", "-10..99", "trunc", "far", "nonnum"]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
# S1-level and record-level context from the full tau-set (streamed, only aggregates kept)
src = pl.concat([pl.scan_parquet(f"output_v11s3/p3/train/fold{f}.parquet").select("s1", "m", "p1", "p3", pl.lit(f).alias("fold")) for f in (0, 7, 8, 9)]) \
        .join(SYN.lazy(), on="m", how="anti").filter(pl.col("p1") >= 0.05)
s1ctx = src.group_by("s1").agg(pl.col("p3").max().alias("p3_max_s1"), (pl.col("p3") >= 0.9).sum().alias("n_strong_s1"), pl.len().alias("n_tau_s1")).collect()
band = src.filter(pl.col("p3").is_between(0.02, 0.98)).collect()
mctx = src.join(band.select("m").unique().lazy(), on="m", how="semi").select("s1", "m", "p3").collect()
mtop = mctx.sort("p3", descending=True).group_by("m").agg(pl.col("s1").first().alias("s1_top"), pl.col("p3").first().alias("p3_top"),
                                                             pl.col("p3").get(1, null_on_oob=True).fill_null(0).alias("p3_2nd"))
band = band.join(mtop, on="m", how="left").with_columns(pl.when(pl.col("s1") == pl.col("s1_top")).then(pl.col("p3_2nd")).otherwise(pl.col("p3_top")).alias("m_comp")) \
           .drop("s1_top", "p3_top", "p3_2nd")
y = pl.concat([pl.read_parquet(f"output_v11s3/p3/train/fold{f}.parquet", columns=["s1", "m", "y"]) for f in (0, 7, 8, 9)]).join(band.select("s1", "m"), on=["s1", "m"], how="semi")
band = band.join(y, on=["s1", "m"]).join(s1ctx, on="s1").join(t1, on="s1")
for c in CES:
    band = band.join(pl.read_parquet(f"work/{c}_band_train.parquet").rename({"ce": c}), on=["s1", "m"])
band = band.with_columns(pl.mean_horizontal(CES).alias("cem"))
band = band.with_columns(pl.col("cem").rank("ordinal", descending=True).over("s1").alias("ce_rank"), pl.len().over("s1").alias("n_band"),
                         (pl.col("cem") - pl.col("cem").sort(descending=True).get(0).over("s1")).alias("ce_gap_top"),
                         pl.col("m").str.slice(0, 2).alias("src"))
band = band.with_columns(pl.col("cem").rank("ordinal", descending=True).over(["s1", "src"]).alias("ce_rank_src"))
band = annotate(band, "train")
print(f"band {band.height}", flush=True)


def X(z, ctx):
    cols = [lg(z["p3"].to_numpy()), *[z[c].to_numpy() for c in CES], lg(z["p1"].to_numpy()),
            z["cls"].replace_strict(CLS, list(range(len(CLS))), default=len(CLS)).to_numpy(), z["off"].replace_strict(OFF, list(range(len(OFF))), default=len(OFF)).to_numpy(),
            (z["country"] == "India").to_numpy().astype(float)]
    if ctx:
        cols += [z[c].to_numpy().astype(float) for c in ("ce_rank", "ce_rank_src", "n_band", "ce_gap_top", "p3_max_s1", "n_strong_s1", "n_tau_s1", "m_comp")]
    return np.column_stack(cols).astype(np.float64)


A, B = band.filter(pl.col("fold") == 7), band.filter(pl.col("fold") != 7); yb = B["y"].to_numpy()
cat = [1 + len(CES), 2 + len(CES)]
for ctx in (False, True):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                       lambda_l2=10.0, verbose=-1, seed=7, num_threads=2), lgb.Dataset(X(A, ctx), A["y"].to_numpy().astype(int), categorical_feature=cat), num_boost_round=300)
    p = m.predict(X(B, ctx))
    print(f"  context={ctx}: band log-loss {log_loss(yb, p):.5f} AUC {roc_auc_score(yb, p):.4f} errors {int(((p >= 0.5) != yb).sum())}", flush=True)
    for c in ("US", "India"):
        mk = (B["country"] == c).to_numpy(); print(f"     {c}: log-loss {log_loss(yb[mk], p[mk]):.5f} errors {int(((p[mk] >= 0.5) != yb[mk]).sum())}")
