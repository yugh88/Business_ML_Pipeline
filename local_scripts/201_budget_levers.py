"""Levers behind the holdout error budget (200):
Q1 transitive (pool-pool) retrieval: can missed true pairs be found through an ACCEPTED record of the same S1 sharing an
   exact key (normalized name+address, or raw address string)? recall of missed pairs vs precision of all proposals.
Q2 what the V9 false merges are: orphan copy (dropped look-alike S1) / copy of another present S1 / natural distractor.
Q3 calibration + decoder regret: p3 reliability by bin; G re-decoded with isotonic p3 (fit fold 0, eval folds 8/9)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)
d9 = json.load(open("output_v9/decision_s3.json")); HF = [0, 8, 9]
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc", "cls"])
st = pl.read_parquet("work/error_budget_pairs.parquet")
g = pl.read_parquet("work/error_budget_s1.parquet")
N = g.height


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


# ---------------- Q1 transitive retrieval
keys = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "country", "ncore", "a") for s in (2, 3)]) \
    .join(pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet").select("entity_id", pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("ra")) for s in (2, 3)]), on="entity_id") \
    .with_columns(pl.when(pl.col("a") != "").then(pl.concat_str("country", pl.lit("|"), "ncore", pl.lit("|"), "a")).otherwise(None).alias("k_na"),
                  pl.when(pl.col("ra") != "").then(pl.concat_str("country", pl.lit("|"), "ra")).otherwise(None).alias("k_ra")).select("entity_id", "k_na", "k_ra").collect()
acc = h.filter("acc").select("s1", pl.col("m").alias("r"))
tset = h.select("s1", "m")                                    # tau-set candidates (already scored)
truth = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"]).with_columns(pl.lit(True).alias("true_pair"))
missed = st.filter(pl.col("stage") < "3").select("s1", "m")
print(f"Q1 TRANSITIVE RETRIEVAL — missed true pairs (not in tau-set): {missed.height}")
for kname in ("k_na", "k_ra"):
    kk = keys.select("entity_id", pl.col(kname).alias("k")).drop_nulls()
    freq = kk.group_by("k").len()
    kk = kk.join(freq.filter(pl.col("len") <= 50), on="k", how="semi")
    prop = acc.join(kk.rename({"entity_id": "r"}), on="r").join(kk.rename({"entity_id": "m"}), on="k").filter(pl.col("m") != pl.col("r")) \
              .select("s1", "m").unique().join(tset, on=["s1", "m"], how="anti")
    prop = prop.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("true_pair").fill_null(False))
    got = missed.join(prop, on=["s1", "m"], how="semi").height
    # value if all proposals were ACCEPTED (upper bound on harm) and if only true ones were (upper bound on gain)
    add = prop.group_by("s1").agg(pl.col("true_pair").sum().alias("atp"), (~pl.col("true_pair")).sum().alias("afp"))
    z = g.join(add, on="s1", how="left").fill_null(0)
    f0 = F(z["tp"], z["fp"], z["n"]).mean(); fall = F(z["tp"] + z["atp"], z["fp"] + z["afp"], z["n"]).mean(); ftrue = F(z["tp"] + z["atp"], z["fp"], z["n"]).mean()
    print(f"  key {kname}: proposals {prop.height} (true {int(prop['true_pair'].sum())}, precision {prop['true_pair'].mean():.3f}); recovers {got} missed pairs ({got / missed.height:.1%});"
          f"  F if all accepted {fall:.5f} ({fall - f0:+.5f}), if only true accepted {ftrue:.5f} ({ftrue - f0:+.5f})")
del keys
# ---------------- Q2 false merges
drop = pl.read_parquet("work/v5_dropped_s1.parquet").select("s1").with_columns(pl.lit(True).alias("dropped"))
fp = h.filter(pl.col("acc") & ~pl.col("y")).select("s1", "m", "cls", "p3")
owner = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"]).rename({"s1": "owner"})
fp = fp.join(owner, on="m", how="left").join(drop.rename({"s1": "owner"}), on="owner", how="left").with_columns(
    pl.when(pl.col("owner").is_null()).then(pl.lit("natural distractor")).when(pl.col("dropped").fill_null(False)).then(pl.lit("orphan copy (owner S1 dropped)"))
      .otherwise(pl.lit("copy of another PRESENT S1")).alias("fp_type"))
fp = fp.join(g.select("s1", "n", "tp", pl.col("fp").alias("nfp")), on="s1")
fp = fp.with_columns(pl.Series("value", F(fp["tp"], fp["nfp"] - 1, fp["n"]) - F(fp["tp"], fp["nfp"], fp["n"])))
print("\nQ2 FALSE MERGES by type (value = F gain per holdout S1 if removed):")
print(fp.group_by("fp_type").agg(pl.len(), (pl.col("value").sum() / N).round(5).alias("value"), pl.col("p3").mean().round(3).alias("mean_p3"),
                                 (pl.col("n") == 0).mean().round(3).alias("on_singleton")).sort("value", descending=True))
print(fp.group_by("fp_type", "cls").agg(pl.len(), (pl.col("value").sum() / N).round(5).alias("value")).sort("value", descending=True).head(12))
# ---------------- Q3 calibration + decoder regret
print("\nQ3 RELIABILITY of p3 on holdout (all tau-set candidates):")
hb = h.with_columns(pl.col("p3").cut([0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]).alias("bin"))
print(hb.group_by("bin").agg(pl.len(), pl.col("p3").mean().round(4).alias("mean_p3"), pl.col("y").mean().round(4).alias("true_rate")).sort("bin"))
p3 = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet") for f in HF])
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
p3 = p3.join(SYN, on="m", how="anti")
fold = duckdb.connect().execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
p3 = p3.drop("fold").join(fold, on="s1")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(p3.filter(pl.col("fold") == 0)["p3"].to_numpy(), p3.filter(pl.col("fold") == 0)["y"].to_numpy())
gt = g.join(fold, on="s1")
for f in (8, 9):
    x = p3.filter(pl.col("fold") == f)
    xi = x.with_columns(pl.Series("p3", iso.predict(x["p3"].to_numpy()).astype(np.float32)))
    out = []
    for nm, xx in (("V9", x), ("isotonic", xi)):
        a = decide(prep(xx, d9["tau"], "p3"), d9).join(xx.select("s1", "m", "y"), on=["s1", "m"])
        z = gt.filter(pl.col("fold") == f).select("s1", "n").join(a.group_by("s1").agg(pl.col("y").sum().alias("a"), (~pl.col("y")).sum().alias("b")), on="s1", how="left").fill_null(0)
        out.append(f"{nm} {F(z['a'], z['b'], z['n']).mean():.5f}")
    print(f"  fold {f} decoder: " + " | ".join(out))
