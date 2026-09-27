"""Holdout-referenced value of a decoder change (no simulator): per country x class, the change's removals/additions per 1k S1
on TEST vs on the labelled HOLDOUT (TP/FP split). Holdout-like part valued at holdout truth; test EXCESS removals valued as
look-alike FPs removed (test copy density per class = train, as the clean classes show); excess additions valued as FPs (conservative).
Value per pair: FP removed +0.21, TP removed -0.09, TP added +0.09, FP added -0.21 (typical set of ~3.4 copies)."""
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
cl = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "cls"])
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
x = x.join(cl, on=["s1", "m"], how="left").join(t1, on="s1")
nsh = dict(x.group_by("country").agg(pl.col("s1").n_unique()).rows())
nst = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
def dec(q, ad):
    q = q.with_columns((ad * pl.col("p3") + (1 - ad) * pl.col("ip1")).alias("qa"))
    return decide(prep(q.select("s1", "m", "p1", "qa"), TAU, "qa"), d9).select("s1", "m")
h1 = dec(x, 1.0)
tot = {}
for ad in (0.75, 0.6, 0.5):
    ha = dec(x, ad)
    hr = h1.join(ha, on=["s1", "m"], how="anti").join(x.select("s1", "m", "y", "cls", "country"), on=["s1", "m"])
    hd = ha.join(h1, on=["s1", "m"], how="anti").join(x.select("s1", "m", "y", "cls", "country"), on=["s1", "m"])
    for c in ("US", "India"):
        cls_t = pl.read_parquet(f"work/copy_channel_test_{c}.parquet", columns=["s1", "m", "cls"])
        v9 = pl.read_parquet(f"work/mix_acc_{c}_1.0.parquet"); d = pl.read_parquet(f"work/mix_acc_{c}_{ad}.parquet")
        tr_ = v9.join(d, on=["s1", "m"], how="anti").join(cls_t, on=["s1", "m"], how="left").group_by("cls").len().with_columns(pl.col("len") / nst[c] * 1000).rename({"len": "rem_t"})
        ta_ = d.join(v9, on=["s1", "m"], how="anti").join(cls_t, on=["s1", "m"], how="left").group_by("cls").len().with_columns(pl.col("len") / nst[c] * 1000).rename({"len": "add_t"})
        k = 1000 / nsh[c]
        hr_ = hr.filter(pl.col("country") == c).group_by("cls").agg((pl.col("y").sum() * k).alias("remTP_h"), ((~pl.col("y")).sum() * k).alias("remFP_h"))
        hd_ = hd.filter(pl.col("country") == c).group_by("cls").agg((pl.col("y").sum() * k).alias("addTP_h"), ((~pl.col("y")).sum() * k).alias("addFP_h"))
        z = tr_.join(ta_, on="cls", how="full", coalesce=True).join(hr_, on="cls", how="full", coalesce=True).join(hd_, on="cls", how="full", coalesce=True).fill_null(0)
        z = z.with_columns((pl.col("rem_t") - pl.col("remTP_h") - pl.col("remFP_h")).alias("rem_excess"), (pl.col("add_t") - pl.col("addTP_h") - pl.col("addFP_h")).alias("add_excess"))
        z = z.with_columns((0.21 * pl.col("remFP_h") - 0.09 * pl.col("remTP_h") + 0.21 * pl.col("rem_excess").clip(0) - 0.09 * (-pl.col("rem_excess")).clip(0)
                            + 0.09 * pl.col("addTP_h") - 0.21 * pl.col("addFP_h") - 0.21 * pl.col("add_excess").clip(0)).alias("value_per1k"))
        v = float(z["value_per1k"].sum()) / 1000
        tot[(c, ad)] = v
        print(f"{c:6s} alpha_d={ad:<4} value {v:+.5f} (country F)  | " + " ".join(f"{r[0]}: rem_t {r[1]:.1f} (h TP {r[3]:.1f}/FP {r[4]:.1f}) add_t {r[2]:.1f} (h TP {r[5]:.1f}/FP {r[6]:.1f})"
              for r in z.sort("rem_t", descending=True).select("cls", "rem_t", "add_t", "remTP_h", "remFP_h", "addTP_h", "addFP_h").head(4).rows()), flush=True)
for ad in (0.75, 0.6, 0.5):
    print(f"alpha_d={ad}: weighted US+India LB value {(tot[('US', ad)] * 663106 + tot[('India', ad)] * 809986) / 1732544:+.5f}")
