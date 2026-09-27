"""V11 finalize: per-country stage-3 ensemble scores (output_v11s3) -> clean holdout vs V9, re-validate alpha-0.75 pairwise tempering
for US/India on THESE scores (holdout cost + holdout-referenced test value, 211 method), assemble final matches (France = stage-3
decision), then the French dotted legal-form rule is applied by 220 and the validator runs (shell).
usage: 229_v11_finalize.py <stage3_dir> <out_dir>"""
import sys, os, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
S3D, OUT = sys.argv[1], sys.argv[2]
dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def decode(d, alpha, c=None):
    d = d.with_columns(pl.Series("ip1", iso.predict(d["p1"].to_numpy())).cast(pl.Float32))
    d = d.with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(d.select("s1", "m", "p1", "qa"), TAU, "qa"), c or dec).select("s1", "m")


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")


def holdout(pdir, alpha, d_=None):
    x = pl.concat([pl.read_parquet(f"{pdir}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti")
    c = d_ or json.load(open(f"{pdir}/decision_s3.json"))
    a = decide(prep(x.with_columns(pl.Series("ip1", iso.predict(x["p1"].to_numpy())).cast(pl.Float32)).with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("qa"))
                    .select("s1", "m", "p1", "qa"), TAU, "qa"), c).join(x.select("s1", "m", "y"), on=["s1", "m"])
    z = gt.join(a.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    z = z.with_columns(pl.Series("f", F(z["tp"], z["fp"], z["n"])))
    return z, a, x


z9, _, _ = holdout("output_v9", 1.0)
z11, a11, x11 = holdout(S3D, 1.0)
z11t, a11t, _ = holdout(S3D, 0.75)
fold_f = lambda z: [round(float(z.filter(pl.col("fold") == f)["f"].mean()), 5) for f in (0, 8, 9)]
cf = lambda z: {c: round(float(z.filter(pl.col("country") == c)["f"].mean()), 5) for c in ("US", "India")}
print(f"HOLDOUT clean  V9 {fold_f(z9)} {cf(z9)}\n               V11 {fold_f(z11)} {cf(z11)}\n       V11 a0.75 {fold_f(z11t)} {cf(z11t)}", flush=True)
# holdout-referenced test value of alpha 0.75 vs 1.0 (US/India)
nh = dict(gt.group_by("country").len().rows()); nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
_r = a11.select("s1", "m", "y").join(a11t.select("s1", "m"), on=["s1", "m"], how="anti").with_columns(pl.lit("a1").alias("w"))     # only CHANGED pairs
_d = a11t.select("s1", "m", "y").join(a11.select("s1", "m"), on=["s1", "m"], how="anti").with_columns(pl.lit("a075").alias("w"))
hx = annotate(pl.concat([_r, _d]), "train").join(t1, on="s1")
choice, final = {}, []
for c in ("US", "India", "France"):
    x = pl.read_parquet(f"{S3D}/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= TAU)
    d1 = decode(x, 1.0)
    if c == "France":
        choice[c] = 1.0; final.append(d1); continue
    d7 = decode(x, 0.75)
    h1 = hx.filter((pl.col("w") == "a1") & (pl.col("country") == c)); h7 = hx.filter((pl.col("w") == "a075") & (pl.col("country") == c))
    k = 1000 / nh[c]
    hr = h1.group_by("cls", "off").agg((pl.col("y").sum() * k).alias("remTP"), ((~pl.col("y")).sum() * k).alias("remFP"))
    hd = h7.group_by("cls", "off").agg((pl.col("y").sum() * k).alias("addTP"), ((~pl.col("y")).sum() * k).alias("addFP"))
    tr_ = annotate(d1.join(d7, on=["s1", "m"], how="anti"), "test").group_by("cls", "off").len().with_columns(pl.col("len") / nt[c] * 1000).rename({"len": "rem_t"})
    td_ = annotate(d7.join(d1, on=["s1", "m"], how="anti"), "test").group_by("cls", "off").len().with_columns(pl.col("len") / nt[c] * 1000).rename({"len": "add_t"})
    zz = tr_.join(td_, on=["cls", "off"], how="full", coalesce=True).join(hr, on=["cls", "off"], how="full", coalesce=True).join(hd, on=["cls", "off"], how="full", coalesce=True).fill_null(0)
    zz = zz.with_columns((pl.col("rem_t") - pl.col("remTP") - pl.col("remFP")).alias("rex"), (pl.col("add_t") - pl.col("addTP") - pl.col("addFP")).alias("aex"))
    val = float(zz.select(0.21 * pl.col("remFP") - 0.09 * pl.col("remTP") + 0.21 * pl.col("rex").clip(0) - 0.09 * (-pl.col("rex")).clip(0)
                          + 0.09 * pl.col("addTP") - 0.21 * pl.col("addFP") - 0.21 * pl.col("aex").clip(0)).sum().item()) / 1000
    hcost = float(z11t.filter(pl.col("country") == c)["f"].mean() - z11.filter(pl.col("country") == c)["f"].mean())
    choice[c] = 0.75 if (val > 0 and hcost > -0.0005) else 1.0
    print(f"{c}: alpha 0.75 test changes -{d1.join(d7, on=['s1','m'], how='anti').height / nt[c] * 1000:.1f}/+{d7.join(d1, on=['s1','m'], how='anti').height / nt[c] * 1000:.1f} per 1k; "
          f"holdout-referenced value {val:+.5f}, holdout cost {hcost:+.5f} -> use alpha {choice[c]}", flush=True)
    final.append(d7 if choice[c] == 0.75 else d1)
pred = pl.concat(final)
os.makedirs(OUT, exist_ok=True)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = pred.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id") \
     .write_csv(os.path.join(OUT, "matching_results.tsv"), separator="\t", quote_style="never")
json.dump(dict(alpha=choice, holdout_v9=fold_f(z9), holdout_v11=fold_f(z11), holdout_v11_country=cf(z11)), open(os.path.join(OUT, "v11_info.json"), "w"), indent=1)
print("written", OUT, pred.height, "pairs", choice)
