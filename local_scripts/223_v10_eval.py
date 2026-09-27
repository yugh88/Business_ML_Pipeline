"""V10 final evaluation (run after 222): clean holdout F per country vs V9, error-budget ladder, test category excess / +-k / promoted,
and whether the mild pairwise tempering (alpha 0.75, US/India) still pays on V10 (holdout cost + holdout-referenced test value).
usage: 223_v10_eval.py <stage3_dir e.g. output_v10s3> <p2_dir e.g. work/v10_p2>"""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate(), offset()
S3D, P2D = sys.argv[1], sys.argv[2]
dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = lambda c: pl.col(c).str.contains(r"^S[23]-[89]\d{9}$")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).filter(~SYN("m") & (pl.col("p1") >= TAU))
tr = pl.scan_parquet(f"{P2D}/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().filter(~SYN("m"))
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy())
pos4 = pl.scan_parquet(f"{P2D}/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & pl.col("y")).select("s1", "m", "p1").collect(); del tr


def decode(d, alpha):
    d = d.with_columns(pl.Series("ip1", iso.predict(d["p1"].to_numpy())).cast(pl.Float32))
    d = d.with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(d.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m")


res = {}
for alpha in (1.0, 0.75):
    a = decode(h, alpha).join(h.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
    z = gt.join(a, on="s1", how="left").fill_null(0)
    z = z.with_columns(pl.Series("f", F(z["tp"], z["fp"], z["n"])))                     # computed from z itself (join order not guaranteed)
    res[alpha] = {c: float(z.filter(pl.col("country") == c)["f"].mean()) for c in ("US", "India")} | {"all": float(z["f"].mean())}
    if alpha == 1.0:
        Z = z
print(f"HOLDOUT clean F (folds 0/8/9): V10 {res[1.0]}  (V9: all 0.99152, US 0.99266, India 0.98980)")
print(f"   alpha 0.75 tempering: {res[0.75]}")
# error-budget ladder for V10
tp_pairs = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"]).join(gt.select("s1"), on="s1", how="semi")
st = tp_pairs.join(pos4, on=["s1", "m"], how="left").join(decode(h, 1.0).with_columns(pl.lit(True).alias("acc")), on=["s1", "m"], how="left") \
             .with_columns(pl.when(pl.col("p1").is_null()).then(pl.lit(1)).when(pl.col("p1") < TAU).then(pl.lit(2)).when(~pl.col("acc").fill_null(False)).then(pl.lit(3)).otherwise(pl.lit(4)).alias("stage"))
per = st.group_by("s1").agg((pl.col("stage") >= 2).sum().alias("r4"), (pl.col("stage") >= 3).sum().alias("rt"))
g = Z.join(per, on="s1", how="left").fill_null(0)
n, r4, rt, tpv, fpv = (g[c].to_numpy() for c in ("n", "r4", "rt", "tp", "fp"))
lad = {"V10": F(tpv, fpv, n).mean(), "- all FP": F(tpv, 0, n).mean(), "+ all decision FN": F(rt, fpv, n).mean(), "perfect on tau-set": F(rt, 0, n).mean(), "perfect on p1>=1e-4": F(r4, 0, n).mean()}
print("LADDER:", {k: round(float(v), 5) for k, v in lad.items()})
# test diagnostics
rows = []
for c in ("US", "India", "France"):
    x = pl.read_parquet(f"{S3D}/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= TAU)
    acc = {al: decode(x, al) for al in ((1.0, 0.75) if c != "France" else (1.0,))}
    pw = decide(prep(x.select("s1", "m", "p1"), TAU, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).with_columns(pl.lit(True).alias("pw"))
    xa = annotate(acc[1.0].join(pw, on=["s1", "m"], how="left").with_columns(pl.col("pw").fill_null(False)), "test")
    ns = pl.read_parquet("work/test_s1.parquet", columns=["country"]).filter(pl.col("country") == c).height / 1000
    sym = (xa.filter(pl.col("off") == "+1..9").height - xa.filter(pl.col("off") == "-1..9").height) / ns
    print(f"TEST {c}: accepted {xa.height / ns:.1f}/1k, twin excess (+k - -k) {sym:+.2f}/1k, promoted {xa.filter(~pl.col('pw')).height / ns:.1f}/1k")
    rows.append(xa.group_by("cls", "off").len().with_columns((pl.col("len") / ns).alias("acc_t"), pl.lit(c).alias("country")).drop("len"))
    if c != "France":
        chg = acc[1.0].join(acc[0.75], on=["s1", "m"], how="anti").height / ns, acc[0.75].join(acc[1.0], on=["s1", "m"], how="anti").height / ns
        print(f"   alpha 0.75 would remove {chg[0]:.1f}/1k and add {chg[1]:.1f}/1k on test")
    for al, a in acc.items():
        a.write_parquet(f"work/v10_acc_{c}_{al}.parquet")
pl.concat(rows).write_parquet("work/v10_test_cat.parquet")
json.dump(dict(holdout=res, ladder={k: float(v) for k, v in lad.items()}), open("analysis/out/223_v10_eval.json", "w"), indent=1)
