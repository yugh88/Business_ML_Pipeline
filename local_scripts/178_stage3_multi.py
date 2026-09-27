"""Stage-3 with several stage-2 model scores as inputs (LightGBM p2, XGBoost p2x, big XGBoost p2x2) vs p2x2 only.
Uses the cached stage-3 frames of 176 (same features) + extra score columns joined from the p2 files.
usage: 178_stage3_multi.py <p2_dir> <decision.json> <cache_dir>"""
import sys, json, os, gc
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, lightgbm as lgb
p2dir, decf, cache = sys.argv[1:4]
src = open("scripts/173_stage3_apply.py").read()
g = {"__name__": "s3"}; sys.argv = ["x", p2dir, decf, "/dev/null", cache]
exec(src.split("tr = pl.concat([build(")[0], g)
build, F, f05, gt, TAU, V, prep, decide = g["build"], g["F"], g["f05"], g["gt"], g["TAU"], g["V"], g["prep"], g["decide"]
EXTRA_SCORES = [c for c in ("p2", "p2x") if c != V]


def frame(f):
    p = os.path.join(cache, f"train_fold{f}.parquet")
    d = pl.read_parquet(p) if os.path.exists(p) else build("train", pl.col("fold") == f)
    ex = pl.scan_parquet(f"{p2dir}/train/*.parquet").filter(pl.col("fold") == f).select(["s1", "m"] + EXTRA_SCORES) \
           .rename({c: f"s_{c}" for c in EXTRA_SCORES}).collect()
    return d.join(ex, on=["s1", "m"], how="left")


def score(d, p, c, s1f):
    x = d.select("s1", "m", "y", "fold", "p1", "pc2", "pa").with_columns(pl.Series("p3", p))
    pr = decide(prep(x, TAU, "p3"), c).join(d.select("s1", "m", "y"), on=["s1", "m"])
    gg = s1f.join(pr.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return float(f05(gg["tp"].to_numpy().astype(float), gg["fp"].to_numpy().astype(float), gg["n"].to_numpy().astype(float)).mean())


FM = F + [f"s_{c}" for c in EXTRA_SCORES]
Xc = lambda d, cols: d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
tr = pl.concat([frame(f) for f in range(1, 7)])
models = {n: lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1, num_threads=6),
                       lgb.Dataset(Xc(tr, cols), tr["y"].to_numpy()), 700) for n, cols in (("s3", F), ("s3_multi", FM))}
del tr; gc.collect()
GRID = [dict(policy="G_expected_f", a=a, lam=l, cap=1) for a in (1.0, 1.25, 1.5) for l in (0.0, 0.05)]
va = frame(7); s17 = gt.filter(pl.col("fold") == 7); pol = {}
for n, cols in (("s3", F), ("s3_multi", FM)):
    pv = models[n].predict(Xc(va, cols)); pol[n] = max(((score(va, pv, c, s17), c) for c in GRID), key=lambda z: z[0])
    print("fold7", n, round(pol[n][0], 5), pol[n][1], flush=True)
del va; gc.collect()
for f in (0, 8, 9):
    d = frame(f); s1f = gt.filter(pl.col("fold") == f)
    print(f"fold {f}:", {n: round(score(d, models[n].predict(Xc(d, cols)), pol[n][1], s1f), 5) for n, cols in (("s3", F), ("s3_multi", FM))}, flush=True)
    del d; gc.collect()
