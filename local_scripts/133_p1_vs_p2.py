"""Pairwise stage-1 (p1) vs collective stage-2 (p2x2) decisions: holdout F and test twin diagnostics."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
from tune_decision import prep, decide
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
def score(pr, d, s1f):
    x = pr.join(d.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
    g = s1f.join(x, on="s1", how="left").fill_null(0)
    return float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean())
# tune p1 on fold 7
d7 = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == 7).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
best = None
for a in (0.7, 1.0, 1.25, 1.5, 2.0):
    for lam in (0.0, 0.15, 0.5):
        c = dict(policy="G_expected_f", a=a, lam=lam, cap=1)
        f = score(decide(prep(d7, 0.05, "p1"), c), d7, gt.filter(pl.col("fold") == 7))
        if best is None or f > best[0]: best = (f, c)
print("p1 best on fold7:", best, flush=True)
del d7
cp1 = best[1]; cp2 = json.load(open("analysis/aws_v5/decision_v5.json"))
for f in [0, 8, 9]:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    s1f = gt.filter(pl.col("fold") == f)
    print(f"fold {f}: p2x2 {score(decide(prep(d, 0.05, 'p2x2'), cp2), d, s1f):.5f}  p1 {score(decide(prep(d, 0.05, 'p1'), cp1), d, s1f):.5f}", flush=True)
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1t = dict(s1c.group_by("country").len().rows())
for c in ["US", "India", "France"]:
    d = pl.scan_parquet("work/v5_p2/test/*.parquet").filter(pl.col("country") == c).select("s1", "m", "p1", "pc2", "pa", "p2x2").collect()
    for v, cc in [("p2x2", cp2), ("p1", cp1)]:
        pr = attach(decide(prep(d, 0.05, v), cc), "test"); n = nS1t[c] / 1000
        cnt = dict(pr.group_by("off").len().rows())
        print(f"TEST {c} {v}: pred/S1 {pr.height/nS1t[c]:.3f}  +1..9 {cnt.get('+1..9',0)/n:.1f}/1k  -1..9 {cnt.get('-1..9',0)/n:.1f}/1k  empty {cnt.get('empty',0)/n:.1f}  same {cnt.get('same',0)/n:.0f}", flush=True)
    del d
