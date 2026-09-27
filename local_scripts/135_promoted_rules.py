"""Reject 'collectively promoted' records (p2x2 accepts, pairwise p1 rejects) by offset kind: holdout F + test removals."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/134_promoted.py").read().split("d = pl.scan_parquet")[0])
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
VAR = {"none": [], "plusk": ["+1..9"], "num_change": ["+1..9", "|d|10..99", "+100", "-100"], "num_change+empty": ["+1..9", "|d|10..99", "+100", "-100", "empty"],
       "all_but_same": ["+1..9", "|d|10..99", "+100", "-100", "empty", "-1..9", "nonint"], "all": ["+1..9", "|d|10..99", "+100", "-100", "empty", "-1..9", "same", "nonint"]}
for f in [0, 8, 9]:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    x = promoted(d, "train").join(d.select("s1", "m", "y"), on=["s1", "m"]); s1f = gt.filter(pl.col("fold") == f)
    line = f"fold {f}:"
    for k, offs in VAR.items():
        rm = (~x["acc1"]) & x["off"].is_in(offs)
        g = s1f.join(x.filter(~rm).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
        line += f" {k} {f05(g['tp'].to_numpy().astype(float), g['fp'].to_numpy().astype(float), g['n'].to_numpy().astype(float)).mean():.5f}"
    print(line, flush=True)
te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
nS1t = dict(te.group_by("country").len().rows())
for c in ["US", "India", "France"]:
    d = pl.scan_parquet("work/v5_p2/test/*.parquet").filter(pl.col("country") == c).select("s1", "m", "p1", "pc2", "pa", "p2x2").collect()
    x = promoted(d, "test"); n = nS1t[c] / 1000
    print(f"TEST {c}: " + " ".join(f"{k} -{((~x['acc1']) & x['off'].is_in(offs)).sum()/n:.1f}/1k" for k, offs in VAR.items()), flush=True)
    x.filter(~pl.col("acc1")).select("s1", "m", "off").write_parquet(f"work/test_promoted_{c}.parquet"); del d
