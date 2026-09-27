"""Why is the orphan-exposed segment at F~0.97 (stage-3 on V5)? FP from orphan copies vs other FP vs FN; examples."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
dec = json.load(open("output_v5s3/decision_s3.json"))
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
dropped = pl.read_parquet("work/v5_dropped_s1.parquet").select("s1")
gt = gt.join(dropped, on="s1", how="anti")
pairs = pl.read_parquet("work/train_pairs.parquet").select("s1", "m")
orph = pairs.join(dropped, on="s1", how="semi").select(pl.col("m"), pl.col("s1").alias("orph_s1"))
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
tot = {"orph_fp": 0, "other_fp": 0, "fn": 0, "S1": 0, "loss": 0.0, "loss_fp_only": 0.0}
ex = []
for f in (0, 8, 9):
    d = pl.read_parquet(f"output_v5s3/p3/train/fold{f}.parquet")
    pr = decide(prep(d, dec["tau"], "p3"), dec).join(d.select("s1", "m", "y"), on=["s1", "m"])
    exp = d.join(orph, on="m", how="semi").select("s1").unique()
    s1f = gt.filter(pl.col("fold") == f).join(exp, on="s1", how="semi")
    p = pr.join(exp, on="s1", how="semi").join(orph, on="m", how="left")
    g = s1f.join(p.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"), (~pl.col("y") & pl.col("orph_s1").is_not_null()).sum().alias("ofp")), on="s1", how="left").fill_null(0)
    F = f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float))
    Fnofp = f05(g["tp"].to_numpy().astype(float), 0 * g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float))
    tot["S1"] += s1f.height; tot["loss"] += float((1 - F).sum()); tot["loss_fp_only"] += float((Fnofp - F).sum())
    tot["orph_fp"] += int(g["ofp"].sum()); tot["other_fp"] += int((g["fp"] - g["ofp"]).sum()); tot["fn"] += int((g["n"] - g["tp"]).clip(0).sum())
    ex.append(p.filter(~pl.col("y") & pl.col("orph_s1").is_not_null()).head(6).select("s1", "m", "orph_s1"))
print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in tot.items()}, " mean F", round(1 - tot["loss"] / tot["S1"], 5), " FP share of loss", round(tot["loss_fp_only"] / tot["loss"], 3))
ex = pl.concat(ex).head(8)
ids = pl.concat([ex.select(pl.col("s1").alias("entity_id")), ex.select(pl.col("m").alias("entity_id")), ex.select(pl.col("orph_s1").alias("entity_id"))]).unique()
raw = pl.concat([pl.scan_parquet(f"work/train_s{k}.parquet") for k in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
R = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
for s1, m, o in ex.iter_rows():
    print(f"S1 {R[s1][0]} | {R[s1][1]}\n   accepted orphan copy {R[m][0]} | {R[m][1]}\n   (true owner, dropped S1: {R[o][0]} | {R[o][1]})")
