"""User idea: dedicated per-country models + ensemble. Stage-3 level (same features as V9's stage-3, trained on folds 1-6):
global (V9-like) vs US-only / India-only vs ensemble (mean of global and country model). G policy tuned on fold 7 per variant.
Holdout folds 0/8/9 (clean: synthetic records excluded from scoring) reported PER COUNTRY."""
import sys, os, gc, json
sys.argv = [sys.argv[0], "work/v8_p2", "analysis/aws_v8/decision_v8.json", "output_v9/candidate_pairs.tsv", "work/s3_country_tmp"]
src = open("scripts/173_stage3_apply.py").read()
exec(src.split("gt = duckdb.connect()")[0])                      # imports, build(), F, X, f05
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from tune_decision import prep, decide
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else "
                              "len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
gt = gt.join(t1, on="s1")
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.concat([build("train", pl.col("fold") == f) for f in TRF]).join(t1, on="s1")
print("train rows", tr.height, dict(tr.group_by("country").len().rows()), flush=True)
prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1, num_threads=6)
models = {"global": lgb.train(prm, lgb.Dataset(X(tr), tr["y"].to_numpy()), 700)}
for c in ("US", "India"):
    z = tr.filter(pl.col("country") == c)
    models[c] = lgb.train(prm, lgb.Dataset(X(z), z["y"].to_numpy()), 700); print("trained", c, z.height, flush=True)
del tr; gc.collect()


def preds(d):
    pg = models["global"].predict(X(d)); pc = pg.copy()
    for c in ("US", "India"):
        msk = (d["country"] == c).to_numpy()
        if msk.any():
            pc[msk] = models[c].predict(X(d.filter(pl.col("country") == c)))
    return {"global": pg, "country": pc, "ensemble": 0.5 * (pg + pc)}


def dec_score(d, p, c, s1f):
    x = d.select("s1", "m", "y", "p1", "pc2", "pa").with_columns(pl.Series("p3", p))
    a = decide(prep(x, TAU, "p3"), c).join(d.select("s1", "m", "y"), on=["s1", "m"]).join(SYN, on="m", how="anti")
    z = s1f.join(a.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return z.with_columns(pl.Series("f", f05(z["tp"].to_numpy().astype(float), z["fp"].to_numpy().astype(float), z["n"].to_numpy().astype(float))))


va = build("train", pl.col("fold") == 7).join(t1, on="s1"); pv = preds(va); s17 = gt.filter(pl.col("fold") == 7)
grid = [dict(policy="G_expected_f", a=a, lam=l, cap=1) for a in (1.0, 1.25, 1.5) for l in (0.0, 0.05, 0.15)]
best = {}
for k, p in pv.items():
    r = sorted(((float(dec_score(va, p, c, s17)["f"].mean()), c) for c in grid), key=lambda z: -z[0])
    best[k] = r[0][1]; print(f"fold7 {k}: best {r[0][0]:.5f} {r[0][1]}", flush=True)
del va; gc.collect()
out = {k: {"US": [], "India": []} for k in pv}
for f in (0, 8, 9):
    d = build("train", pl.col("fold") == f).join(t1, on="s1"); s1f = gt.filter(pl.col("fold") == f); ph = preds(d)
    for k, p in ph.items():
        z = dec_score(d, p, best[k], s1f)
        for c in ("US", "India"):
            out[k][c].append(float(z.filter(pl.col("country") == c)["f"].mean()))
    del d; gc.collect()
print("\nHOLDOUT clean F by country (folds 0/8/9 mean) — global = V9-like stage-3")
for k in out:
    print(f"  {k:9s} US {np.mean(out[k]['US']):.5f} ({np.mean(out[k]['US']) - np.mean(out['global']['US']):+.5f})  "
          f"India {np.mean(out[k]['India']):.5f} ({np.mean(out[k]['India']) - np.mean(out['global']['India']):+.5f})   per fold US {[round(v, 5) for v in out[k]['US']]} India {[round(v, 5) for v in out[k]['India']]}")
json.dump({"best": {k: v for k, v in best.items()}, "holdout": out}, open("analysis/out/215_country_models.json", "w"), indent=1, default=str)
for k, m in models.items():
    m.save_model(f"work/s3_{k}.txt")
