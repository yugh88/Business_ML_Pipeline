"""Stage-3 variants on cached per-fold features (from 173.build): training folds, capacity, name fingerprint, 2nd pass.
usage: 174_stage3_variants.py <p2_dir> <decision.json> <cache_dir>"""
import sys, json, os, gc
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
p2dir, decf, cache = sys.argv[1:4]
os.makedirs(cache, exist_ok=True)
src = open("scripts/173_stage3_apply.py").read()
g = {"__name__": "s3"}
sys.argv = ["x", p2dir, decf, "/dev/null", cache]
exec(src.split("tr = pl.concat([build(")[0], g)            # defines build, F, X, f05, gt, decide/prep imports, TAU, V
build, F, f05, gt, TAU, V, prep, decide = g["build"], g["F"], g["f05"], g["gt"], g["TAU"], g["V"], g["prep"], g["decide"]


def cached(f):
    p = os.path.join(cache, f"fold{f}.parquet")
    if not os.path.exists(p):
        d = build("train", pl.col("fold") == f)
        raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet").select("entity_id", "business_name") for s in (2, 3)] +
                        ([pl.scan_parquet(f"work/aug_v8_{s}.parquet").select("entity_id", "business_name") for s in ("s2", "s3")] if os.path.exists("work/aug_v8_s2.parquet") else [])) \
                .join(d.select(pl.col("m").alias("entity_id")).unique().lazy(), on="entity_id", how="semi").collect() \
                .select(pl.col("entity_id").alias("m"), pl.col("business_name").fill_null("").str.to_lowercase().alias("rn"))
        d = d.join(raw, on="m", how="left").with_columns(pl.col("rn").fill_null(""))
        st = (pl.col("p2") >= 0.9).cast(pl.Float32)
        d = d.with_columns(st.alias("_st")).with_columns((pl.col("_st").sum().over(["s1", "src", "rn"]) - pl.col("_st")).alias("fp_name_src"),
                                                         (pl.col("_st").sum().over(["s1", "rn"]) - pl.col("_st").sum().over(["s1", "src", "rn"])).alias("fp_name_osrc")).drop("_st", "rn")
        d.write_parquet(p); del d; gc.collect()
    return pl.read_parquet(p)


def score(d, p, c, s1f):
    x = d.select("s1", "m", "y", "fold", "p1", "pc2", "pa").with_columns(pl.Series("p3", p))
    pr = decide(prep(x, TAU, "p3"), c).join(d.select("s1", "m", "y"), on=["s1", "m"])
    gg = s1f.join(pr.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return float(f05(gg["tp"].to_numpy().astype(float), gg["fp"].to_numpy().astype(float), gg["n"].to_numpy().astype(float)).mean())


def second_pass(d, p):
    """recompute the p2-weighted group features from stage-3 probabilities (collective re-reasoning, iteration 2)."""
    x = d.with_columns(pl.Series("q", p))
    w = pl.when(pl.col("q") >= 0.5).then(pl.col("q")).otherwise(0.0)
    isown = (pl.col("h2") == pl.col("h1")) & (pl.col("h1") != "")
    x = x.with_columns(w.alias("_w"), pl.when(isown).then(w).otherwise(0.0).alias("_wo"))
    return x.with_columns((pl.col("_wo").sum().over(["s1", "src"]) - pl.col("_wo")).alias("own_same_src2"),
                          (pl.col("_wo").sum().over("s1") - pl.col("_wo").sum().over(["s1", "src"])).alias("own_other_src2"),
                          pl.when(pl.col("h2") == "").then(-1.0).otherwise(pl.col("_w").sum().over(["s1", "h2"]) - pl.col("_w")).alias("grp_w2"),
                          pl.col("q").rank("ordinal", descending=True).over("s1").alias("rank2"), pl.col("q").max().over("m").alias("m_max2")).drop("_w", "_wo")


GRID = [dict(policy="G_expected_f", a=a, lam=l, cap=1) for a in (1.0, 1.25, 1.5) for l in (0.0, 0.05)]
FN_ = F + ["fp_name_src", "fp_name_osrc"]
F2P = ["q", "own_same_src2", "own_other_src2", "grp_w2", "rank2", "m_max2"]
Xc = lambda d, cols: d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
VARS = {"A_f1-4": dict(folds=[1, 2, 3, 4], cols=F, leaves=63, rounds=700),
        "B_f1-6": dict(folds=[1, 2, 3, 4, 5, 6], cols=F, leaves=63, rounds=700),
        "C_f1-6_big": dict(folds=[1, 2, 3, 4, 5, 6], cols=F, leaves=127, rounds=1200),
        "D_f1-6_name": dict(folds=[1, 2, 3, 4, 5, 6], cols=FN_, leaves=127, rounds=1200)}
out = {}
va, s17 = cached(7), gt.filter(pl.col("fold") == 7)
hold = {f: cached(f) for f in (0, 8, 9)}
for name, cfg in VARS.items():
    tr = pl.concat([cached(f) for f in cfg["folds"]])
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=cfg["leaves"], min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1, num_threads=6),
                  lgb.Dataset(Xc(tr, cfg["cols"]), tr["y"].to_numpy()), cfg["rounds"])
    pv = m.predict(Xc(va, cfg["cols"]))
    best = max(((score(va, pv, c, s17), c) for c in GRID), key=lambda z: z[0])
    hs = [round(score(hold[f], m.predict(Xc(hold[f], cfg["cols"])), best[1], gt.filter(pl.col("fold") == f)), 5) for f in (0, 8, 9)]
    out[name] = dict(valid=round(best[0], 5), policy=best[1], holdout=hs); print(name, out[name], flush=True)
    if name == "D_f1-6_name":        # second pass on top of D
        tr2 = second_pass(tr, m.predict(Xc(tr, cfg["cols"])))
        m2 = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=2, num_threads=6),
                       lgb.Dataset(Xc(tr2, cfg["cols"] + F2P), tr2["y"].to_numpy()), 600)
        va2 = second_pass(va, pv); pv2 = m2.predict(Xc(va2, cfg["cols"] + F2P))
        best2 = max(((score(va2, pv2, c, s17), c) for c in GRID), key=lambda z: z[0])
        hs2 = []
        for f in (0, 8, 9):
            h2 = second_pass(hold[f], m.predict(Xc(hold[f], cfg["cols"])))
            hs2.append(round(score(h2, m2.predict(Xc(h2, cfg["cols"] + F2P)), best2[1], gt.filter(pl.col("fold") == f)), 5))
        out["E_D+second_pass"] = dict(valid=round(best2[0], 5), policy=best2[1], holdout=hs2); print("E_D+second_pass", out["E_D+second_pass"], flush=True)
        del tr2, va2
    del tr, m; gc.collect()
json.dump(out, open(os.path.join(cache, "variants.json"), "w"), indent=1, default=str)
