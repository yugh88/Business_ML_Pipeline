"""S2<->S3 relational signal (within each S1's cascade candidate set), tested as an ADDITION to stage-3.
Pool-pool similarity (name = token_set_ratio on normalized core name; address = token_set_ratio on normalized address;
house-number equality) between a candidate and the OTHER-source (xs_*) / SAME-source (ws_*) candidates of the same S1,
weighted by how strongly those partners are matched to the S1 (OOF p2) and whether they are owned by another S1.
Leakage-safe: similarities are unsupervised; p2 of partners is out-of-fold (folds 1-6) / fold-disjoint (0,7,8,9).
Protocol: stage-3 WITHOUT vs WITH xs/ws features — train folds 1-6, tune policy on fold 7, evaluate folds 0/8/9;
segments: chain names, orphan-exposed S1, twin/sibling-exposed S1, singletons; + label-free test diagnostics.
usage: 176_xsource.py <p2_dir> <decision.json> <cache_dir> <label>"""
import sys, json, os, gc
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from rapidfuzz import process, fuzz
p2dir, decf, cache, label = sys.argv[1:5]
src = open("scripts/173_stage3_apply.py").read()
g = {"__name__": "s3"}; sys.argv = ["x", p2dir, decf, "/dev/null", cache]
exec(src.split("tr = pl.concat([build(")[0], g)
build, F, f05, gt, TAU, V, prep, decide = g["build"], g["F"], g["f05"], g["gt"], g["TAU"], g["V"], g["prep"], g["decide"]
os.makedirs(cache, exist_ok=True)


def base_frame(split, key, filt):
    p = os.path.join(cache, f"{split}_{key}.parquet")
    if not os.path.exists(p):
        build(split, filt).write_parquet(p)
    return pl.read_parquet(p)


def add_xs(d):
    """relational features from pool-pool similarity inside each S1's candidate set."""
    c = d.select("s1", "m", "src", "pc2", "pa", "h2", "p2", "m_max").with_columns(pl.col("pc2").fill_null(""), pl.col("pa").fill_null(""), pl.col("h2").fill_null(""))
    pr = c.join(c, on="s1", suffix="_o").filter(pl.col("m") != pl.col("m_o"))
    ns = process.cpdist(pr["pc2"].to_list(), pr["pc2_o"].to_list(), scorer=fuzz.token_set_ratio, workers=6)
    asim = process.cpdist(pr["pa"].to_list(), pr["pa_o"].to_list(), scorer=fuzz.token_set_ratio, workers=6)
    pr = pr.with_columns(pl.Series("nsim", ns, dtype=pl.Float32), pl.Series("asim", asim, dtype=pl.Float32))
    pr = pr.with_columns(pl.when((pl.col("pa") == "") | (pl.col("pa_o") == "")).then(pl.col("nsim")).otherwise(pl.min_horizontal("nsim", "asim")).alias("both"),
                         ((pl.col("h2") != "") & (pl.col("h2") == pl.col("h2_o"))).alias("numeq"),
                         (pl.col("src") != pl.col("src_o")).alias("xs"), (pl.col("p2_o") >= 0.9).alias("strong"),
                         (pl.col("m_max_o") > pl.col("p2_o") + 0.1).alias("owned_else"))
    feats = []
    for tag, cond in (("xs", pl.col("xs")), ("ws", ~pl.col("xs"))):
        q = pr.filter(cond)
        a = q.group_by("s1", "m").agg(
            pl.col("both").filter(pl.col("strong")).max().alias(f"{tag}_both_strong"),
            pl.col("nsim").filter(pl.col("strong")).max().alias(f"{tag}_name_strong"),
            (pl.col("both").filter(pl.col("strong") & pl.col("numeq")).max()).alias(f"{tag}_both_strong_numeq"),
            pl.col("both").max().alias(f"{tag}_both_any"),
            pl.col("p2_o").sort_by("both", descending=True).first().alias(f"{tag}_partner_p2"),
            pl.col("owned_else").sort_by("both", descending=True).first().cast(pl.Int8).alias(f"{tag}_partner_owned_else"),
            (pl.col("both") >= 90).sum().alias(f"{tag}_n90"),
            ((pl.col("both") >= 90) & ~pl.col("strong")).sum().alias(f"{tag}_n90_weak"))
        feats.append(a)
    out = d
    for a in feats:
        out = out.join(a, on=["s1", "m"], how="left")
    XS = [c for a in feats for c in a.columns if c not in ("s1", "m")]
    return out.with_columns([pl.col(c).fill_null(-1).cast(pl.Float32) for c in XS]), XS


def score(d, p, c, s1f, segment=None):
    x = d.select("s1", "m", "y", "fold", "p1", "pc2", "pa").with_columns(pl.Series("p3", p))
    pr = decide(prep(x, TAU, "p3"), c).join(d.select("s1", "m", "y"), on=["s1", "m"])
    gg = s1f.join(pr.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    fv = f05(gg["tp"].to_numpy().astype(float), gg["fp"].to_numpy().astype(float), gg["n"].to_numpy().astype(float))
    if segment is None:
        return float(fv.mean())
    gg = gg.with_columns(pl.Series("f", fv))
    out = {}
    for k, v in segment.items():
        sub = gg.join(v, on="s1", how="semi")
        out[k] = (round(float(sub["f"].mean()), 5) if sub.height else None, sub.height)
    return out


# ---- segments (holdout S1 sets)
s1n = pl.read_parquet("work/train_s1_norm.parquet", columns=["entity_id", "country", "ncore"]).rename({"entity_id": "s1"})
chain = s1n.join(s1n.group_by("country", "ncore").len().filter(pl.col("len") >= 5), on=["country", "ncore"], how="semi").select("s1")
dropped = pl.read_parquet("work/v5_dropped_s1.parquet").select("s1")
orph_copies = pl.read_parquet("work/train_pairs.parquet").join(dropped, on="s1", how="semi").select("m")
syn = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"}) if os.path.exists("work/aug_v8_s2.parquet") else pl.DataFrame({"m": []}, schema={"m": pl.Utf8})

Xc = lambda d, cols: d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
tr = pl.concat([add_xs(base_frame("train", f"fold{f}", pl.col("fold") == f))[0] for f in range(1, 7)])
XS = add_xs(base_frame("train", "fold7", pl.col("fold") == 7))[1]
print(label, "train rows", tr.height, "xs features", XS, flush=True)
models = {}
for name, cols in (("s3", F), ("s3+xs", F + XS)):
    models[name] = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1, num_threads=6),
                             lgb.Dataset(Xc(tr, cols), tr["y"].to_numpy()), 700)
imp = sorted(zip(F + XS, models["s3+xs"].feature_importance("gain")), key=lambda z: -z[1])
print("importance (s3+xs):", [(a, round(b / 1e3)) for a, b in imp[:16]], flush=True)
del tr; gc.collect()
GRID = [dict(policy="G_expected_f", a=a, lam=l, cap=1) for a in (1.0, 1.25, 1.5) for l in (0.0, 0.05)]
va = add_xs(base_frame("train", "fold7", pl.col("fold") == 7))[0]; s17 = gt.filter(pl.col("fold") == 7)
pol = {}
for name, cols in (("s3", F), ("s3+xs", F + XS)):
    pv = models[name].predict(Xc(va, cols))
    pol[name] = max(((score(va, pv, c, s17), c) for c in GRID), key=lambda z: z[0]); print("fold7", name, round(pol[name][0], 5), pol[name][1], flush=True)
del va; gc.collect()
report = {"label": label, "xs_features": XS, "valid": {k: v[0] for k, v in pol.items()}, "holdout": {}}
for f in (0, 8, 9):
    d = add_xs(base_frame("train", f"fold{f}", pl.col("fold") == f))[0]; s1f = gt.filter(pl.col("fold") == f)
    cand_s1 = lambda m_: d.join(m_, on="m", how="semi").select("s1").unique()
    seg = {"all": s1f.select("s1"), "chain": s1f.join(chain, on="s1", how="semi").select("s1"), "orphan_exposed": cand_s1(orph_copies),
           "twin_sib_exposed(synthetic)": cand_s1(syn), "singletons": s1f.filter(pl.col("n") == 0).select("s1")}
    r = {}
    for name, cols in (("s3", F), ("s3+xs", F + XS)):
        r[name] = score(d, models[name].predict(Xc(d, cols)), pol[name][1], s1f, seg)
    report["holdout"][f] = r
    print(f"fold {f}:", {seg_: (r["s3"][seg_], r["s3+xs"][seg_]) for seg_ in seg}, flush=True)
    del d; gc.collect()
# ---- test diagnostics (label-free)
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1 = dict(s1c.group_by("country").len().rows())
report["test"] = {}
for c in ["US", "India", "France"]:
    d = add_xs(base_frame("test", c, pl.col("country") == c))[0]
    a1 = decide(prep(d.select("s1", "m", "y", "fold", "p1", "pc2", "pa"), 0.05, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).with_columns(pl.lit(True).alias("a1"))
    rr = {}
    for name, cols in (("s3", F), ("s3+xs", F + XS)):
        x = d.select("s1", "m", "y", "fold", "p1", "pc2", "pa").with_columns(pl.Series("p3", models[name].predict(Xc(d, cols))))
        pr = decide(prep(x, TAU, "p3"), pol[name][1]).join(d.select("s1", "m", "sdiff"), on=["s1", "m"]).join(a1, on=["s1", "m"], how="left")
        n = nS1[c] / 1000
        rr[name] = dict(pred_per_s1=round(pr.height / nS1[c], 4), plus_k=round(float(pr["sdiff"].is_in(list(range(1, 10))).sum()) / n, 2),
                        minus_k=round(float(pr["sdiff"].is_in(list(range(-9, 0))).sum()) / n, 2), promoted=round(float(pr["a1"].is_null().sum()) / n, 2))
    report["test"][c] = rr; print("TEST", c, rr, flush=True)
    del d; gc.collect()
json.dump(report, open(f"analysis/out/176_xsource_{label}.json", "w"), indent=1, default=str)
