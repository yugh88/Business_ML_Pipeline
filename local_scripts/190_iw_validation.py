"""Importance-weighted (covariate-shift) validation: can a holdout reweighted to the TEST distribution of S1 neighbourhoods
reproduce the LB ordering V5 (0.967) < V5s3r (0.973) < V9 (0.977)?
S1 features come from the raw candidate structure (V5 candidates p1>=0.02; model-independent relations), a domain classifier
(holdout S1 = 0, test S1 = 1, per country, cross-fitted) gives w = p/(1-p) * n_hold/n_test, and every version's per-S1 F0.5 on the
holdout (clean view) is averaged with w."""
import sys, json, functools
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
exec(open("scripts/130_rules_v7.py").read().split("gt = duckdb.connect()")[0])       # enrich(), RULES, f05
con = duckdb.connect()


def s1_features(split, filt, country):
    d = pl.scan_parquet(f"work/v5_p2/{split}/*.parquet").filter(filt & (pl.col("p1") >= 0.02) & (pl.col("country") == country)).select("s1", "m").collect()
    ids = pl.concat([d.select(pl.col("s1").alias("entity_id")), d.select(pl.col("m").alias("entity_id"))]).unique()
    n = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in n.iter_rows()}
    nrel = [name_rel(N[a][0], N[b][0]) for a, b in zip(d["s1"], d["m"])]
    arel = [addr_rel(N[a][1], N[b][1], N[a][2], N[b][2]) for a, b in zip(d["s1"], d["m"])]
    h1 = [(N[a][2].split() or [""])[0] for a in d["s1"]]; h2 = [(N[b][2].split() or [""])[0] for b in d["m"]]
    off = []
    for a, b in zip(h1, h2):
        if not a or not b or not a.isdigit() or not b.isdigit(): off.append(0 if (a and a == b) else 9); continue
        k = int(b[:9]) - int(a[:9]); off.append(0 if k == 0 else (1 if 1 <= k <= 99 else (-1 if -99 <= k <= -1 else 2)))
    d = d.with_columns(pl.Series("nrel", nrel), pl.Series("arel", arel), pl.Series("off", off), pl.col("m").str.slice(0, 2).alias("src"))
    ident = pl.col("nrel") == "N0_identical"; heavy = pl.col("nrel").is_in(["N3_extra_other", "N6_word_swap", "N7_no_overlap"])
    same = pl.col("arel").is_in(["A0_identical", "A1_same_num"]); desc = pl.col("nrel").is_in(["N2_extra_DESCRIPTOR", "N6_word_swap_DESC"])
    f = d.group_by("s1").agg(pl.len().alias("n_cand"), ident.sum().alias("n_ident"), (ident & same).sum().alias("n_ident_same"), (ident & (pl.col("off") == 1)).sum().alias("n_twin_plus"),
                             (ident & (pl.col("off") == -1)).sum().alias("n_twin_minus"), (ident & (pl.col("arel") == "A3_empty")).sum().alias("n_ident_empty"),
                             desc.sum().alias("n_desc"), (heavy & same).sum().alias("n_heavy_same"), (heavy & ~same).sum().alias("n_heavy_other"),
                             (pl.col("arel").str.starts_with("A2")).sum().alias("n_numchg"), pl.col("src").n_unique().alias("n_src"),
                             pl.col("nrel").filter(ident).n_unique().alias("dummy"))
    s1n = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select(pl.col("entity_id").alias("s1"), "ncore", "nums").join(f.select("s1").lazy(), on="s1", how="semi").collect()
    return f.drop("dummy").join(s1n.with_columns((pl.col("nums") == "").cast(pl.Int8).alias("s1_nonum")).select("s1", "ncore", "s1_nonum"), on="s1")


gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                 "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
hold = gt.filter(pl.col("fold").is_in([0, 8, 9])).join(t1, on="s1")
# ---------------- per-S1 F of each version on the holdout (clean)
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
def per_s1_f(pred, truthy):
    x = pred.join(truthy, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(False))
    g = hold.join(x.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return g.select("s1", pl.Series("f", f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float))))
truth = pl.read_parquet("work/train_pairs.parquet").select("s1", "m").with_columns(pl.lit(True).alias("y"))
d5 = json.load(open("analysis/aws_v5/decision_v5.json")); ds3 = json.load(open("output_v5s3/decision_s3.json")); d9 = json.load(open("output_v9/decision_s3.json"))
preds = {"V5": [], "V5s3r": [], "V9": []}
for f in (0, 8, 9):
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    preds["V5"].append(decide(prep(d, d5["tau"], "p2x2"), d5).select("s1", "m"))
    p5 = pl.read_parquet(f"output_v5s3/p3/train/fold{f}.parquet")
    a = decide(prep(p5, ds3["tau"], "p3"), ds3).select("s1", "m").join(p5.select("s1", "m", "y"), on=["s1", "m"]).join(t1, on="s1")
    x = enrich(a, "train"); rm = functools.reduce(lambda u, v: u | v, [RULES[k](x) for k in ("R1_twin_same_src", "R2_twin_2src")])
    preds["V5s3r"].append(x.filter(~rm).select("s1", "m"))
    p9 = pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet").join(SYN, on="m", how="anti")
    preds["V9"].append(decide(prep(p9, d9["tau"], "p3"), d9).select("s1", "m"))
F = {k: per_s1_f(pl.concat(v), truth) for k, v in preds.items()}
print("unweighted holdout F:", {k: round(float(v["f"].mean()), 5) for k, v in F.items()}, flush=True)
# ---------------- domain classifier per country
FEAT = ["n_cand", "n_ident", "n_ident_same", "n_twin_plus", "n_twin_minus", "n_ident_empty", "n_desc", "n_heavy_same", "n_heavy_other", "n_numchg", "n_src", "s1_nonum", "chain"]
out = {}
for c in ("US", "India"):
    hf = s1_features("train", pl.col("fold").is_in([0, 8, 9]), c).join(hold.select("s1"), on="s1", how="semi")
    tf = s1_features("test", pl.lit(True), c)
    allc = pl.concat([hf.select("ncore"), tf.select("ncore")])
    hf = hf.join(hf.group_by("ncore").len().rename({"len": "chain"}), on="ncore"); tf = tf.join(tf.group_by("ncore").len().rename({"len": "chain"}), on="ncore")
    X = np.vstack([hf.select(FEAT).to_numpy(), tf.select(FEAT).to_numpy()]).astype(np.float32); yd = np.r_[np.zeros(hf.height), np.ones(tf.height)]
    pd_ = np.zeros(len(yd)); fold = np.random.default_rng(0).integers(0, 5, len(yd))
    for k in range(5):
        m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=500, verbose=-1, seed=k), lgb.Dataset(X[fold != k], yd[fold != k]), 200)
        pd_[fold == k] = m.predict(X[fold == k])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yd, pd_)
    ph = np.clip(pd_[:hf.height], 1e-4, 1 - 1e-4); w = ph / (1 - ph) * (hf.height / tf.height)
    wdf = hf.select("s1").with_columns(pl.Series("w", w))
    ess = w.sum() ** 2 / (w ** 2).sum()
    imp = sorted(zip(FEAT, m.feature_importance("gain")), key=lambda z: -z[1])[:6]
    res = {}
    for k, v in F.items():
        j = v.join(wdf, on="s1")
        res[k] = (round(float(j["f"].mean()), 5), round(float((j["f"] * j["w"]).sum() / j["w"].sum()), 5))
    out[c] = res
    print(f"{c}: domain AUC {auc:.3f}  effective sample {ess:.0f} of {hf.height}  top shift features {[(a, round(b)) for a, b in imp]}")
    print(f"   version: (unweighted, TEST-WEIGHTED) holdout F0.5 = {res}", flush=True)
json.dump(out, open("analysis/out/190_iw_validation.json", "w"), indent=1)
print("LB: V5 0.967  V5s3r 0.973  V9 0.977 (incl. France)")
