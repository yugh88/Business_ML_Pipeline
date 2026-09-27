"""Stage-3 (post-stage-2 collective re-reasoning) — train on OOF scores, tune on fold 7, report holdout, APPLY TO TEST,
write a submission and label-free test diagnostics.
usage: 173_stage3_apply.py <p2_dir> <decision.json> <base_candidate_pairs.tsv> <out_dir> [train_folds=1,2,3,4]"""
import sys, json, os, gc, shutil
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from tune_decision import prep, decide
p2dir, decf, base_cand, outdir = sys.argv[1:5]
TRF = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,3,4,5,6").split(",")]   # 174: folds 1-6 best
dec = json.load(open(decf)); V = dec["variant"]; TAU = dec["tau"]
os.makedirs(outdir, exist_ok=True)
TOK = r"([^\s,]*\d[^\s,]*)"
wc = json.load(open("aws/scripts/word_classes.json")); DESC, NOISE = list(set(wc["desc"])), list(set(wc["noise"]))
HAS_AUG = os.path.exists("work/aug_v8_s2.parquet")
_sch = pl.read_parquet_schema(sorted(__import__("glob").glob(f"{p2dir}/train/*.parquet"))[0])
EXTRA_SCORES = [c for c in ("p2", "p2x", "p2x2") if c != V and c in _sch] if os.environ.get("S3_MULTI", "1") == "1" else []   # 178


def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


_MM = {}
def mmax(split):
    """GLOBAL best stage-2 score of each pool record over all S1 of the split (fold-local max under-counts competition)."""
    if split not in _MM:
        _MM[split] = pl.scan_parquet(f"{p2dir}/{split}/*.parquet").filter(pl.col("p1") >= TAU).group_by("m").agg(pl.col(V).max().alias("m_max_g")).collect()
    return _MM[split]


def build(split, filt):
    cols = ["s1", "m", "p1", "pc2", "pa", V] + EXTRA_SCORES + (["y", "fold"] if split == "train" else [])
    d = pl.scan_parquet(f"{p2dir}/{split}/*.parquet").filter(filt & (pl.col("p1") >= TAU)).select(cols).collect() \
          .rename({c: f"s_{c}" for c in EXTRA_SCORES})
    if split == "test":
        d = d.with_columns(pl.lit(False).alias("y"), pl.lit(-1).alias("fold"))
    ids = d.select(pl.col("m").alias("entity_id")).unique(); ids1 = d.select(pl.col("s1").alias("entity_id")).unique()
    rawp = [pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", "business_address") for s in (2, 3)]
    normp = [pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "nums") for s in (1, 2, 3)]
    if split == "train" and HAS_AUG:
        rawp += [pl.scan_parquet(f"work/aug_v8_{s}.parquet").select("entity_id", "business_address") for s in ("s2", "s3")]
        normp += [pl.scan_parquet("work/aug_v8_norm.parquet").select("entity_id", "ncore", "nums")]
    raw = pl.concat(rawp).join(ids.lazy(), on="entity_id", how="semi").collect().select(pl.col("entity_id").alias("m"), pl.col("business_address").fill_null("").str.to_lowercase().alias("ra"))
    s1r = pl.scan_parquet(f"work/{split}_s1.parquet").join(ids1.lazy(), on="entity_id", how="semi").collect() \
            .select(pl.col("entity_id").alias("s1"), pl.col("business_address").fill_null("").str.to_lowercase().str.extract(TOK, 1).fill_null("").alias("tok1"))
    nm = pl.concat(normp).join(pl.concat([ids, ids1]).lazy(), on="entity_id", how="semi").collect() \
           .with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("h")).select("entity_id", "ncore", "h")
    d = d.join(raw, on="m", how="left").with_columns(pl.col("ra").fill_null(""), pl.col("m").str.slice(0, 2).alias("src")).join(s1r, on="s1", how="left") \
         .join(nm.rename({"entity_id": "s1", "h": "h1", "ncore": "c1"}), on="s1", how="left").join(nm.rename({"entity_id": "m", "h": "h2", "ncore": "c2"}), on="m", how="left")
    d = d.with_columns(pl.col("ra").str.extract(TOK, 1).fill_null("").alias("tok"), *[pl.col(c).fill_null("") for c in ("h1", "h2", "tok1", "c1", "c2")])
    st = (pl.col(V) >= 0.9).cast(pl.Float32)
    d = d.with_columns(st.alias("_st"))
    d = d.with_columns((pl.col("_st").sum().over(["s1", "src", "ra"]) - pl.col("_st")).alias("fp_addr_src"),
                       (pl.col("_st").sum().over(["s1", "src", "tok"]) - pl.col("_st")).alias("fp_tok_src"),
                       (pl.col("_st").sum().over(["s1", "tok"]) - pl.col("_st").sum().over(["s1", "src", "tok"])).alias("fp_tok_osrc"),
                       ((pl.col("tok") == pl.col("tok1")) & (pl.col("tok") != "")).cast(pl.Int8).alias("tok_eq_s1"),
                       (pl.col("ra") == "").cast(pl.Int8).alias("ra_empty"))
    d = d.with_columns(pl.when(pl.col("tok") == "").then(-1.0).otherwise(pl.col("fp_tok_src")).alias("fp_tok_src"),
                       pl.when(pl.col("ra") == "").then(-1.0).otherwise(pl.col("fp_addr_src")).alias("fp_addr_src"))
    dd = pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)
    d = d.join(mmax(split), on="m", how="left")
    d = d.with_columns(dd.clip(-100, 100).fill_null(-999).alias("sdiff"), pl.col(V).rank("ordinal", descending=True).over("s1").alias("rank_s1"),
                       pl.col("m_max_g").fill_null(pl.col(V)).alias("m_max"), pl.col("_st").sum().over("s1").alias("n_strong")).drop("m_max_g")
    E = pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" "))
    M = pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" "))
    d = d.with_columns(E.list.eval(pl.element().is_in(DESC)).list.any().cast(pl.Int8).alias("x_desc"),
                       (E.list.len().gt(0) & E.list.eval(pl.element().is_in(NOISE)).list.all()).cast(pl.Int8).alias("x_noise"),
                       E.list.len().alias("n_extra"), M.list.len().alias("n_missing"), (pl.col(V) - pl.col("p1")).alias("promo"))
    w = pl.when(pl.col(V) >= 0.5).then(pl.col(V)).otherwise(0.0)
    isown = (pl.col("h2") == pl.col("h1")) & (pl.col("h1") != "")
    d = d.with_columns(w.alias("_w"), pl.when(isown).then(w).otherwise(0.0).alias("_wo"))
    d = d.with_columns((pl.col("_wo").sum().over(["s1", "src"]) - pl.col("_wo")).alias("own_same_src"),
                       (pl.col("_wo").sum().over("s1") - pl.col("_wo").sum().over(["s1", "src"])).alias("own_other_src"),
                       pl.when(pl.col("h2") == "").then(-1.0).otherwise(pl.col("_w").sum().over(["s1", "h2"]) - pl.col("_w")).alias("grp_w"),
                       pl.when(pl.col("h2") == "").then(-1).otherwise(pl.col("src").filter(pl.col("_w") > 0).n_unique().over(["s1", "h2"])).alias("grp_nsrc"),
                       pl.col("h2").filter((pl.col("_w") > 0) & (pl.col("h2") != "")).n_unique().over("s1").alias("n_numgroups"))
    return d.drop("c1", "c2", "_w", "_wo", "_st", "ra", "tok", "tok1").rename({V: "p2"})


F = ["p2", "p1", "fp_addr_src", "fp_tok_src", "fp_tok_osrc", "tok_eq_s1", "ra_empty", "sdiff", "rank_s1", "m_max", "n_strong",
     "x_desc", "x_noise", "n_extra", "n_missing", "promo", "own_same_src", "own_other_src", "grp_w", "grp_nsrc", "n_numgroups"] \
    + [f"s_{c}" for c in EXTRA_SCORES]
X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy()
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else "
                              "len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
tr = pl.concat([build("train", pl.col("fold") == f) for f in TRF])
print("stage-3 train rows", tr.height, "folds", TRF, flush=True)
model = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1, num_threads=6),
                  lgb.Dataset(X(tr), tr["y"].to_numpy()), 700)
model.save_model(os.path.join(outdir, "stage3.txt"))
imp = sorted(zip(F, model.feature_importance("gain")), key=lambda z: -z[1]); print("importance:", [(a, round(b / 1e3)) for a, b in imp[:12]], flush=True)
del tr; gc.collect()


def decide_p3(d, p, c):
    x = d.select("s1", "m", "y", "fold", "p1", "pc2", "pa").with_columns(pl.Series("p3", p))
    return decide(prep(x, TAU, "p3"), c)


def score(pred, d, s1f):
    x = pred.join(d.select("s1", "m", "y"), on=["s1", "m"])
    g = s1f.join(x.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean())


P3 = os.path.join(outdir, "p3"); os.makedirs(os.path.join(P3, "train"), exist_ok=True); os.makedirs(os.path.join(P3, "test"), exist_ok=True)
def save_p3(d, p, path):
    cols = ["s1", "m", "y", "fold", "p1", "pc2", "pa"] + (["country"] if "country" in d.columns else [])
    d.select(cols).with_columns(pl.Series("p3", p.astype(np.float32))).write_parquet(path)
va = build("train", pl.col("fold") == 7); pv = model.predict(X(va)); s17 = gt.filter(pl.col("fold") == 7)
save_p3(va, pv, os.path.join(P3, "train", "fold7.parquet"))
grid = [dict(policy="G_expected_f", a=a, lam=l, cap=cap) for a in (0.8, 1.0, 1.25, 1.5) for l in (0.0, 0.05, 0.15, 0.3) for cap in (0, 1)]
res = sorted(((score(decide_p3(va, pv, c), va, s17), c) for c in grid), key=lambda z: -z[0])
best = res[0][1]; print("fold7 best", round(res[0][0], 5), best, " base-policy on p2:", round(score(decide(prep(va.rename({"p2": V}), TAU, V), dec), va, s17), 5), flush=True)
del va; gc.collect()
report = {"train_folds": TRF, "policy": best, "valid": res[0][0], "holdout": {}}
for f in (0, 8, 9):
    d = build("train", pl.col("fold") == f); s1f = gt.filter(pl.col("fold") == f)
    p3h = model.predict(X(d)); save_p3(d, p3h, os.path.join(P3, "train", f"fold{f}.parquet"))
    b = score(decide(prep(d.rename({"p2": V}), TAU, V), dec), d, s1f); s3 = score(decide_p3(d, p3h, best), d, s1f)
    report["holdout"][f] = dict(base=round(b, 5), stage3=round(s3, 5)); print(f"holdout fold {f}: base {b:.5f} stage3 {s3:.5f}", flush=True)
    del d; gc.collect()
# ---- test
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1 = dict(s1c.group_by("country").len().rows())
preds = []; report["test"] = {}
for c in ["US", "India", "France"]:
    d = build("test", pl.col("country") == c)
    p3t = model.predict(X(d)); save_p3(d.with_columns(pl.lit(c).alias("country")), p3t, os.path.join(P3, "test", f"{c}.parquet"))
    pr = decide_p3(d, p3t, best)
    a1 = decide(prep(d.select("s1", "m", "y", "fold", "p1", "pc2", "pa"), 0.05, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).with_columns(pl.lit(True).alias("a1"))
    x = pr.join(d.select("s1", "m", "sdiff"), on=["s1", "m"]).join(a1, on=["s1", "m"], how="left")
    n = nS1[c] / 1000
    report["test"][c] = dict(pred_per_s1=round(pr.height / nS1[c], 4), plus_k=round(float(x["sdiff"].is_in([1, 2, 3, 4, 5, 6, 7, 8, 9]).sum()) / n, 2),
                             minus_k=round(float(x["sdiff"].is_in([-1, -2, -3, -4, -5, -6, -7, -8, -9]).sum()) / n, 2),
                             promoted=round(float(x["a1"].is_null().sum()) / n, 2))
    print("TEST", c, report["test"][c], flush=True)
    preds.append(pr.select("s1", "m")); del d, pr, a1, x; gc.collect()
pred = pl.concat(preds)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = pred.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id") \
     .write_csv(os.path.join(outdir, "matching_results.tsv"), separator="\t", quote_style="never")
shutil.copy(base_cand, os.path.join(outdir, "candidate_pairs.tsv"))
report["test_pairs"] = pred.height
json.dump(report, open(os.path.join(outdir, "stage3_report.json"), "w"), indent=1, default=str)
json.dump(dict(tau=TAU, variant="p3", **{k: v for k, v in best.items()}), open(os.path.join(outdir, "decision_s3.json"), "w"))
print("written", outdir, pred.height)
