"""(v2: + promotion / word classes / p2-weighted number groups) Does the same-source FINGERPRINT (raw address / raw house token shared with other strong same-source candidates) add
signal beyond the stage-2 score? Stage-3 LightGBM on out-of-fold p2 (folds 1-3 train, fold 7 decision tuning,
folds 0/8/9 holdout).  usage: 171_stage3_fp.py <p2_dir> <decision.json>"""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from tune_decision import prep, decide, configs
p2dir, decf = sys.argv[1:3]
dec = json.load(open(decf)); V = dec["variant"]; TAU = dec["tau"]
TOK = r"([^\s,]*\d[^\s,]*)"
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else "
                              "len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")


def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def feats(f):
    d = pl.scan_parquet(f"{p2dir}/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", V).collect()
    d = d.filter(pl.col("p1") >= TAU)
    ids = d.select(pl.col("m").alias("entity_id")).unique()
    rawp = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet").select("entity_id", "business_address") for s in (2, 3)])
    try:
        rawp = pl.concat([rawp, pl.scan_parquet("work/aug_v8_s2.parquet").select("entity_id", "business_address"), pl.scan_parquet("work/aug_v8_s3.parquet").select("entity_id", "business_address")])
    except Exception:
        pass
    raw = rawp.join(ids.lazy(), on="entity_id", how="semi").collect().select(pl.col("entity_id").alias("m"), pl.col("business_address").fill_null("").str.to_lowercase().alias("ra"))
    s1r = pl.scan_parquet("work/train_s1.parquet").join(d.select(pl.col("s1").alias("entity_id")).unique().lazy(), on="entity_id", how="semi").collect() \
            .select(pl.col("entity_id").alias("s1"), pl.col("business_address").fill_null("").str.to_lowercase().str.extract(TOK, 1).fill_null("").alias("tok1"))
    nums = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "nums") for s in (1, 2, 3)] + [pl.scan_parquet("work/aug_v8_norm.parquet").select("entity_id", "nums")]) \
             .join(pl.concat([ids, d.select(pl.col("s1").alias("entity_id")).unique()]).lazy(), on="entity_id", how="semi").collect() \
             .select("entity_id", pl.col("nums").str.split(" ").list.first().fill_null("").alias("h"))
    d = d.join(raw, on="m", how="left").with_columns(pl.col("ra").fill_null(""), pl.col("m").str.slice(0, 2).alias("src")).join(s1r, on="s1", how="left") \
         .join(nums.rename({"entity_id": "s1", "h": "h1"}), on="s1", how="left").join(nums.rename({"entity_id": "m", "h": "h2"}), on="m", how="left")
    d = d.with_columns(pl.col("ra").str.extract(TOK, 1).fill_null("").alias("tok"), pl.col("h1").fill_null(""), pl.col("h2").fill_null(""), pl.col("tok1").fill_null(""))
    strong = (pl.col(V) >= 0.9).cast(pl.Float32)
    d = d.with_columns(strong.alias("_st"))
    d = d.with_columns((pl.col("_st").sum().over(["s1", "src", "ra"]) - pl.col("_st")).alias("fp_addr_src"),
                       (pl.col("_st").sum().over(["s1", "src", "tok"]) - pl.col("_st")).alias("fp_tok_src"),
                       (pl.col("_st").sum().over(["s1", "tok"]) - pl.col("_st").sum().over(["s1", "src", "tok"])).alias("fp_tok_osrc"),
                       ((pl.col("tok") == pl.col("tok1")) & (pl.col("tok") != "")).cast(pl.Int8).alias("tok_eq_s1"),
                       (pl.col("ra") == "").cast(pl.Int8).alias("ra_empty"))
    d = d.with_columns(pl.when(pl.col("tok") == "").then(-1.0).otherwise(pl.col("fp_tok_src")).alias("fp_tok_src"),
                       pl.when(pl.col("ra") == "").then(-1.0).otherwise(pl.col("fp_addr_src")).alias("fp_addr_src"))
    dd = pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)
    d = d.with_columns(dd.clip(-100, 100).fill_null(-999).alias("sdiff"),
                       pl.col(V).rank("ordinal", descending=True).over("s1").alias("rank_s1"), pl.col(V).max().over("m").alias("m_max"),
                       pl.col("_st").sum().over("s1").alias("n_strong"))
    # --- v2 additions: promotion, word classes, name relation, p2-weighted house-number groups by source
    wc = json.load(open("aws/scripts/word_classes.json")); DESC, NOISE = set(wc["desc"]), set(wc["noise"])
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique()
    nm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "ncore") for s in (1, 2, 3)] + [pl.scan_parquet("work/aug_v8_norm.parquet").select("entity_id", "ncore")]) \
           .join(pl.concat([ids, ids1]).lazy(), on="entity_id", how="semi").collect()
    d = d.join(nm.rename({"entity_id": "s1", "ncore": "c1"}), on="s1", how="left").join(nm.rename({"entity_id": "m", "ncore": "c2"}), on="m", how="left") \
         .with_columns(pl.col("c1").fill_null(""), pl.col("c2").fill_null(""))
    E = pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" "))
    M = pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" "))
    d = d.with_columns(E.list.eval(pl.element().is_in(list(DESC))).list.any().cast(pl.Int8).alias("x_desc"),
                       (E.list.len().gt(0) & E.list.eval(pl.element().is_in(list(NOISE))).list.all()).cast(pl.Int8).alias("x_noise"),
                       E.list.len().alias("n_extra"), M.list.len().alias("n_missing"), (pl.col(V) - pl.col("p1")).alias("promo"))
    w = pl.when(pl.col(V) >= 0.5).then(pl.col(V)).otherwise(0.0)
    isown = (pl.col("h2") == pl.col("h1")) & (pl.col("h1") != "")
    d = d.with_columns(w.alias("_w"), pl.when(isown).then(w).otherwise(0.0).alias("_wo"))
    d = d.with_columns((pl.col("_wo").sum().over(["s1", "src"]) - pl.col("_wo")).alias("own_same_src"),
                       (pl.col("_wo").sum().over("s1") - pl.col("_wo").sum().over(["s1", "src"])).alias("own_other_src"),
                       pl.when(pl.col("h2") == "").then(-1.0).otherwise(pl.col("_w").sum().over(["s1", "h2"]) - pl.col("_w")).alias("grp_w"),
                       pl.when(pl.col("h2") == "").then(-1).otherwise(pl.col("src").filter(pl.col("_w") > 0).n_unique().over(["s1", "h2"])).alias("grp_nsrc"),
                       pl.col("h2").filter((pl.col("_w") > 0) & (pl.col("h2") != "")).n_unique().over("s1").alias("n_numgroups"))
    return d.drop("c1", "c2", "_w", "_wo")


F = ["p2", "p1", "fp_addr_src", "fp_tok_src", "fp_tok_osrc", "tok_eq_s1", "ra_empty", "sdiff", "rank_s1", "m_max", "n_strong"]
F0 = ["p2", "p1", "sdiff", "rank_s1", "m_max", "n_strong"]            # same without fingerprints (ablation)
F2 = F + ["x_desc", "x_noise", "n_extra", "n_missing", "promo", "own_same_src", "own_other_src", "grp_w", "grp_nsrc", "n_numgroups"]
tr = pl.concat([feats(f) for f in (1, 2, 3)]).rename({V: "p2"})
va = feats(7).rename({V: "p2"})
X = lambda d, cols: d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
models = {}
for name, cols in (("with_fp", F), ("no_fp", F0), ("full", F2)):
    models[name] = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1),
                             lgb.Dataset(X(tr, cols), tr["y"].to_numpy()), 600)
    imp = sorted(zip(cols, models[name].feature_importance("gain")), key=lambda z: -z[1]); print(name, [(a, round(b / 1e3)) for a, b in imp])
del tr


def score(d, s1f, p, c):
    x = d.with_columns(pl.Series(V, p))
    pr = decide(prep(x.select("s1", "m", "y", "fold", "p1", "pc2", "pa", V), TAU, V), c).join(x.select("s1", "m", "y"), on=["s1", "m"])
    g = s1f.join(pr.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean())


best = {}
s17 = gt.filter(pl.col("fold") == 7)
for name, cols in (("base", None), ("with_fp", F), ("no_fp", F0), ("full", F2)):
    p = va["p2"].to_numpy() if cols is None else models[name].predict(X(va, cols))
    res = [(score(va, s17, p, c), c) for c in configs() if c["policy"] == "G_expected_f"]
    best[name] = max(res, key=lambda z: z[0]); print("fold7", name, round(best[name][0], 5), best[name][1], flush=True)
del va
for f in (0, 8, 9):
    d = feats(f).rename({V: "p2"}); s1f = gt.filter(pl.col("fold") == f)
    line = f"holdout fold {f}:"
    for name, cols in (("base", None), ("with_fp", F), ("no_fp", F0), ("full", F2)):
        p = d["p2"].to_numpy() if cols is None else models[name].predict(X(d, cols))
        line += f" {name} {score(d, s1f, p, best[name][1]):.5f}"
    print(line, flush=True)
