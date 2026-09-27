"""Shared dev-experiment helpers: splits, leaderboard-exact macro-F0.5, LightGBM fit, feature lists."""
import polars as pl, numpy as np, lightgbm as lgb
from rapidfuzz.distance import Levenshtein
NAME = ["n_tsr", "n_tsort", "n_ratio", "n_pr", "ns_jw", "ns_ratio", "ns_pr", "n_eq", "core_eq", "core_sorted_eq", "n_jac", "n_idf_cov",
        "n_max_idf_shared", "n_min_df1", "n_max_idf_missing", "n_ntok1", "n_ntok2", "n_shared", "n_len_ratio", "name_freq_s1", "dom2", "indic2", "brand2", "junk2"]
ADDR = ["a_tsr", "a_tsort", "a_pr", "a_eq", "a2_empty", "a_jac", "a_idf_cov", "a_max_idf_shared", "a_max_idf_missing", "a_ntok1", "a_ntok2", "a_shared",
        "num_shared", "num_n1", "num_n2", "num_first_eq", "num_conflict", "state_agree", "num_min_ed", "num_first_ed"]
COMP = ["rscore", "rrank", "m_best", "m_gap12", "r_margin_best", "s1_rank_rscore", "s1_n_core_eq", "s1_ncand", "s1_rank_sum"]
CTX = ["is_india", "is_s3"]
BASE = NAME + ADDR + COMP + CTX

def _close(a, b):
    A, B = a.split(), b.split()
    if not A or not B: return -1
    best = 9
    for x in A:
        for z in B:
            d = Levenshtein.distance(x, z)
            if d and (x.startswith(z) or z.startswith(x)): d = 1
            best = min(best, d)
    return best

def load_dev():
    d = pl.read_parquet("work/dev_feats.parquet")
    ev = pl.read_parquet("work/eval_s1_ids.parquet").with_columns((pl.col("entity_id").hash(99) % 10).alias("fold"))
    ev = ev.with_columns(pl.when(pl.col("fold") < 6).then(pl.lit("train")).when(pl.col("fold") < 8).then(pl.lit("valid")).otherwise(pl.lit("hold")).alias("split"))
    d = d.join(ev.select(pl.col("entity_id").alias("s1"), "split", "fold"), on="s1")
    nm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "nums") for s in (1, 2, 3)]).collect()
    d = d.join(nm, left_on="s1", right_on="entity_id").join(nm, left_on="m", right_on="entity_id", suffix="_2")
    A, B = d["nums"].to_list(), d["nums_2"].to_list()
    d = d.with_columns(pl.Series("num_min_ed", [_close(a, b) for a, b in zip(A, B)]),
                       pl.Series("num_first_ed", [Levenshtein.distance(a.split()[0], b.split()[0]) if a and b else -1 for a, b in zip(A, B)]),
                       (pl.col("country") == "India").cast(pl.Int8).alias("is_india"), (pl.col("src") == "S3").cast(pl.Int8).alias("is_s3")).drop("nums", "nums_2")
    gt = pl.read_parquet("work/train_per_s1.parquet").join(ev, left_on="s1", right_on="entity_id").select("s1", "n", "split", "country", "fold")
    return d, gt

def X(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()

def fit(tr, va, cols, rounds=800, **kw):
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4); prm.update(kw)
    return lgb.train(prm, lgb.Dataset(X(tr, cols), tr["y"].to_numpy()), rounds,
                     valid_sets=[lgb.Dataset(X(va, cols), va["y"].to_numpy())], callbacks=[lgb.early_stopping(50, verbose=False)])

def macro_f05(df, p, thr, gt_rows, policy="none", margin=0.0):
    """Exact leaderboard metric over all S1 in gt_rows (singletons included; blocking misses = FN).
    policy: none | best_s1 (pool record kept only for its best S1 among candidates) | margin (best_s1 and p1-p2>=margin)."""
    x = df.select("s1", "m", "y").with_columns(pl.Series("p", p), pl.Series("thr", np.broadcast_to(np.asarray(thr, float), len(p)).copy()))
    if policy in ("best_s1", "margin"):
        x = x.with_columns(pl.col("p").max().over("m").alias("pm"),
                           pl.col("p").sort(descending=True).get(1, null_on_oob=True).over("m").fill_null(0).alias("p2"))
        x = x.filter(pl.col("p") == pl.col("pm"))
        if policy == "margin": x = x.filter(pl.col("p") - pl.col("p2") >= margin)
    x = x.filter(pl.col("p") >= pl.col("thr"))
    per = x.group_by("s1").agg(pl.len().alias("pred"), pl.col("y").sum().alias("tp"))
    g = gt_rows.join(per, on="s1", how="left").fill_null(0)
    n, pr, tp = g["n"].to_numpy(), g["pred"].to_numpy(), g["tp"].to_numpy()
    P = tp / np.maximum(pr, 1); R = tp / np.maximum(n, 1)
    f = np.where(n == 0, (pr == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
    s = n == 0
    return float(f.mean()), dict(P=float(tp.sum() / max(1, pr.sum())), R=float(tp.sum() / n.sum()), single=float(f[s].mean()),
                                 multi=float(f[~s].mean()), matches_per_s1=float(pr.mean()), false_merges=int((pr - tp).sum()))

def best_thr(df, p, gt_rows, policy="none", grid=np.arange(0.30, 0.98, 0.02)):
    return max((macro_f05(df, p, t, gt_rows, policy)[0], float(t)) for t in grid)
