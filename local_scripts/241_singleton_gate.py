"""idea.txt Pillar 4 (dedicated singleton gate) on the labelled holdout: per-S1 classifier P(n == 0) from candidate-score features
(top-1/2/3 score, gap, entropy, counts above cut-offs, stage-1 max, name rarity proxy = #candidates), fit on clean fold 7,
applied on folds 0/8/9 on top of the V11 decoding: empty the S1's prediction when P(singleton) >= t (t chosen on fold 7).
Upper bound = the 'singleton + FP' loss of the error budget."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
COL = sys.argv[1] if len(sys.argv) > 1 else "p3"                     # probability column: p3 (V11) or p3c (CE-stacked, from 239 cache)


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 7, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 7, 8, 9)]).join(SYN, on="m", how="anti") \
      .filter(pl.col("p1") >= TAU).join(t1, on="s1")
if COL != "p3":
    h = h.join(pl.read_parquet("work/ce_holdout_probs.parquet"), on=["s1", "m"], how="left").with_columns(pl.col(COL).fill_null(pl.col("p3"))).drop("p3").rename({COL: "p3"})
h = h.with_columns(pl.Series("ip1", iso.predict(h["p1"].to_numpy())).cast(pl.Float32)).with_columns((0.75 * pl.col("p3") + 0.25 * pl.col("ip1")).alias("qa"))
acc = decide(prep(h.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m").join(h.select("s1", "m", "y"), on=["s1", "m"])
# per-S1 features over the candidate set (all S1 in gt; S1 without candidates get zeros)
q = h.sort("qa", descending=True).group_by("s1").agg(
    pl.col("qa").first().alias("q1"), pl.col("qa").get(1, null_on_oob=True).fill_null(0).alias("q2"), pl.col("qa").get(2, null_on_oob=True).fill_null(0).alias("q3"),
    (pl.col("qa") >= 0.5).sum().alias("n50"), (pl.col("qa") >= 0.9).sum().alias("n90"), (pl.col("qa") >= 0.2).sum().alias("n20"), pl.len().alias("ncand"),
    pl.col("p1").max().alias("p1max"), (1 - pl.col("qa")).log().sum().alias("logp_empty"),
    (-(pl.col("qa").clip(1e-6, 1 - 1e-6) * pl.col("qa").clip(1e-6, 1 - 1e-6).log())).sum().alias("ent"))
nacc = acc.group_by("s1").agg(pl.len().alias("nacc"))
S = gt.join(q, on="s1", how="left").join(nacc, on="s1", how="left").fill_null(0).with_columns((pl.col("q1") - pl.col("q2")).alias("gap"), (pl.col("country") == "India").cast(pl.Int8).alias("ind"))
S = S.join(acc.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
S = S.with_columns(pl.Series("f0", F(S["tp"], S["fp"], S["n"])))
FEATS = ["q1", "q2", "q3", "gap", "n50", "n90", "n20", "ncand", "p1max", "logp_empty", "ent", "nacc", "ind"]
A, B = S.filter(pl.col("fold") == 7), S.filter(pl.col("fold") != 7)
m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, verbose=-1, seed=3, num_threads=4),
              lgb.Dataset(A.select(FEATS).to_numpy(), (A["n"] == 0).to_numpy().astype(int)), num_boost_round=400)
pa, pb_ = m.predict(A.select(FEATS).to_numpy()), m.predict(B.select(FEATS).to_numpy())


def gated(z, p, t):
    """empty the prediction of S1 with P(singleton) >= t and a non-empty prediction."""
    g = (p >= t) & (z["nacc"].to_numpy() > 0)
    f = np.where(g, (z["n"].to_numpy() == 0).astype(float), z["f0"].to_numpy())
    return f, int(g.sum()), int((g & (z["n"].to_numpy() == 0)).sum())


print(f"[{COL}] holdout 0/8/9: S1 {B.height}, singletons {int((B['n'] == 0).sum())}, singleton+FP (upper bound pool) {int(((B['n'] == 0) & (B['nacc'] > 0)).sum())} "
      f"= max gain {float(((B['n'] == 0) & (B['nacc'] > 0)).sum()) / B.height:+.5f}")
best_t = max(np.arange(0.3, 0.99, 0.02), key=lambda t: gated(A, pa, t)[0].mean())
for t in sorted(set([round(float(best_t), 2), 0.5, 0.7, 0.9])):
    fb, ng, ok = gated(B, pb_, t)
    by = {c: float(fb[(B["country"] == c).to_numpy()].mean() - B.filter(pl.col("country") == c)["f0"].mean()) for c in ("US", "India")}
    print(f"  t={t:.2f}{' (fold-7 choice)' if abs(t - best_t) < 1e-9 else ''}: gated {ng} S1 ({ok} truly singleton) -> holdout F change US {by['US']:+.5f} India {by['India']:+.5f}")
