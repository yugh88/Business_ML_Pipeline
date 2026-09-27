"""Cross-encoder integration, HOLDOUT gate: stack the pilot CE logit (238) with p3 on the uncertain band (0.02 <= p3 <= 0.98),
fit on clean fold 7, then run the unchanged V11 decoding (G expected-F, alpha 0.75 US/India tempering, exclusivity, caps) on clean
folds 0/8/9 and compare macro-F0.5 per country with the V11 baseline. Two stackers (logistic, small LightGBM), chosen by band log-loss.
Saves the fitted stackers to work/ce_stackers.pkl for the test build (240)."""
import sys, json, pickle
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate() -> cls, off
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
ALPHA = {"US": 0.75, "India": 0.75, "France": 1.0}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
CLS = ["identical@same", "light_edit@same", "swap_extra@same", "first_word_replaced@same", "no_overlap@same", "desc@same",
       "identical@num_changed", "identical@empty", "identical@other", "other"]
OFF = ["same", "no_num", "+1..9", "-1..9", "+10..99", "-10..99", "trunc", "far", "nonnum"]


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def Xlr(z, pooled=False):
    ce, l3 = z["ce"].to_numpy(), lg(z["p3"].to_numpy())
    cd = [(z["cls"] == c).to_numpy().astype(float) for c in CLS]; od = [(z["off"] == o).to_numpy().astype(float) for o in OFF]
    cols = [l3, ce] + cd + od + [c * ce for c in cd] + [o * ce for o in od]
    if not pooled:
        ind = (z["country"] == "India").to_numpy().astype(float); cols += [ind, ind * ce, ind * l3]
    return np.column_stack(cols)


def Xgb(z, pooled=False):
    cols = [lg(z["p3"].to_numpy()), z["ce"].to_numpy(), lg(z["p1"].to_numpy()),
            z["cls"].replace_strict(CLS, list(range(len(CLS))), default=len(CLS)).to_numpy(), z["off"].replace_strict(OFF, list(range(len(OFF))), default=len(OFF)).to_numpy()]
    if not pooled: cols.append((z["country"] == "India").to_numpy().astype(float))
    return np.column_stack(cols).astype(np.float64)


def fit(a, kind, pooled):
    y = a["y"].to_numpy().astype(int)
    if kind == "lr":
        return LogisticRegression(max_iter=5000, C=1.0).fit(Xlr(a, pooled), y)
    ds = lgb.Dataset(Xgb(a, pooled), y, categorical_feature=[3, 4], free_raw_data=False)
    return lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
                          bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=4), ds, num_boost_round=300)


def pred(m, z, kind, pooled):
    return m.predict_proba(Xlr(z, pooled))[:, 1] if kind == "lr" else m.predict(Xgb(z, pooled))


# ---------------- data
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 7, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]).with_columns(pl.lit(f).alias("fold")) for f in (0, 7, 8, 9)]) \
      .join(SYN, on="m", how="anti").filter(pl.col("p1") >= TAU).join(t1, on="s1")
ce = pl.read_parquet("work/ce_band_train.parquet")
band = h.filter(pl.col("p3").is_between(0.02, 0.98)).join(ce, on=["s1", "m"], how="left")
print(f"band pairs {band.height}, CE missing {band['ce'].null_count()}", flush=True)
band = annotate(band.filter(pl.col("ce").is_not_null()), "train")
A, B = band.filter(pl.col("fold") == 7), band.filter(pl.col("fold") != 7)
yb = B["y"].to_numpy()
print(f"fit fold 7: {A.height} pairs (true {A['y'].mean():.3f}); eval folds 0/8/9: {B.height} pairs", flush=True)
print(f"  p3 alone: log-loss {log_loss(yb, np.clip(B['p3'].to_numpy(), 1e-6, 1 - 1e-6)):.5f} AUC {roc_auc_score(yb, B['p3']):.4f} errors {int(((B['p3'].to_numpy() >= 0.5) != yb).sum())}")
M, P = {}, {}
for kind in ("lr", "gb"):
    for pooled in (False, True):
        M[(kind, pooled)] = fit(A, kind, pooled); P[(kind, pooled)] = p = pred(M[(kind, pooled)], B, kind, pooled)
        print(f"  stack {kind} pooled={pooled}: log-loss {log_loss(yb, p):.5f} AUC {roc_auc_score(yb, p):.4f} errors {int(((p >= 0.5) != yb).sum())}", flush=True)
# per-country band check for the country-aware stackers
for c in ("US", "India"):
    mk = (B["country"] == c).to_numpy()
    print(f"  {c}: p3 ll {log_loss(yb[mk], np.clip(B['p3'].to_numpy()[mk], 1e-6, 1 - 1e-6)):.5f} | " +
          " | ".join(f"{k}{'P' if pl_ else ''} {log_loss(yb[mk], P[(k, pl_)][mk]):.5f}" for k, pl_ in P))
best = min(P, key=lambda k: log_loss(yb, P[k]) if not k[1] else 9)                              # country-aware stacker for US/India
bestp = min([k for k in P if k[1]], key=lambda k: log_loss(yb, P[k]))                             # pooled stacker (France candidate)
print(f"chosen: US/India {best}, pooled {bestp}", flush=True)
pickle.dump({"country": (best, M[best]), "pooled": (bestp, M[bestp])}, open("work/ce_stackers.pkl", "wb"))

# ---------------- decode on clean holdout folds 0/8/9
H = h.filter(pl.col("fold") != 7)
Bp = B.select("s1", "m").with_columns(pl.Series("p3c", P[best]).cast(pl.Float32), pl.Series("p3p", P[bestp]).cast(pl.Float32))
Bp.write_parquet("work/ce_holdout_probs.parquet")
H = H.join(Bp, on=["s1", "m"], how="left").with_columns(pl.col("p3c").fill_null(pl.col("p3")), pl.col("p3p").fill_null(pl.col("p3")))
H = H.with_columns(pl.Series("ip1", iso.predict(H["p1"].to_numpy())).cast(pl.Float32))
g = gt.filter(pl.col("fold") != 7)


def decode(x, col, alpha):
    x = x.with_columns((alpha * pl.col(col) + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(x.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m")


def hscore(acc):
    a = acc.join(H.select("s1", "m", "y"), on=["s1", "m"])
    z = g.join(a.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    z = z.with_columns(pl.Series("f", F(z["tp"], z["fp"], z["n"])))
    return {c: float(z.filter(pl.col("country") == c)["f"].mean()) for c in ("US", "India")}, a


res = {}
for alpha_mode in ("v11", "a1"):
    for col in ("p3", "p3c", "p3p"):
        acc = pl.concat([decode(H.filter(pl.col("country") == c), col, ALPHA[c] if alpha_mode == "v11" else 1.0) for c in ("US", "India")])
        f, a = hscore(acc); res[(alpha_mode, col)] = (f, a)
        b0 = res[(alpha_mode, "p3")][0]
        print(f"HOLDOUT clean F [{alpha_mode}] {col:4s}: US {f['US']:.5f} ({f['US'] - b0['US']:+.5f})  India {f['India']:.5f} ({f['India'] - b0['India']:+.5f})  "
              f"accepted {a.height} TP {int(a['y'].sum())} FP {int((~a['y']).sum())}", flush=True)
# composition of changes (V11 logic, country-aware stacker) by relation class, per 1k holdout S1
nh = dict(g.group_by("country").len().rows())
a0, a1 = res[("v11", "p3")][1], res[("v11", "p3c")][1]
ad = annotate(a1.join(a0, on=["s1", "m"], how="anti"), "train").join(t1, on="s1")
rm = annotate(a0.join(a1, on=["s1", "m"], how="anti"), "train").join(t1, on="s1")
for nm_, z in (("ADDED", ad), ("REMOVED", rm)):
    print(f"{nm_} by class (per 1k holdout S1): TP / FP")
    t = z.group_by("country", "cls").agg(pl.col("y").sum().alias("TP"), (~pl.col("y")).sum().alias("FP")).with_columns(
        (pl.col("TP") / pl.col("country").replace_strict(nh) * 1000).round(2).alias("TP/1k"), (pl.col("FP") / pl.col("country").replace_strict(nh) * 1000).round(2).alias("FP/1k")).sort("country", "TP", descending=[False, True])
    print(t)
ad.select("s1", "m", "y", "cls", "off", "country").write_parquet("work/ce_holdout_added.parquet"); rm.select("s1", "m", "y", "cls", "off", "country").write_parquet("work/ce_holdout_removed.parquet")
