"""Controlled model experiments on dev candidates (eval S1 sample, ~44k S1). Split by S1 hash: 60% train / 20% valid / 20% holdout.
Macro-F0.5 is computed exactly as the leaderboard does, over ALL S1 in the fold (singletons included; GT matches that
blocking missed count as false negatives)."""
import polars as pl, numpy as np, time, json, sys
sys.path.insert(0, "scripts"); import memguard
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
R = {}
d = pl.read_parquet("work/dev_feats.parquet")
ev = pl.read_parquet("work/eval_s1_ids.parquet").with_columns((pl.col("entity_id").hash(99) % 10).alias("fold"))
fold = lambda f: pl.when(f < 6).then(pl.lit("train")).when(f < 8).then(pl.lit("valid")).otherwise(pl.lit("hold"))
ev = ev.with_columns(fold(pl.col("fold")).alias("split"))
d = d.join(ev.select(pl.col("entity_id").alias("s1"), "split"), on="s1")
gt = pl.read_parquet("work/train_per_s1.parquet").join(ev, left_on="s1", right_on="entity_id").select("s1", "n", "split", "country")
d = d.with_columns((pl.col("country") == "India").cast(pl.Int8).alias("is_india"), (pl.col("src") == "S3").cast(pl.Int8).alias("is_s3"))
NAME = ["n_tsr", "n_tsort", "n_ratio", "n_pr", "ns_jw", "ns_ratio", "ns_pr", "n_eq", "core_eq", "core_sorted_eq", "n_jac", "n_idf_cov",
        "n_max_idf_shared", "n_min_df1", "n_max_idf_missing", "n_ntok1", "n_ntok2", "n_shared", "n_len_ratio", "name_freq_s1", "dom2", "indic2", "brand2", "junk2"]
ADDR = ["a_tsr", "a_tsort", "a_pr", "a_eq", "a2_empty", "a_jac", "a_idf_cov", "a_max_idf_shared", "a_max_idf_missing", "a_ntok1", "a_ntok2", "a_shared",
        "num_shared", "num_n1", "num_n2", "num_first_eq", "num_conflict", "state_agree"]
COMP = ["rscore", "rrank", "m_best", "m_gap12", "r_margin_best", "s1_rank_rscore", "s1_n_core_eq", "s1_ncand", "s1_rank_sum"]
CTX = ["is_india", "is_s3"]
ALL = NAME + ADDR + COMP + CTX
def X(df, cols): return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
tr, va, ho = [d.filter(pl.col("split") == s) for s in ("train", "valid", "hold")]
print({s: (x.height, int(x["y"].sum())) for s, x in zip(("train", "valid", "hold"), (tr, va, ho))}, flush=True)

def macro_f05(df, p, thr, policy="argmax", margin=0.0, split="valid"):
    """df: candidate rows of one split; p: probabilities. policy argmax = each pool record keeps only its best S1 (1-to-many)."""
    x = df.select("s1", "m").with_columns(pl.Series("p", p), df["y"], pl.Series("thr", np.broadcast_to(np.asarray(thr, dtype=float), len(p)).copy()))
    if policy in ("argmax", "argmax_margin"):
        x = x.with_columns(pl.col("p").max().over("m").alias("pmax"), pl.col("p").sort(descending=True).get(1, null_on_oob=True).over("m").fill_null(0).alias("p2"))
        x = x.filter(pl.col("p") == pl.col("pmax"))
        if policy == "argmax_margin": x = x.filter(pl.col("p") - pl.col("p2") >= margin)
    x = x.filter(pl.col("p") >= pl.col("thr"))
    per = x.group_by("s1").agg(pl.len().alias("pred"), pl.col("y").sum().alias("tp"))
    g = gt.filter(pl.col("split") == split).join(per, on="s1", how="left").fill_null(0)
    P = (g["tp"] / g["pred"].clip(1)).to_numpy(); Rr = (g["tp"] / g["n"].clip(1)).to_numpy()
    f = np.where((g["n"] == 0).to_numpy(), (g["pred"] == 0).to_numpy().astype(float),
                 np.where((g["tp"] > 0).to_numpy(), 1.25 * P * Rr / np.maximum(0.25 * P + Rr, 1e-9), 0.0))
    single = (g["n"] == 0).to_numpy()
    return float(f.mean()), dict(P_micro=float(g["tp"].sum() / max(1, g["pred"].sum())), R_micro=float(g["tp"].sum() / g["n"].sum()),
                                 singleton_acc=float(f[single].mean()), f_nonsingle=float(f[~single].mean()),
                                 f_US=float(f[(g["country"] == "US").to_numpy()].mean()), f_India=float(f[(g["country"] == "India").to_numpy()].mean()))
def best_thr(df, p, policy="argmax", split="valid", grid=np.arange(0.3, 0.99, 0.02)):
    s = [(macro_f05(df, p, t, policy, split=split)[0], t) for t in grid]
    return max(s)
def fit_lgb(cols, trd=tr, params=None):
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
    prm.update(params or {})
    m = lgb.train(prm, lgb.Dataset(X(trd, cols), trd["y"].to_numpy()), 600, valid_sets=[lgb.Dataset(X(va, cols), va["y"].to_numpy())],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m
# ---------- E1 feature groups (LightGBM) ----------
print("\nE1 feature-group ablation (LightGBM, valid split, argmax policy, best threshold)")
for name, cols in [("name_only", NAME + CTX), ("addr_only", ADDR + CTX), ("name+addr", NAME + ADDR + CTX), ("all(+competition)", ALL)]:
    t = time.time(); m = fit_lgb(cols); p = m.predict(X(va, cols))
    f, thr = best_thr(va, p); _, det = macro_f05(va, p, thr)
    R[f"E1_{name}"] = dict(ap=float(average_precision_score(va["y"], p)), f05=f, thr=float(thr), **det)
    print(f"  {name:20s} AP={R[f'E1_{name}']['ap']:.4f} macroF0.5={f:.4f} thr={thr:.2f} {det} ({time.time()-t:.0f}s)", flush=True)
    if name.startswith("all"): best_m, p_lgb = m, p
imp = sorted(zip(ALL, best_m.feature_importance("gain")), key=lambda z: -z[1])
R["E1_importance_gain"] = [(a, float(b)) for a, b in imp]
print("  top gain:", [(a, round(b / imp[0][1], 3)) for a, b in imp[:20]])
# ---------- E2 model families ----------
print("\nE2 model families (all features)")
Xtr, Xva = np.nan_to_num(X(tr, ALL), nan=-1), np.nan_to_num(X(va, ALL), nan=-1)
sc = StandardScaler().fit(Xtr)
fams = {}
t = time.time(); lr = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xtr), tr["y"].to_numpy()); fams["logreg"] = (lr.predict_proba(sc.transform(Xva))[:, 1], time.time() - t)
fams["lightgbm"] = (p_lgb, 0)
t = time.time(); xm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4, early_stopping_rounds=50, eval_metric="logloss")
xm.fit(X(tr, ALL), tr["y"].to_numpy(), eval_set=[(X(va, ALL), va["y"].to_numpy())], verbose=False); fams["xgboost"] = (xm.predict_proba(X(va, ALL))[:, 1], time.time() - t)
t = time.time(); sub = np.random.RandomState(0).rand(len(Xtr)) < 0.5
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=40, batch_size=2048, random_state=0).fit(sc.transform(Xtr[sub]), tr["y"].to_numpy()[sub])
fams["mlp(128,64)"] = (mlp.predict_proba(sc.transform(Xva))[:, 1], time.time() - t)
for k, (p, secs) in fams.items():
    f, thr = best_thr(va, p); _, det = macro_f05(va, p, thr)
    R[f"E2_{k}"] = dict(ap=float(average_precision_score(va["y"], p)), auc=float(roc_auc_score(va["y"], p)), f05=f, thr=float(thr), secs=secs, **det)
    print(f"  {k:12s} AP={R[f'E2_{k}']['ap']:.4f} AUC={R[f'E2_{k}']['auc']:.5f} macroF0.5={f:.4f} thr={thr:.2f} P={det['P_micro']:.4f} R={det['R_micro']:.4f} single={det['singleton_acc']:.4f} ({secs:.0f}s)", flush=True)
# ---------- E3 decision policy ----------
print("\nE3 decision policy (LightGBM all features)")
for pol in ["threshold_only", "argmax"]:
    f, thr = best_thr(va, p_lgb, policy=pol); _, det = macro_f05(va, p_lgb, thr, policy=pol)
    R[f"E3_{pol}"] = dict(f05=f, thr=float(thr), **det); print(f"  {pol:16s} macroF0.5={f:.4f} thr={thr:.2f} {det}")
thr_a = R["E3_argmax"]["thr"]
for mg in (0.05, 0.1, 0.2, 0.3):
    f, det = macro_f05(va, p_lgb, thr_a, "argmax_margin", margin=mg); R[f"E3_margin{mg}"] = dict(f05=f, **det); print(f"  argmax+margin {mg}: macroF0.5={f:.4f} P={det['P_micro']:.4f} R={det['R_micro']:.4f}")
curve = [(float(t), macro_f05(va, p_lgb, t)[0]) for t in np.arange(0.1, 0.99, 0.05)]
R["E3_threshold_curve"] = curve; print("  thr curve:", [(round(a, 2), round(b, 4)) for a, b in curve])
# ---------- E4 segment thresholds ----------
print("\nE4 per-segment thresholds (tuned on valid, reported on holdout)")
p_ho = best_m.predict(X(ho, ALL))
f_glob, det = macro_f05(ho, p_ho, thr_a, split="hold"); R["E4_global_hold"] = dict(f05=f_glob, **det); print(f"  global thr {thr_a:.2f} holdout macroF0.5={f_glob:.4f} {det}")
seg_thr = {}
segv = (va["country"] + "_" + va["src"]).to_numpy(); segh = (ho["country"] + "_" + ho["src"]).to_numpy()
for sg in np.unique(segv):
    best = (-1, thr_a)
    for t in np.arange(0.3, 0.99, 0.02):
        th = np.where(segv == sg, t, thr_a)
        best = max(best, (macro_f05(va, p_lgb, th)[0], t))
    seg_thr[sg] = float(best[1])
th_h = np.array([seg_thr.get(z, thr_a) for z in segh])
f_seg, det = macro_f05(ho, p_ho, th_h, split="hold")
R["E4_segment_hold"] = dict(f05=f_seg, thr=seg_thr, **det)
print(f"  segment thresholds {seg_thr} holdout macroF0.5={f_seg:.4f} {det}")
# ---------- E5 state-missing robustness (France proxy) ----------
print("\nE5 robustness: state feature forced unknown at inference (France has no state)")
ho0 = ho.with_columns(pl.lit(0).alias("state_agree"))
f0, det0 = macro_f05(ho, best_m.predict(X(ho0, ALL)), thr_a, split="hold")
trd = tr.with_columns(pl.when(pl.col("s1").hash(5) % 4 == 0).then(0).otherwise(pl.col("state_agree")).alias("state_agree"))
m_do = fit_lgb(ALL, trd=trd)
f1, _ = macro_f05(ho, m_do.predict(X(ho0, ALL)), thr_a, split="hold"); f2, _ = macro_f05(ho, m_do.predict(X(ho, ALL)), thr_a, split="hold")
m_ns = fit_lgb([c for c in ALL if c != "state_agree"])
f3, _ = macro_f05(ho, m_ns.predict(X(ho, [c for c in ALL if c != "state_agree"])), thr_a, split="hold")
R["E5"] = dict(normal=f_glob, state_blanked=f0, dropout_model_blanked=f1, dropout_model_normal=f2, no_state_model=f3)
print(f"  normal={f_glob:.4f} state-blanked={f0:.4f} | trained w/ 25% state dropout: blanked={f1:.4f} normal={f2:.4f} | no-state model={f3:.4f}")
json.dump(R, open("analysis/out/43_models.json", "w"), indent=1, default=str)
best_m.save_model("work/dev_lgb_all.txt")
print(memguard.report())
