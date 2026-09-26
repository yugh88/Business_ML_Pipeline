"""Stage stage2_xgb2: XGBoost retrained with a much larger round budget (V3's XGB hit the 1500-round cap at
1497/1499/1498 -> underfit). Same features / cross-fitting as stage 2; early stopping decides the size.
Adds p2x2 and p2x2b = mean(p2, p2x2) to p2/{split}/*.parquet (variant choice happens in evaluate on the valid fold)."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fit_xgb_big(tr, va, cols, weight=None):
    import xgboost as xgb, polars as pl
    from lib import CFG
    X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    m = xgb.XGBClassifier(n_estimators=CFG["model"].get("xgb2_rounds", 8000), learning_rate=0.05, max_depth=8, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=CFG["model"]["lgb"]["num_threads"],
                          early_stopping_rounds=100, eval_metric="logloss", random_state=CFG["seed"])
    m.fit(X(tr), tr["y"].to_numpy(), eval_set=[(X(va), va["y"].to_numpy())], verbose=False)
    m.predict = lambda Xn, _m=m: _m.predict_proba(Xn)[:, 1]
    m.save_model = lambda p, _m=m: _m.get_booster().save_model(p.replace(".txt", ".json"))
    m.feature_importance = lambda kind="gain", _m=m, _c=cols: [_m.get_booster().get_score(importance_type="gain").get(f"f{i}", 0.0) for i in range(len(_c))]
    return m


def main():
    import polars as pl
    from lib import CFG, P, log, done, mark_done, s3_push, write_parquet_atomic
    from build_features import FEATURES
    from train_model import coll_columns, cons_columns, _load_rows, _train_kway, _predict_crossfit
    if done("stage2_xgb2") or not CFG["model"].get("xgb2_rounds"):
        return
    groups = CFG["model"]["cv_groups"]; cols = FEATURES + coll_columns() + cons_columns()
    t = time.time()
    models = _train_kway(lambda f: _load_rows("train", f, FEATURES, extra="coll"), groups, cols, "stage2_p2x2", log, P, fitter=_fit_xgb_big)
    iters = [int(m.best_iteration) for m in models]
    log(f"stage2_xgb2 trained iters={iters} (cap {CFG['model']['xgb2_rounds']}) {time.time()-t:.0f}s")
    for split in CFG["splits"]:
        for b in range(CFG["features"]["buckets"][split]):
            pf = P(f"p2/{split}/bucket_{b:04d}.parquet")
            if not os.path.exists(pf) or "p2x2" in pl.read_parquet_schema(pf):
                continue
            d = pl.read_parquet(P(f"features/{split}/bucket_{b:04d}.parquet")).join(pl.read_parquet(P(f"coll/{split}/bucket_{b:04d}.parquet")), on=["s1", "m"])
            p2 = pl.read_parquet(pf); d = p2.select("s1", "m").join(d, on=["s1", "m"], how="left")
            p = _predict_crossfit(models, groups, d if split == "train" else d.drop("fold"), cols)
            p2 = p2.join(d.select("s1", "m").with_columns(pl.Series("p2x2", p)), on=["s1", "m"], how="left") \
                   .with_columns(((pl.col("p2") + pl.col("p2x2")) / 2).alias("p2x2b"))
            write_parquet_atomic(p2, f"p2/{split}/bucket_{b:04d}.parquet")
        log(f"stage2_xgb2 predicted {split} {time.time()-t:.0f}s")
    s3_push("p2", recursive=True)
    mark_done("stage2_xgb2", dict(iters=iters))


if __name__ == "__main__":
    main()
