"""Stages stage1 / collective / stage2 with cross-fitting (every probability used downstream is out-of-sample).
stage1: M1a trained on fold 1, M1b on fold 2  -> p1 (train fold1<-M1b, fold2<-M1a, others & test <- mean)
collective: global per-pool-record best/2nd p1 (DuckDB) + per-S1 sibling/anchor features (collective.py)
stage2: M2a on folds {3,4}, M2b on {5,6} (features + p1 + collective) -> p2 by the same cross-fit rule."""
import os, sys, glob, json, time, multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _feat_path(split, b):
    from lib import P
    return P(f"features/{split}/bucket_{b:04d}.parquet")


def _load_rows(split, folds, cols, extra=None, sub=None):
    """Concatenate rows of the given folds across buckets (optionally joined with an extra per-bucket table)."""
    import polars as pl
    from lib import CFG, P
    parts = []
    for b in range(CFG["features"]["buckets"][split]):
        f = _feat_path(split, b)
        if not os.path.exists(f):
            continue
        d = pl.read_parquet(f, columns=list(dict.fromkeys(["s1", "m", "y", "fold"] + [c for c in cols if c in pl.read_parquet_schema(f)])))
        d = d.filter(pl.col("fold").is_in(folds))
        if extra:
            d = d.join(pl.read_parquet(P(f"{extra}/{split}/bucket_{b:04d}.parquet")), on=["s1", "m"])
        if sub is not None:
            d = sub(d)
        parts.append(d)
    return pl.concat(parts)


def _fit(tr, va, cols, weight=None):
    import lightgbm as lgb
    from lib import CFG
    p = dict(CFG["model"]["lgb"]); rounds = p.pop("max_rounds"); es = p.pop("early_stopping")
    p.update(objective="binary", verbose=-1, seed=CFG["seed"])
    X = lambda d: d.select([__import__("polars").col(c).cast(__import__("polars").Float32) for c in cols]).to_numpy()
    w = tr[weight].to_numpy() if weight else None
    return lgb.train(p, lgb.Dataset(X(tr), tr["y"].to_numpy(), weight=w), rounds,
                     valid_sets=[lgb.Dataset(X(va), va["y"].to_numpy())], callbacks=[lgb.early_stopping(es, verbose=False)])


def _fit_xgb(tr, va, cols, weight=None):
    import xgboost as xgb, polars as pl
    from lib import CFG
    X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          tree_method="hist", n_jobs=CFG["model"]["lgb"]["num_threads"], early_stopping_rounds=50, eval_metric="logloss",
                          random_state=CFG["seed"])
    m.fit(X(tr), tr["y"].to_numpy(), eval_set=[(X(va), va["y"].to_numpy())], verbose=False)
    m.predict = lambda Xn, _m=m: _m.predict_proba(Xn)[:, 1]
    m.save_model = lambda p, _m=m: _m.get_booster().save_model(p.replace(".txt", ".json"))
    m.feature_importance = lambda kind="gain", _m=m, _c=cols: [_m.get_booster().get_score(importance_type="gain").get(f"f{i}", 0.0) for i in range(len(_c))]
    return m


def _predict_crossfit(models, groups, d, cols):
    """K-way cross-fit: model k was trained on all groups except groups[k]; rows of groups[k] get model k,
    every other row (valid/holdout/test) gets the mean of all K models."""
    import polars as pl
    X = d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    P = np.vstack([m.predict(X) for m in models])
    f = d["fold"].to_numpy() if "fold" in d.columns else np.full(d.height, -1)
    out = P.mean(axis=0)
    for k, g in enumerate(groups):
        out = np.where(np.isin(f, g), P[k], out)
    return out.astype(np.float32)


def _train_kway(split_rows_fn, groups, cols, tag, log, P, weight=None, fitter=None):
    import polars as pl
    models = []
    for k, g in enumerate(groups):
        tr_f = [f for gg in groups if gg is not g for f in gg]
        tr = split_rows_fn(tr_f)
        es = tr.filter(pl.col("s1").hash(3) % 10 == 0); tr = tr.filter(pl.col("s1").hash(3) % 10 != 0)
        log(f"{tag} model {k}: train folds {tr_f} rows={tr.height} es={es.height}")
        m = (fitter or _fit)(tr, es, cols, weight=weight); m.save_model(P(f"models/{tag}_{k}.txt")); models.append(m); del tr, es
    return models


def stage1():
    import polars as pl, lightgbm as lgb
    from lib import CFG, P, log, done, mark_done, s3_push, write_parquet_atomic
    from build_features import FEATURES
    if done("stage1"):
        return
    groups = CFG["model"]["cv_groups"]; nf = CFG["model"].get("stage1_neg_frac", 1.0)
    t = time.time()
    sub = lambda d: d.filter(pl.col("y") | (pl.col("m").hash(9) % 1000 < int(nf * 1000))).with_columns(
        pl.when(pl.col("y")).then(1.0).otherwise(1.0 / nf).alias("w"))
    models = _train_kway(lambda f: _load_rows("train", f, FEATURES, sub=sub), groups, FEATURES, "stage1", log, P, weight="w")
    imp = sorted(zip(FEATURES, models[0].feature_importance("gain")), key=lambda z: -z[1])
    json.dump([(a, float(b)) for a, b in imp], open(P("models/stage1_importance.json"), "w"), indent=1)
    log(f"stage1 trained iters={[m.best_iteration for m in models]} {time.time()-t:.0f}s")
    for split in CFG["splits"]:
        for b in range(CFG["features"]["buckets"][split]):
            f = _feat_path(split, b)
            if not os.path.exists(f) or os.path.exists(P(f"p1/{split}/bucket_{b:04d}.parquet")):
                continue
            d = pl.read_parquet(f)
            p = _predict_crossfit(models, groups, d if split == "train" else d.drop("fold"), FEATURES)
            write_parquet_atomic(d.select("s1", "m", "y", "fold").with_columns(pl.Series("p1", p)), f"p1/{split}/bucket_{b:04d}.parquet")
        log(f"stage1 predicted {split}")
    s3_push("models", recursive=True); s3_push("p1", recursive=True)
    mark_done("stage1", dict(iters=[m.best_iteration for m in models]))


def _coll_bucket(args):
    split, b, env = args
    os.environ.update(env)
    import polars as pl
    from lib import P, write_parquet_atomic
    from collective import add_collective, add_consensus
    from lib import WORK
    out = f"coll/{split}/bucket_{b:04d}.parquet"
    if os.path.exists(P(out)) or not os.path.exists(_feat_path(split, b)):
        return
    d = pl.read_parquet(_feat_path(split, b), columns=["s1", "m", "pc2", "pa", "pns", "pnum1", "pstreet"]) \
          .join(pl.read_parquet(P(f"p1/{split}/bucket_{b:04d}.parquet"), columns=["s1", "m", "p1"]), on=["s1", "m"]) \
          .filter(pl.col("p1") >= float(os.environ["CASCADE_P1"]))   # learned candidate filter: stage 2 only sees these
    mstat = pl.read_parquet(P(f"p1/{split}_mstat.parquet")).join(d.select("m").unique(), on="m", how="semi")
    c, cols = add_collective(d, mstat)
    s1n = pl.scan_parquet(os.path.join(WORK, "normalized", f"{split}_s1", "*.parquet")).select(pl.col("entity_id").alias("s1"), pl.col("pnum1").alias("s1num1")) \
            .join(d.lazy().select("s1").unique(), on="s1", how="semi").collect()
    k, kcols = add_consensus(d.join(s1n, on="s1", how="left").with_columns(pl.col("s1num1").fill_null("")))
    write_parquet_atomic(c.join(k, on=["s1", "m"]).join(d.select("s1", "m", "p1"), on=["s1", "m"]), out)


def collective():
    import duckdb, polars as pl
    from lib import CFG, P, WORK, log, done, mark_done, s3_push
    if done("collective"):
        return
    for split in CFG["splits"]:
        con = duckdb.connect(); con.execute(f"SET threads={os.cpu_count()}; SET temp_directory='{P('duck_tmp','x')[:-2]}'")
        con.execute(f"""copy (select m, max(p1) mx, coalesce(list_sort(list(p1), 'DESC')[2], 0) sd
                        from '{os.path.join(WORK, 'p1', split, '*.parquet')}' group by m) to '{P(f"p1/{split}_mstat.parquet")}' (format parquet)""")
        W = CFG["features"]["workers"]; env = {"POLARS_MAX_THREADS": str(max(1, (os.cpu_count() or 4) // W)), "CASCADE_P1": str(CFG["model"]["cascade_p1"])}
        jobs = [(split, b, env) for b in range(CFG["features"]["buckets"][split])]
        if W <= 1:
            for j in jobs: _coll_bucket(j)
        else:
            with mp.get_context("spawn").Pool(W, maxtasksperchild=8) as pool:
                list(pool.imap_unordered(_coll_bucket, jobs))
        log(f"collective {split} done")
    s3_push("coll", recursive=True)
    mark_done("collective")


def coll_columns():
    from collective import SIB_KEYS
    cols = ["c1_rank_s1", "c1_s1_n50", "c1_s1_sum", "c1_s1_max", "c1_s1_other", "c1_m_other"]
    for _, nm in SIB_KEYS:
        cols += [f"c1_sib_{nm}_n", f"c1_sib_{nm}_max"]
    return ["p1"] + cols + ["c1_anc_p", "c1_anc_name", "c1_anc_addr", "c1_anc_num"]


def cons_columns():
    return ["k1_vote_num", "k1_vote_street", "k1_vote_name", "k1_vote_addr", "k1_s1num_share",
            "k1_s1num_same_src", "k1_s1num_other_src", "k1_num_same_src", "k1_num_other_src"]   # v8: per-source version / twin cluster


def stage2():
    import polars as pl
    from lib import CFG, P, log, done, mark_done, s3_push, write_parquet_atomic
    from build_features import FEATURES
    if done("stage2"):
        return
    groups = CFG["model"]["cv_groups"]
    F2 = FEATURES + coll_columns() + cons_columns()
    variants = {"p2": F2, "p2x": F2}                       # v3: LightGBM vs XGBoost on identical features (blend = mean, in evaluate)
    fitters = {"p2": None, "p2x": _fit_xgb}
    t = time.time(); M = {}
    for v, cols in variants.items():
        M[v] = _train_kway(lambda f: _load_rows("train", f, FEATURES, extra="coll"), groups, cols, f"stage2_{v}", log, P, fitter=fitters[v])
        imp = sorted(zip(cols, M[v][0].feature_importance("gain")), key=lambda z: -z[1])
        json.dump([(a, float(b)) for a, b in imp], open(P(f"models/stage2_{v}_importance.json"), "w"), indent=1)
    log(f"stage2 trained {[(v, [m.best_iteration for m in ms]) for v, ms in M.items()]} {time.time()-t:.0f}s")
    for split in CFG["splits"]:
        for b in range(CFG["features"]["buckets"][split]):
            f = _feat_path(split, b)
            if not os.path.exists(f) or os.path.exists(P(f"p2/{split}/bucket_{b:04d}.parquet")):
                continue
            d = pl.read_parquet(f).join(pl.read_parquet(P(f"coll/{split}/bucket_{b:04d}.parquet")), on=["s1", "m"])
            dd = d if split == "train" else d.drop("fold")
            ps = {v: pl.Series(v, _predict_crossfit(M[v], groups, dd, cols)) for v, cols in variants.items()}
            ps["p2blend"] = pl.Series("p2blend", ((ps["p2"].to_numpy() + ps["p2x"].to_numpy()) / 2).astype(np.float32))
            write_parquet_atomic(d.select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "a2_empty", "rrank").with_columns(*ps.values()),
                                 f"p2/{split}/bucket_{b:04d}.parquet")
        log(f"stage2 predicted {split}")
    s3_push("models", recursive=True); s3_push("p2", recursive=True)
    mark_done("stage2", {v: [m.best_iteration for m in ms] for v, ms in M.items()})


if __name__ == "__main__":
    {"stage1": stage1, "collective": collective, "stage2": stage2}[sys.argv[1]]()
