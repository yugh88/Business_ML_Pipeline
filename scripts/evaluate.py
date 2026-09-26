"""Stage evaluate: full-partition end-to-end validation on train (complete S1 folds, global pool competition).
Variants: p2 (stage-2 + consensus features), p2base (stage-2 without consensus).
Policies on the chosen probability p:
  A threshold | B + each pool record kept only for its best S1 (global) | D = B + abstain if the record's 2nd-best S1 also
  passes | E = B + margin | F = B + twin propagation | G = B + per-S1 expected-F0.5-optimal set (calibration slope a,
  expected missed positives lam; k=0 allowed -> singleton).
Selection on valid fold only (conservative tie-break); holdout folds reported. Writes validation/decision.json."""
import os, sys, json, time
import numpy as np, polars as pl, duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PMIN = 0.01
POLICY_ORDER = ["A_threshold", "B_best_s1", "D_ambiguity", "E_margin", "F_twins", "G_expected_f"]
VARIANTS = ["p2", "p2x", "p2blend", "p2x2", "p2x2b", "p2nn", "p2nnb", "p2nnL", "p2nnLb", "p2base"]   # order = tie-break preference (simplest first); missing columns are skipped


def mstats(split, pcol):
    from lib import WORK, P
    con = duckdb.connect(); con.execute(f"SET threads={os.cpu_count()}; SET temp_directory='{P('duck_tmp','x')[:-2]}'")
    return con.execute(f"""select m, max({pcol}) m_max, coalesce(list_sort(list({pcol}), 'DESC')[2], 0) m_2nd
                           from '{os.path.join(WORK, 'p2', split, '*.parquet')}' group by m""").pl()


def load_scored(split, pcol, folds=None):
    from lib import WORK
    q = pl.scan_parquet(os.path.join(WORK, "p2", split, "*.parquet")).with_columns(pl.col(pcol).alias("p")).filter(pl.col("p") >= PMIN)
    if folds is not None:
        q = q.filter(pl.col("fold").is_in(folds))
    return q.select("s1", "m", "y", "fold", "p", "pc2", "pa").collect().join(mstats(split, pcol), on="m", how="left")


def apply_policy(d, policy, t=0.5, margin=0.0, t_low=0.3, a=1.0, lam=0.0):
    if policy == "G_expected_f":
        x = d.filter(pl.col("p") >= pl.col("m_max"))
        pp = np.clip(x["p"].to_numpy(), 1e-6, 1 - 1e-6); lg = np.log(pp) - np.log1p(-pp)
        x = x.with_columns(pl.Series("q", 1 / (1 + np.exp(-a * lg)))).sort(["s1", "q"], descending=[False, True])
        x = x.with_columns(pl.col("q").cum_sum().over("s1").alias("cq"), pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                           pl.col("q").sum().over("s1").alias("tq"), (1 - pl.col("q")).log().sum().over("s1").alias("l0"))
        x = x.with_columns((1.25 * pl.col("cq") / (0.25 * (pl.col("tq") + lam) + pl.col("k"))).alias("ef"),
                           (pl.col("l0").exp() * np.exp(-lam)).alias("e0"))
        best = x.group_by("s1").agg(pl.col("ef").max().alias("efmax"), pl.col("k").sort_by("ef", descending=True).first().alias("kbest"), pl.col("e0").first())
        x = x.join(best, on="s1").filter((pl.col("efmax") > pl.col("e0")) & (pl.col("k") <= pl.col("kbest")))
        return x.select("s1", "m")
    x = d.filter(pl.col("p") >= t)
    if policy == "A_threshold":
        return x.select("s1", "m")
    x = x.filter(pl.col("p") >= pl.col("m_max"))
    if policy == "D_ambiguity":
        x = x.filter(pl.col("m_2nd") < t)
    elif policy == "E_margin":
        x = x.filter(pl.col("p") - pl.col("m_2nd") >= margin)
    elif policy == "F_twins":
        acc = x.filter(pl.col("pa") != "").select("s1", "pc2", "pa").unique()
        tw = d.filter((pl.col("p") >= t_low) & (pl.col("p") >= pl.col("m_max")) & (pl.col("pa") != "")).join(acc, on=["s1", "pc2", "pa"], how="semi")
        x = pl.concat([x.select("s1", "m"), tw.select("s1", "m")]).unique()
    return x.select("s1", "m")


def score(pred, truth, s1tab):
    tp = pred.join(truth, on=["s1", "m"], how="semi").group_by("s1").len().rename({"len": "tp"})
    pr = pred.group_by("s1").len().rename({"len": "pred"})
    g = s1tab.join(pr, on="s1", how="left").join(tp, on="s1", how="left").fill_null(0)
    n, p, t = g["n"].to_numpy(), g["pred"].to_numpy(), g["tp"].to_numpy()
    P_ = t / np.maximum(p, 1); R_ = t / np.maximum(n, 1)
    f = np.where(n == 0, (p == 0).astype(float), np.where(t > 0, 1.25 * P_ * R_ / np.maximum(0.25 * P_ + R_, 1e-12), 0.0))
    s = n == 0
    return dict(f05=float(f.mean()), P=float(t.sum() / max(1, p.sum())), R=float(t.sum() / max(1, n.sum())),
                singleton_acc=float(f[s].mean()) if s.any() else None, multi_f05=float(f[~s].mean()),
                false_merges=int((p - t).sum()), false_negatives=int((n - t).sum()), matches_per_s1=float(p.mean()), n_s1=int(len(n)))


def s1_table():
    from lib import P, CFG, keep_s1_expr
    con = duckdb.connect()
    return con.execute(f"""with g as (select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0
                             else len(string_split(matched_entity_ids, ',')) end n from '{P('input', 'train_gt.parquet')}')
                           select g.s1, g.n, hash(g.s1 || 'fold') % {CFG['model']['folds']} fold from g""").pl().filter(keep_s1_expr("s1"))


def configs(CFG):
    lo, hi, st = CFG["decision"]["threshold_grid"]
    for t in np.round(np.arange(lo, hi + 1e-9, st), 3):
        for pol in ("A_threshold", "B_best_s1", "D_ambiguity", "F_twins"):
            yield dict(policy=pol, t=float(t))
        for mg in CFG["decision"]["margins"]:
            yield dict(policy="E_margin", t=float(t), margin=mg)
    for a in (0.8, 1.0, 1.25, 1.5, 2.0):
        for lam in (0.0, 0.05, 0.15, 0.3, 0.5):
            yield dict(policy="G_expected_f", a=a, lam=lam)


def argk(r):
    return {k: r[k] for k in ("t", "margin", "a", "lam") if k in r}


def main():
    from lib import CFG, P, WORK, log, done, mark_done, s3_push, keep_s1_sql
    if done("evaluate"):
        return
    t0 = time.time()
    vf, hf = CFG["model"]["valid_fold"], CFG["model"]["holdout_folds"]
    s1t = s1_table(); sv = s1t.filter(pl.col("fold") == vf)
    res, data = [], {}
    variants = [v for v in VARIANTS if v in pl.read_parquet_schema(sorted(__import__("glob").glob(os.path.join(WORK, "p2", "train", "*.parquet")))[0])]
    for v in variants:
        d = load_scored("train", v, [vf] + hf); data[v] = d
        truth = d.filter(pl.col("y")).select("s1", "m").unique()
        dv = d.filter(pl.col("fold") == vf)
        for c in configs(CFG):
            r = score(apply_policy(dv, c["policy"], **argk(c)), truth, sv); r.update(c); r["variant"] = v; res.append(r)
        log(f"  evaluated variant {v}: best valid {max(r['f05'] for r in res if r['variant'] == v):.5f}")
    best = max(res, key=lambda r: r["f05"])
    near = [r for r in res if r["f05"] >= best["f05"] - 0.0003]
    pick = sorted(near, key=lambda r: (variants.index(r["variant"]), POLICY_ORDER.index(r["policy"]), -r.get("t", 0)))[0]
    dec = dict(variant=pick["variant"], policy=pick["policy"], params=argk(pick), valid=pick, best_valid=best)
    report = dict(decision=dec, holdout={}, best_valid_per_variant_policy={})
    for v in variants:
        d = data[v]; truth = d.filter(pl.col("y")).select("s1", "m").unique()
        for pol in POLICY_ORDER:
            cand = [r for r in res if r["variant"] == v and r["policy"] == pol]
            b = max(cand, key=lambda r: r["f05"]); key = f"{v}/{pol}"
            report["best_valid_per_variant_policy"][key] = dict(valid=b["f05"], params=argk(b))
            report["holdout"][key] = {int(f): score(apply_policy(d.filter(pl.col("fold") == f), pol, **argk(b)), truth, s1t.filter(pl.col("fold") == f)) for f in hf}
    d = data[pick["variant"]]; truth = d.filter(pl.col("y")).select("s1", "m").unique()
    H = d.filter(pl.col("fold").is_in(hf)); sh = s1t.filter(pl.col("fold").is_in(hf))
    pred = apply_policy(H, pick["policy"], **argk(pick))
    cmap = pl.scan_parquet(os.path.join(WORK, "normalized", "train_s1", "*.parquet")).select(pl.col("entity_id").alias("s1"), "country").collect()
    seg = {}
    for c in cmap["country"].unique().to_list():
        ss = sh.join(cmap.filter(pl.col("country") == c), on="s1", how="semi"); seg[f"country={c}"] = score(pred.join(ss, on="s1", how="semi"), truth, ss)
    for src in ("S2", "S3"):
        pp = pred.filter(pl.col("m").str.starts_with(src)); seg[f"precision_src={src}"] = pp.join(truth, on=["s1", "m"], how="semi").height / max(1, pp.height)
    report["segments_holdout"] = seg
    con = duckdb.connect(); con.execute(f"SET threads={os.cpu_count()}")
    gtp = int(s1t["n"].sum()); q = lambda s: con.execute(s).fetchone()[0]
    import glob as _g
    has_c = bool(_g.glob(os.path.join(WORK, 'candidates', 'train', '*', '*.parquet')))
    report["recall_ceilings"] = dict(
        union=(q(f"select sum(y::int) from '{os.path.join(WORK, 'candidates', 'train', '*', '*.parquet')}' where {keep_s1_sql('s1')}") / gtp) if has_c else None,
        cascade=q(f"select sum(y::int) from '{os.path.join(WORK, 'p2', 'train', '*.parquet')}'") / gtp,
        union_pairs_per_s1=(q(f"select count(*) from '{os.path.join(WORK, 'candidates', 'train', '*', '*.parquet')}'") / len(s1t)) if has_c else None,
        cascade_pairs_per_s1=q(f"select count(*) from '{os.path.join(WORK, 'p2', 'train', '*.parquet')}'") / len(s1t))
    report["all_valid_results"] = res
    json.dump(report, open(P("validation/report.json"), "w"), indent=1, default=float)
    json.dump(dec, open(P("validation/decision.json"), "w"), indent=1, default=float)
    hk = f"{pick['variant']}/{pick['policy']}"
    log(f"evaluate: pick={hk} {argk(pick)} valid={pick['f05']:.5f} best={best['variant']}/{best['policy']}={best['f05']:.5f} "
        f"holdout={[round(report['holdout'][hk][f]['f05'], 5) for f in hf]} ceilings={report['recall_ceilings']} {time.time()-t0:.0f}s")
    s3_push("validation", recursive=True)
    mark_done("evaluate", dec)


if __name__ == "__main__":
    main()
