"""V10 ablation: arm A (V8 look-alikes) vs arm B (V10 test-density look-alikes), same 30% S1 subsample, same test features.
Per arm: variant/policy picked on fold 7 (tau 0.05, G grid) -> holdout folds 0/8/9 CLEAN F per country + errors by category;
test: accepted density per category vs the arm's holdout TP density (excess), +/-k symmetry, promoted (stage-2 accepts, pairwise rejects).
B vs A: holdout-referenced LB value of the test changes per category (211 method: holdout-like part valued at holdout truth,
test excess removals = look-alike FPs removed, excess additions = FPs, conservative).  Inputs: work/abl_p2_{a,b}/{train,test}/*.parquet"""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])      # annotate(), offset()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
TAU = 0.05
SYN = lambda c: pl.col(c).str.contains(r"^S[23]-[89]\d{9}$")
GRID = [dict(policy="G_expected_f", a=a, lam=l, cap=1) for a in (0.8, 1.0, 1.25) for l in (0.0, 0.05, 0.15)]


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/abl/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1 = pl.read_parquet("work/abl/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
gt = gt.join(t1, on="s1")
R = {}
for arm in ("a", "b"):
    d = pl.scan_parquet(f"work/abl_p2_{arm}/train/*.parquet").filter(pl.col("fold").is_in([0, 7, 8, 9]) & (pl.col("p1") >= TAU)).collect()
    vars_ = [v for v in ("p2", "p2x", "p2blend") if v in d.columns]
    clean = d.filter(~SYN("m"))
    v7 = clean.filter(pl.col("fold") == 7); s7 = gt.filter(pl.col("fold") == 7)
    def score(dd, v, c, s1f):
        a = decide(prep(dd.select("s1", "m", "p1", v), TAU, v), c).join(dd.select("s1", "m", "y"), on=["s1", "m"])
        z = s1f.join(a.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
        return float(F(z["tp"], z["fp"], z["n"]).mean())
    best = max(((score(v7, v, c, s7), v, c) for v in vars_ for c in GRID), key=lambda z: z[0])
    _, V, C = best
    h = clean.filter(pl.col("fold").is_in([0, 8, 9]))
    acc = decide(prep(h.select("s1", "m", "p1", V), TAU, V), C).with_columns(pl.lit(True).alias("acc"))
    h = h.join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False)).join(t1, on="s1")
    hf = gt.filter(pl.col("fold").is_in([0, 8, 9])).join(h.filter("acc").group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    hf = hf.with_columns(pl.Series("f", F(hf["tp"], hf["fp"], hf["n"])))
    fc = {c: float(hf.filter(pl.col("country") == c)["f"].mean()) for c in ("US", "India")}
    h = annotate(h.select("s1", "m", "y", "acc", "country", "p1", V), "train")
    te = pl.scan_parquet(f"work/abl_p2_{arm}/test/*.parquet").filter(pl.col("p1") >= TAU).select("s1", "m", "p1", "country", V).collect()
    tacc = decide(prep(te.select("s1", "m", "p1", V), TAU, V), C).with_columns(pl.lit(True).alias("acc"))
    p1acc = decide(prep(te.select("s1", "m", "p1"), TAU, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).with_columns(pl.lit(True).alias("pw"))
    te = te.join(tacc, on=["s1", "m"], how="left").join(p1acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False), pl.col("pw").fill_null(False))
    te = annotate(te.filter(pl.col("country").is_in(["US", "India"])).select("s1", "m", "acc", "pw", "country"), "test")
    R[arm] = dict(V=V, C=C, valid=best[0], holdout=fc, h=h, te=te)
    print(f"ARM {arm.upper()}: variant {V} {C} fold7 {best[0]:.5f} | holdout clean US {fc['US']:.5f} India {fc['India']:.5f}", flush=True)
nsh = dict(t1.join(gt.filter(pl.col("fold").is_in([0, 8, 9])).select("s1"), on="s1", how="semi").group_by("country").len().rows())
nst = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
print("\nPER CATEGORY (per 1k S1): holdout TP/FP accepted, test accepted; excess = test acc - holdout TP (lower = fewer look-alike false merges)")
rows = []
for arm in ("a", "b"):
    h, te = R[arm]["h"], R[arm]["te"]
    H = h.group_by("country", "cls", "off").agg((pl.col("acc") & pl.col("y")).sum().alias("TP"), (pl.col("acc") & ~pl.col("y")).sum().alias("FP"), (~pl.col("acc") & pl.col("y")).sum().alias("FN"))
    H = H.with_columns(*[(pl.col(c) / pl.col("country").replace_strict(nsh) * 1000).alias(c) for c in ("TP", "FP", "FN")])
    T = te.group_by("country", "cls", "off").agg(pl.col("acc").sum().alias("acc_t"), (pl.col("acc") & ~pl.col("pw")).sum().alias("promo_t"))
    T = T.with_columns(*[(pl.col(c) / pl.col("country").replace_strict(nst) * 1000).alias(c) for c in ("acc_t", "promo_t")])
    rows.append(H.join(T, on=["country", "cls", "off"], how="full", coalesce=True).fill_null(0).with_columns(pl.lit(arm).alias("arm"), (pl.col("acc_t") - pl.col("TP")).alias("excess")))
X = pl.concat(rows)
W = X.pivot(on="arm", index=["country", "cls", "off"], values=["TP", "FP", "FN", "acc_t", "excess", "promo_t"]).fill_null(0)
W = W.with_columns((pl.col("excess_b") - pl.col("excess_a")).alias("d_excess"), (pl.col("TP_b") - pl.col("TP_a")).alias("d_holdTP"), (pl.col("FP_b") - pl.col("FP_a")).alias("d_holdFP"))
print(W.select("country", "cls", "off", *[pl.col(c).round(2) for c in ("TP_a", "TP_b", "FP_a", "FP_b", "acc_t_a", "acc_t_b", "excess_a", "excess_b", "d_excess")])
      .filter((pl.col("acc_t_a") + pl.col("acc_t_b")) > 4).sort("country", "d_excess"))
for arm in ("a", "b"):
    te = R[arm]["te"].with_columns(pl.col("off").alias("o"))
    sym = {c: (te.filter((pl.col("country") == c) & pl.col("acc") & (pl.col("o") == "+1..9")).height - te.filter((pl.col("country") == c) & pl.col("acc") & (pl.col("o") == "-1..9")).height) / nst[c] * 1000 for c in ("US", "India")}
    promo = {c: te.filter((pl.col("country") == c) & pl.col("acc") & ~pl.col("pw")).height / nst[c] * 1000 for c in ("US", "India")}
    print(f"ARM {arm.upper()}: test +k minus -k accepted per 1k (twin excess) {({k: round(v, 2) for k, v in sym.items()})}; promoted per 1k {({k: round(v, 1) for k, v in promo.items()})}")
# holdout-referenced value of B's test changes vs A (per country)
val = {}
for c in ("US", "India"):
    ha, hb = R["a"]["h"].filter(pl.col("country") == c), R["b"]["h"].filter(pl.col("country") == c)
    k = 1000 / nsh[c]
    hr = ha.filter("acc").select("s1", "m", "y", "cls", "off").join(hb.filter("acc").select("s1", "m"), on=["s1", "m"], how="anti")
    hd = hb.filter("acc").select("s1", "m", "y", "cls", "off").join(ha.filter("acc").select("s1", "m"), on=["s1", "m"], how="anti")
    ta, tb = R["a"]["te"].filter((pl.col("country") == c) & pl.col("acc")), R["b"]["te"].filter((pl.col("country") == c) & pl.col("acc"))
    tr_ = ta.select("s1", "m", "cls", "off").join(tb.select("s1", "m"), on=["s1", "m"], how="anti").group_by("cls", "off").len().with_columns(pl.col("len") / nst[c] * 1000).rename({"len": "rem_t"})
    td_ = tb.select("s1", "m", "cls", "off").join(ta.select("s1", "m"), on=["s1", "m"], how="anti").group_by("cls", "off").len().with_columns(pl.col("len") / nst[c] * 1000).rename({"len": "add_t"})
    hr_ = hr.group_by("cls", "off").agg((pl.col("y").sum() * k).alias("remTP"), ((~pl.col("y")).sum() * k).alias("remFP"))
    hd_ = hd.group_by("cls", "off").agg((pl.col("y").sum() * k).alias("addTP"), ((~pl.col("y")).sum() * k).alias("addFP"))
    z = tr_.join(td_, on=["cls", "off"], how="full", coalesce=True).join(hr_, on=["cls", "off"], how="full", coalesce=True).join(hd_, on=["cls", "off"], how="full", coalesce=True).fill_null(0)
    z = z.with_columns((pl.col("rem_t") - pl.col("remTP") - pl.col("remFP")).alias("rex"), (pl.col("add_t") - pl.col("addTP") - pl.col("addFP")).alias("aex"))
    z = z.with_columns((0.21 * pl.col("remFP") - 0.09 * pl.col("remTP") + 0.21 * pl.col("rex").clip(0) - 0.09 * (-pl.col("rex")).clip(0)
                        + 0.09 * pl.col("addTP") - 0.21 * pl.col("addFP") - 0.21 * pl.col("aex").clip(0)).alias("value"))
    val[c] = float(z["value"].sum()) / 1000
    print(f"\n{c}: B vs A test changes: removed {z['rem_t'].sum():.1f}/1k, added {z['add_t'].sum():.1f}/1k -> holdout-referenced value {val[c]:+.5f} ({c} F)")
    print(z.sort("value", descending=True).select("cls", "off", *[pl.col(x).round(2) for x in ("rem_t", "remTP", "remFP", "add_t", "addTP", "addFP", "value")]).head(10))
wUS, wIN = nst["US"], nst["India"]
print(f"\nB vs A estimated LB value (US+India share of all test S1): {(val['US'] * wUS + val['India'] * wIN) / sum(nst.values()):+.5f}; "
      f"holdout clean delta US {R['b']['holdout']['US'] - R['a']['holdout']['US']:+.5f} India {R['b']['holdout']['India'] - R['a']['holdout']['India']:+.5f}")
json.dump({arm: dict(V=R[arm]["V"], C=R[arm]["C"], valid=R[arm]["valid"], holdout=R[arm]["holdout"]) for arm in R} | {"value": val}, open("analysis/out/219_ablation_eval.json", "w"), indent=1, default=str)
