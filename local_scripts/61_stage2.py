"""Stage-2 collective features on the dev set. OOF stage-1 (5 folds by S1) -> group/sibling aggregates -> stage-2 LightGBM.
Reports holdout macro-F0.5 for stage-1 vs stage-2 and per segment."""
import sys; sys.path.insert(0, "scripts"); import memguard
import polars as pl, numpy as np, json, time
from devlib import *
from retrieval2 import core2_expr
t0 = time.time()
d, gt = load_dev()
pn = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", core2_expr().alias("pc2"), pl.col("a").alias("pa")) for s in (2, 3)]) \
       .join(d.lazy().select(pl.col("m").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
d = d.join(pn, left_on="m", right_on="entity_id").with_columns(pl.col("pc2").str.replace_all(" ", "").alias("pns"))
print("loaded", d.height, f"{time.time()-t0:.0f}s", flush=True)
# ---- OOF stage-1 ----
d = d.with_columns((pl.col("fold") % 5).alias("f5"))
oof = np.zeros(d.height, np.float32); f5 = d["f5"].to_numpy()
for k in range(5):
    tr = d.filter(pl.col("f5") != k); va_in = tr.filter(pl.col("f5") == (k + 1) % 5); tr_in = tr.filter(pl.col("f5") != (k + 1) % 5)
    m = fit(tr_in, va_in, BASE)
    oof[f5 == k] = m.predict(X(d.filter(pl.col("f5") == k), BASE))
    print("oof fold", k, m.best_iteration, f"{time.time()-t0:.0f}s", flush=True)
d = d.with_columns(pl.Series("p1", oof))
# ---- stage-2 aggregates ----
def top2(col, grp):  # max and 2nd max of col within group
    return [pl.col(col).max().over(grp).alias(f"{col}_max_{'_'.join(grp)}"),
            pl.col(col).sort(descending=True).get(1, null_on_oob=True).over(grp).fill_null(0).alias(f"{col}_2nd_{'_'.join(grp)}")]
d = d.with_columns(*top2("p1", ["s1"]), *top2("p1", ["m"]),
    pl.col("p1").rank("ordinal", descending=True).over("s1").alias("p1_rank_s1"),
    (pl.col("p1") > 0.5).sum().over("s1").alias("s1_n_p50"), pl.col("p1").sum().over("s1").alias("s1_sum_p1"),
    pl.len().over("m").alias("m_ncand_dev"))
d = d.with_columns(
    pl.when(pl.col("p1") == pl.col("p1_max_s1")).then(pl.col("p1_2nd_s1")).otherwise(pl.col("p1_max_s1")).alias("s1_best_other"),
    pl.when(pl.col("p1") == pl.col("p1_max_m")).then(pl.col("p1_2nd_m")).otherwise(pl.col("p1_max_m")).alias("m_best_other_s1"))
# sibling support: other candidates of the SAME S1 sharing identical pool core name / address / space-free name
for key, nm in [("pc2", "sib_name"), ("pa", "sib_addr"), ("pns", "sib_ns")]:
    g = ["s1", key]
    d = d.with_columns(*top2("p1", g), pl.len().over(g).alias(f"{nm}_n"))
    mx, sd = f"p1_max_{'_'.join(g)}", f"p1_2nd_{'_'.join(g)}"
    d = d.with_columns(pl.when(pl.col(f"{nm}_n") <= 1).then(-1.0).when(pl.col("p1") == pl.col(mx)).then(pl.col(sd)).otherwise(pl.col(mx)).alias(f"{nm}_max"))
    if key == "pa": d = d.with_columns(pl.when(pl.col("pa") == "").then(-1.0).otherwise(pl.col("sib_addr_max")).alias("sib_addr_max"))
S2 = ["p1", "p1_max_s1", "s1_best_other", "p1_rank_s1", "s1_n_p50", "s1_sum_p1", "m_best_other_s1", "m_ncand_dev",
      "sib_name_max", "sib_name_n", "sib_addr_max", "sib_addr_n", "sib_ns_max", "sib_ns_n"]
tr, va, ho = [d.filter(pl.col("split") == s) for s in ("train", "valid", "hold")]
gv, gh = gt.filter(pl.col("split") == "valid"), gt.filter(pl.col("split") == "hold")
R = {}
for name, cols in [("stage1", BASE), ("stage2", BASE + S2)]:
    m = fit(tr, va, cols); pv, ph = m.predict(X(va, cols)), m.predict(X(ho, cols))
    for pol in ("none", "best_s1"):
        fv, t = best_thr(va, pv, gv, pol); fh, det = macro_f05(ho, ph, t, gh, pol)
        R[f"{name}_{pol}"] = dict(valid=fv, thr=t, hold=fh, **det)
        print(f"{name:7s} {pol:8s} valid={fv:.4f} thr={t:.2f} HOLD={fh:.4f} {det}", flush=True)
    if name == "stage2":
        imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda z: -z[1]); R["imp"] = [(a, float(b)) for a, b in imp[:25]]
        print("top gain:", [(a, round(b / imp[0][1], 3)) for a, b in imp[:15]])
        ho2 = ho.with_columns(pl.Series("p", ph))
        seg = pl.when(pl.col("a2_empty") == 1).then(pl.lit("addr_empty")).when(pl.col("brand2") == 1).then(pl.lit("brand")).when(pl.col("dom2") == 1).then(pl.lit("domain")) \
                .when(pl.col("n_tsr") < 50).then(pl.lit("weakname")).when(pl.col("a_tsr") < 70).then(pl.lit("weakaddr")).otherwise(pl.lit("normal"))
        print(ho2.with_columns(seg.alias("seg"), (pl.col("p") >= R["stage2_none"]["thr"]).alias("pred")).group_by("seg").agg(
            pl.col("pred").filter(pl.col("y")).mean().alias("recall"), (pl.col("pred") & ~pl.col("y")).sum().alias("fp"), pl.col("y").sum().alias("pos")).sort("pos", descending=True))
json.dump(R, open("analysis/out/61_stage2.json", "w"), indent=1, default=float)
d.select("s1", "m", "y", "split", "fold", "p1").write_parquet("work/dev_oof_p1.parquet")
print(memguard.report(), f"{time.time()-t0:.0f}s")
