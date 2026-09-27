"""Stage-3: iterate collective features with OOF stage-2 probabilities. Compare stage2 (coll(p1)) vs stage3 (coll(p1)+coll(p2oof))."""
import sys; sys.path.insert(0, "scripts"); import memguard
import polars as pl, numpy as np, json, time
from devlib import *
from collective import add_collective
from retrieval2 import core2_expr
t0 = time.time()
d, gt = load_dev()
pn = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", core2_expr().alias("pc2"), pl.col("a").alias("pa"),
        pl.col("nums").str.split(" ").list.first().alias("pnum1"), pl.col("a").str.extract(r"([0-9]+ [a-z]{3,})", 1).alias("pstreet")) for s in (2, 3)]) \
       .join(d.lazy().select(pl.col("m").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
d = d.join(pn, left_on="m", right_on="entity_id").with_columns(pl.col("pc2").str.replace_all(" ", "").alias("pns"), (pl.col("fold") % 5).alias("f5"))
d = d.join(pl.read_parquet("work/dev_oof_p1.parquet").select("s1", "m", "p1"), on=["s1", "m"])
d, C1 = add_collective(d, "p1", "c1")
F2 = BASE + ["p1"] + C1
def oof(d, cols, name):
    o = np.zeros(d.height, np.float32); f5 = d["f5"].to_numpy()
    for k in range(5):
        tr = d.filter(pl.col("f5") != k)
        m = fit(tr.filter(pl.col("f5") != (k + 1) % 5), tr.filter(pl.col("f5") == (k + 1) % 5), cols)
        o[f5 == k] = m.predict(X(d.filter(pl.col("f5") == k), cols))
    print("oof", name, f"{time.time()-t0:.0f}s", flush=True)
    return o
d = d.with_columns(pl.Series("p2o", oof(d, F2, "stage2")))
d, C2 = add_collective(d, "p2o", "c2")
F3 = F2 + ["p2o"] + C2
tr, va, ho = [d.filter(pl.col("split") == s) for s in ("train", "valid", "hold")]
gv, gh = gt.filter(pl.col("split") == "valid"), gt.filter(pl.col("split") == "hold")
R = {}
for name, cols in [("stage2", F2), ("stage3", F3)]:
    m = fit(tr, va, cols); pv, ph = m.predict(X(va, cols)), m.predict(X(ho, cols))
    fv, t = best_thr(va, pv, gv); fh, det = macro_f05(ho, ph, t, gh)
    R[name] = dict(valid=fv, thr=t, hold=fh, **det); print(f"{name} valid={fv:.4f} thr={t:.2f} HOLD={fh:.4f} {det}", flush=True)
json.dump(R, open("analysis/out/69_stage3.json", "w"), indent=1)
print(memguard.report(), f"{time.time()-t0:.0f}s")
