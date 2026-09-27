"""Learning curve: stage-1 LightGBM trained on 12.5/25/50/100% of dev-train S1 (grouped), holdout macro-F0.5."""
import sys; sys.path.insert(0, "scripts"); import memguard
import polars as pl, numpy as np, json
from devlib import *
d, gt = load_dev()
tr, va, ho = [d.filter(pl.col("split") == s) for s in ("train", "valid", "hold")]
gv, gh = gt.filter(pl.col("split") == "valid"), gt.filter(pl.col("split") == "hold")
R = {}
for frac in (0.125, 0.25, 0.5, 1.0):
    t = tr.filter((pl.col("s1").hash(7) % 1000) < int(frac * 1000))
    m = fit(t, va, BASE); pv, ph = m.predict(X(va, BASE)), m.predict(X(ho, BASE))
    fv, th = best_thr(va, pv, gv); fh, det = macro_f05(ho, ph, th, gh)
    R[frac] = dict(n_s1=t["s1"].n_unique(), rows=t.height, valid=fv, hold=fh, thr=th, **det)
    print(f"frac={frac:5.3f} S1={t['s1'].n_unique():6d} valid={fv:.4f} hold={fh:.4f} P={det['P']:.4f} R={det['R']:.4f}", flush=True)
json.dump(R, open("analysis/out/63_learning_curve.json", "w"), indent=1)
print(memguard.report())
