"""E6: add house-number closeness features (min digit edit distance incl. truncation) and retrain LightGBM; holdout macro-F0.5."""
import sys; sys.path.insert(0, "scripts"); import memguard
import polars as pl, numpy as np, lightgbm as lgb, json
from rapidfuzz.distance import Levenshtein
src = open("scripts/43_models.py").read()
exec(src.split("# ---------- E1")[0].split("R = {}")[1])
nm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "nums") for s in (1, 2, 3)]).collect()
def close(a, b):
    A = a.split(); B = b.split()
    if not A or not B: return -1
    best = 9
    for x in A:
        for z in B:
            dd = Levenshtein.distance(x, z)
            if dd and (x.startswith(z) or z.startswith(x)): dd = 1
            best = min(best, dd)
    return best
def first_ed(a, b):
    A = a.split(); B = b.split()
    return Levenshtein.distance(A[0], B[0]) if A and B else -1
d = d.join(nm, left_on="s1", right_on="entity_id").join(nm, left_on="m", right_on="entity_id", suffix="_2")
A, B = d["nums"].to_list(), d["nums_2"].to_list()
d = d.with_columns(pl.Series("num_min_ed", [close(a, b) for a, b in zip(A, B)]), pl.Series("num_first_ed", [first_ed(a, b) for a, b in zip(A, B)])).drop("nums", "nums_2")
tr, va, ho = [d.filter(pl.col("split") == s) for s in ("train", "valid", "hold")]
ALL2 = ALL + ["num_min_ed", "num_first_ed"]
m = fit_lgb(ALL2, trd=tr)
pv = m.predict(X(va, ALL2)); f, thr = best_thr(va, pv)
ph = m.predict(X(ho, ALL2)); fh, det = macro_f05(ho, ph, thr, split="hold")
print(f"E6 +number closeness: valid F0.5={f:.4f} thr={thr:.2f} | holdout F0.5={fh:.4f} {det}")
json.dump(dict(valid=f, thr=float(thr), hold=fh, **det), open("analysis/out/48_e6.json", "w"), indent=1)
m.save_model("work/dev_lgb_all_numfeat.txt")
print(memguard.report())
