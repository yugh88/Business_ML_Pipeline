"""Targeted noise fixes for the gaps found in 231, each VALIDATED on the labelled holdout (US/India, clean, V11 final logic) before use:
 R1 order-invariant numbers : name identical, the SET of address numbers equal (>=1), address-token Jaccard >= 0.5 (order-free)
 R2 number parsing          : name identical, same street, first raw number (N° / bis / ter / # / letter suffix) == S1 number
 R3 typo / OCR names        : name differs only by typos or OCR digits (0->o 1->l 5->s 8->b), same address (A0/A1 or R1 numbers)
Candidates = V11-rejected tau-set pairs whose pool record is not accepted for any S1 (exclusivity), best p3 per record.
Reports per rule: adds/1k, precision, holdout F change per country; test adds per country + twin direction check."""
import sys, re, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from rapidfuzz.distance import Levenshtein
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
ALPHA = {"US": 0.75, "India": 0.75, "France": 1.0}
OCRMAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "3": "e", "4": "a"})
RAWNUM = re.compile(r"(?i)(?:n\s*°\s*|#\s*)?(\d+)")


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def decode(x, alpha):
    x = x.with_columns(pl.Series("ip1", iso.predict(x["p1"].to_numpy())).cast(pl.Float32))
    x = x.with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(x.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m")


def toks(a): return set(t for t in re.split(r"[ ,]+", a) if t)
def numset(nums): return set(nums.split()) if nums else set()


def rules(n1, n2, a1, a2, nu1, nu2, ra2):
    """-> (r1, r2, r3) booleans for pair S1 (n1 core, a1 norm addr, nu1 nums) vs pool (n2, a2, nu2, raw addr ra2)."""
    ident = n1 == n2
    t1, t2 = toks(a1), toks(a2)
    jac = len(t1 & t2) / max(1, len(t1 | t2))
    s1n, s2n = numset(nu1), numset(nu2)
    r1 = ident and bool(s1n) and s1n == s2n and jac >= 0.5
    st1, st2 = set(t for t in t1 if not t.isdigit()), set(t for t in t2 if not t.isdigit())
    sj = len(st1 & st2) / max(1, len(st1 | st2))
    m = RAWNUM.search(ra2 or ""); f1 = (nu1.split() or [""])[0]
    r2 = ident and not nu2 and bool(m) and bool(f1) and m.group(1).lstrip("0") == f1.lstrip("0") and sj >= 0.4
    arel = addr_rel(a1, a2, nu1, nu2)
    same_addr = arel in ("A0_identical", "A1_same_num") and bool(nu1) and bool(nu2)
    typo_ok = False
    if not ident:
        w1, w2 = n1.translate(OCRMAP).split(), n2.translate(OCRMAP).split()
        if len(w1) == len(w2) and sorted(w1) != sorted(w2):
            typo_ok = all(a == b or (len(a) >= 4 and Levenshtein.distance(a, b) <= 1) for a, b in zip(sorted(w1), sorted(w2)))
        elif sorted(w1) == sorted(w2):
            typo_ok = True                                                   # OCR-only difference
    r3 = typo_ok and (same_addr or (bool(s1n) and s1n == s2n and jac >= 0.5))
    return r1, r2, r3


def candidates(split, x, acc):
    rej = x.join(acc.with_columns(pl.lit(True).alias("a")), on=["s1", "m"], how="left").filter(pl.col("a").is_null()).drop("a") \
           .join(acc.select("m").unique(), on="m", how="anti")
    ids = pl.concat([rej.select(pl.col("s1").alias("entity_id")), rej.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", pl.col("nums").fill_null("")) for s in (1, 2, 3)]) \
           .join(ids.lazy(), on="entity_id", how="semi").collect()
    raw = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", pl.col("business_address").fill_null("")) for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in nm.iter_rows()}; RA = dict(raw.iter_rows())
    out = [rules(N[a][0], N[b][0], N[a][1], N[b][1], N[a][2], N[b][2], RA.get(b, "")) if a in N and b in N else (False, False, False)
           for a, b in zip(rej["s1"].to_list(), rej["m"].to_list())]
    return rej.with_columns(*[pl.Series(k, [o[i] for o in out]) for i, k in enumerate(("R1", "R2", "R3"))])


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti").filter(pl.col("p1") >= TAU).join(t1, on="s1")
acc_h = pl.concat([decode(h.filter(pl.col("country") == c), ALPHA[c]) for c in ("US", "India")])
cand_h = candidates("train", h, acc_h)
base = gt.join(acc_h.join(h.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
f0 = F(base["tp"], base["fp"], base["n"])
print("HOLDOUT validation (V11 final logic as baseline):")
VALID = {}
for rule in ("R1", "R2", "R3", "R1|R2|R3"):
    flt = pl.any_horizontal([pl.col(r) for r in rule.split("|")])
    ad = cand_h.filter(flt).sort("p3", descending=True).unique(subset="m", keep="first")
    z = base.join(ad.group_by("s1").agg(pl.col("y").sum().alias("atp"), (~pl.col("y")).sum().alias("afp")), on="s1", how="left").fill_null(0)
    d = F(z["tp"] + z["atp"], z["fp"] + z["afp"], z["n"]) - F(z["tp"], z["fp"], z["n"])
    res = {c: float(d[(z["country"] == c).to_numpy()].mean()) for c in ("US", "India")}
    VALID[rule] = (float(ad["y"].mean() or 0), res)
    print(f"  {rule:9s}: adds {ad.height / gt.height * 1000:5.2f}/1k S1, precision {ad['y'].mean() or 0:.3f} -> holdout F US {res['US']:+.5f}  India {res['India']:+.5f}", flush=True)
nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
print("TEST (would add, per 1k S1):")
tests = {}
for c in ("US", "India", "France"):
    x = pl.read_parquet(f"{S3D}/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= TAU)
    ct = candidates("test", x, decode(x, ALPHA[c]))
    tests[c] = ct
    print(f"  {c:6s}: " + "  ".join(f"{r} {ct.filter(pl.col(r)).sort('p3', descending=True).unique(subset='m').height / nt[c] * 1000:.2f}" for r in ("R1", "R2", "R3")))
    ct.filter(pl.col("R1") | pl.col("R2") | pl.col("R3")).select("s1", "m", "p3", "R1", "R2", "R3").write_parquet(f"work/noisefix_{c}.parquet")
json.dump({k: v for k, v in VALID.items()}, open("analysis/out/232_noise_fixes.json", "w"), indent=1)
