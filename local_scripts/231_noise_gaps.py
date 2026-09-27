"""Systematic noise-gap audit on the LABELLED holdout (US/India, folds 0/8/9, clean) for the FINAL V11 decision logic
(stage-3 per-country ensemble + alpha-0.75 tempering). For every raw-text noise type (pool record vs its S1): among TRUE candidates the
miss rate (FN) with vs without the noise, among FALSE candidates the accept rate (FP) with vs without; excess errors and their F value.
A noise type is a GAP when its excess errors are material -> candidate for a targeted fix. Also scans French test acceptance by noise."""
import sys, re, json, unicodedata
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from rapidfuzz.distance import Levenshtein
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
ALPHA = {"US": 0.75, "India": 0.75, "France": 1.0}

NUM_LOST = re.compile(r"(?i)(n\s*°\s*\d|\b\d+\s*(bis|ter|quater)\b|\b\d+[a-z]\b|#\s*\d)")
PH = re.compile(r"(?i)\b(n/a|null|none|unknown|na)\b")
IDSUF = re.compile(r"(?i)(\bid\s*[:#]?\s*\d{3,}|\b\d{6,}\b|\(\s*\d{3,}\s*\))")
PREFIX = re.compile(r"(?i)^\s*(m/s\.?|m/s|messrs\.?|the)\s+")
ALIAS = re.compile(r"(?i)\b(dba|d/b/a|t/a|aka|a/k/a|fka|f/k/a|trading as|formerly)\b")
DOMAIN = re.compile(r"(?i)(\.com|\.in|\.fr|\.net|\.org|www\.|@)")
OCR = re.compile(r"[A-Za-z][0158][A-Za-z]|\b[0158][a-z]{2,}")
BR = re.compile(r"[\[\]\(\)*~_|]|  ")
UNIT = re.compile(r"(?i)\b(unit|suite|ste|apt|pmb|fl|floor|flat|room|shop)\b")
CITYPFX = re.compile(r"(?i)\b(city of|village of|town of|township|county)\b")


def accents(s): return sum(1 for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) == "Mn")


def flags(n1, a1, n2, a2, nums2):
    f = {}
    f["addr_empty"] = a2.strip() == ""
    f["num_lost_in_parsing"] = bool(re.search(r"\d", a2)) and not nums2 and not f["addr_empty"]
    f["num_format_Nbis#"] = bool(NUM_LOST.search(a2))
    f["addr_placeholder"] = bool(PH.search(a2)) and not PH.search(a1)
    f["addr_unit_added"] = bool(UNIT.search(a2)) and not UNIT.search(a1)
    f["addr_city_prefix"] = bool(CITYPFX.search(a2)) != bool(CITYPFX.search(a1))
    c1 = [c.strip().lower() for c in a1.split(",") if c.strip()]; c2 = [c.strip().lower() for c in a2.split(",") if c.strip()]
    f["addr_reordered"] = len(c2) > 1 and c1[:1] != c2[:1] and bool(set(c1) & set(c2))
    f["name_id_or_phone"] = bool(IDSUF.search(n2)) and not IDSUF.search(n1)
    f["name_prefix_Ms"] = bool(PREFIX.search(n2)) and not PREFIX.search(n1)
    f["name_alias_marker"] = bool(ALIAS.search(n2)) and not ALIAS.search(n1)
    f["name_domain"] = bool(DOMAIN.search(n2)) and not DOMAIN.search(n1)
    f["name_ocr_digit"] = bool(OCR.search(n2)) and not OCR.search(n1)
    f["name_accent_added"] = accents(n2) > accents(n1)
    f["name_brackets_space"] = bool(BR.search(n2)) and not BR.search(n1)
    f["name_case_changed"] = (n2 == n2.upper()) != (n1 == n1.upper())
    A, B = n1.lower().split(), n2.lower().split()
    f["name_typo"] = any(len(t) >= 4 and t not in A and any(1 <= Levenshtein.distance(t, u) <= 2 for u in A if len(u) >= 4) for t in B)
    f["name_word_order"] = sorted(A) == sorted(B) and A != B
    return f


def load_pairs(split, pdir, folds=None, country=None):
    if split == "train":
        x = pl.concat([pl.read_parquet(f"{pdir}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in folds]).join(SYN, on="m", how="anti")
    else:
        x = pl.read_parquet(f"{pdir}/p3/test/{country}.parquet", columns=["s1", "m", "p1", "p3"]).with_columns(pl.lit(False).alias("y"))
    return x.filter(pl.col("p1") >= TAU)


def decode(x, alpha):
    x = x.with_columns(pl.Series("ip1", iso.predict(x["p1"].to_numpy())).cast(pl.Float32))
    x = x.with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(x.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m").with_columns(pl.lit(True).alias("acc"))


def annotate_flags(x, split):
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    raw = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")) for s in (1, 2, 3)]) \
            .join(ids.lazy(), on="entity_id", how="semi").collect()
    nums = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", pl.col("nums").fill_null("")) for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    R = {e: (n, a) for e, n, a in raw.iter_rows()}; NU = dict(nums.iter_rows())
    rows = [flags(R[a][0], R[a][1], R[b][0], R[b][1], NU.get(b, "")) for a, b in zip(x["s1"].to_list(), x["m"].to_list())]
    keys = list(rows[0].keys())
    return x.with_columns(*[pl.Series(k, [r[k] for r in rows]) for k in keys]), keys


t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
h = load_pairs("train", S3D, [0, 8, 9]).join(t1, on="s1")
acc = pl.concat([decode(h.filter(pl.col("country") == c), ALPHA[c]) for c in ("US", "India")])
h = h.join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
h, KEYS = annotate_flags(h, "train")
ns = dict(h.group_by("country").agg(pl.col("s1").n_unique()).rows())
rows = []
for c in ("US", "India"):
    z = h.filter(pl.col("country") == c)
    T, Fz = z.filter(pl.col("y")), z.filter(~pl.col("y"))
    fn0, fp0 = 1 - T["acc"].mean(), Fz["acc"].mean()
    for k in KEYS:
        tw, fw = T.filter(pl.col(k)), Fz.filter(pl.col(k))
        fnr = 1 - tw["acc"].mean() if tw.height else 0.0; fpr = fw["acc"].mean() if fw.height else 0.0
        fnr_wo = 1 - T.filter(~pl.col(k))["acc"].mean(); fpr_wo = Fz.filter(~pl.col(k))["acc"].mean()
        ex_fn = max(0.0, (fnr - fnr_wo)) * tw.height / ns[c]; ex_fp = max(0.0, (fpr - fpr_wo)) * fw.height / ns[c]
        rows.append((c, k, tw.height, round(fnr, 4), round(fnr_wo, 4), fw.height, round(fpr, 4), round(fpr_wo, 4), round(ex_fn * 1000, 2), round(ex_fp * 1000, 2),
                     round((ex_fn * 0.09 + ex_fp * 0.21), 5)))
R = pl.DataFrame(rows, schema=["country", "noise", "true_with", "FN_rate_with", "FN_rate_without", "false_with", "FP_rate_with", "FP_rate_without",
                               "excess_FN_per1k", "excess_FP_per1k", "F_value_if_fixed"], orient="row")
print("HOLDOUT (V11 final logic): noise types ranked by F value of closing their excess errors")
print(R.sort("F_value_if_fixed", descending=True).head(24))
R.write_parquet("work/noise_gaps_holdout.parquet")
# France: acceptance with vs without each noise (label-free), among identical-name same-street candidates (should be ~copies)
x = load_pairs("test", S3D, country="France")
x = x.join(decode(x, 1.0), on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
x = annotate(x, "test").filter(pl.col("cls").is_in(["identical@same", "light_edit@same", "identical@empty"]))
x, _ = annotate_flags(x, "test")
fr = []
for k in KEYS:
    w = x.filter(pl.col(k)); wo = x.filter(~pl.col(k))
    if w.height >= 200:
        fr.append((k, w.height, round(w["acc"].mean(), 4), round(wo["acc"].mean(), 4)))
print("\nFRANCE test (identical / light-edit / empty-address names): acceptance WITH vs WITHOUT each noise (a big drop = noise-induced rejection)")
print(pl.DataFrame(fr, schema=["noise", "n_with", "acc_with", "acc_without"], orient="row").with_columns((pl.col("acc_with") - pl.col("acc_without")).round(4).alias("diff")).sort("diff"))
