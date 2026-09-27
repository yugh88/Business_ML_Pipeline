"""Mechanism test — COPY CHANNEL: the generator applies formatting/noise ops to copies of an S1, while look-alike distractors
are generated separately. Normalization erases that evidence (case, accents, punctuation, exact raw equality, address layout).
Holdout (V9 p3, folds 0/8/9, real records only): per relation class x raw-evidence bucket -> true rate vs V9 mean p3, acceptance,
FP/FN (is V9 blind to it?). Per-op rates for true vs false candidates. Test: accepted/candidate density per bucket."""
import sys, re, unicodedata, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from rapidfuzz.distance import Levenshtein
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
d9 = json.load(open("output_v9/decision_s3.json"))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
OCR = re.compile(r"[A-Za-z][0158][A-Za-z]|\b[0158][a-z]{2,}")
BR = re.compile(r"[\[\]\(\)#@*~_|]|  |--|^\W|\W$")
PLACE = re.compile(r"(?i)\b(n/a|null|none|unknown)\b")
ZP = re.compile(r"(^|[\s,#])0\d")
OPS = ["ocr", "accent", "punct", "placeholder", "zeropad", "typo", "dupword", "case", "addr_reordered"]


def accents(s):
    return sum(1 for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) == "Mn")


def case_style(s):
    return "U" if s == s.upper() else ("L" if s == s.lower() else ("T" if s == s.title() else "M"))


def evidence(n1, a1, n2, a2):
    """raw-text evidence of the copy channel for pool record (n2, a2) relative to S1 (n1, a1)."""
    o = dict.fromkeys(OPS, 0)
    o["ocr"] = int(bool(OCR.search(n2)) and not OCR.search(n1))
    o["accent"] = int(accents(n2) > accents(n1))
    o["punct"] = int(bool(BR.search(n2)) and not BR.search(n1))
    o["placeholder"] = int(bool(PLACE.search(a2)) and not PLACE.search(a1))
    o["zeropad"] = int(bool(ZP.search(a2)) and not ZP.search(a1))
    A, B = n1.lower().split(), n2.lower().split()
    o["typo"] = int(any(len(t) >= 4 and t not in A and any(1 <= Levenshtein.distance(t, u) <= 2 for u in A if len(u) >= 4) for t in B))
    o["dupword"] = int(len(B) != len(set(B)) and len(A) == len(set(A)))
    o["case"] = int(case_style(n2) != case_style(n1))
    c1 = [c.strip().lower() for c in a1.split(",") if c.strip()]; c2 = [c.strip().lower() for c in a2.split(",") if c.strip()]
    o["addr_reordered"] = int(len(c1) > 1 and sorted(c1) == sorted(c2) and c1 != c2)
    bucket = ("clone" if (n2 == n1 and a2 == a1) else "name_clone" if n2 == n1 else "addr_clone" if a2 == a1 else
              "noisy" if sum(o.values()) else "plain")
    return bucket, sum(o.values()), [o[k] for k in OPS]


def klass(nrel, arel, c1, c2):
    same = arel in ("A0_identical", "A1_same_num")
    if nrel == "N0_identical":
        return "identical@same" if same else "identical@num_changed" if arel.startswith("A2") else "identical@empty" if arel == "A3_empty" else "identical@other"
    if not same: return "other"
    if nrel in ("N1_extra_noise", "N4_dropped_words", "N5_typo_subst"): return "light_edit@same"
    if nrel in ("N2_extra_DESCRIPTOR", "N6_word_swap_DESC"): return "desc@same"
    if nrel == "N7_no_overlap": return "no_overlap@same"
    a, b = c1.split(), c2.split()
    if nrel == "N6_word_swap" and a and a[0] not in b: return "first_word_replaced@same"
    return "swap_extra@same"


def annotate(x, split):
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in nm.iter_rows()}; del nm
    raw = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
                     for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    R = {r[0]: r[1:] for r in raw.iter_rows()}; del raw
    cls, bk, nops, opv = [], [], [], []
    for a, b in zip(x["s1"].to_list(), x["m"].to_list()):
        na, nb = N[a], N[b]
        cls.append(klass(name_rel(na[0], nb[0]), addr_rel(na[1], nb[1], na[2], nb[2]), na[0], nb[0]))
        e = evidence(R[a][0], R[a][1], R[b][0], R[b][1]); bk.append(e[0]); nops.append(e[1]); opv.append(e[2])
    cols = {k: [v[i] for v in opv] for i, k in enumerate(OPS)}
    return x.with_columns(pl.Series("cls", cls), pl.Series("bucket", bk), pl.Series("nops", nops), *[pl.Series(k, v, dtype=pl.Int8) for k, v in cols.items()])


def cands(p3):
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    return p3.filter(pl.col("p1") >= d9["tau"]).join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))


pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_hide_dataframe_shape(True)
h = cands(pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti"))
h = annotate(h, "train")
h.select("s1", "m", "y", "p3", "acc", "cls", "bucket", "nops", *OPS).write_parquet("work/copy_channel_holdout.parquet")
print(f"HOLDOUT candidates {h.height}  (true {int(h['y'].sum())})")
print("\n== per-op rate among TRUE vs FALSE candidates (same-address classes only), and V9 FP/FN share with the op")
sa = h.filter(pl.col("cls").str.ends_with("@same"))
print(pl.DataFrame([(k, round(float(sa.filter(pl.col("y"))[k].mean()), 4), round(float(sa.filter(~pl.col("y"))[k].mean()), 4),
                     round(float(sa.filter(pl.col("acc") & ~pl.col("y"))[k].mean() or 0), 4), round(float(sa.filter(~pl.col("acc") & pl.col("y"))[k].mean() or 0), 4)) for k in OPS],
                   schema=["op", "rate_true", "rate_false", "rate_in_FP", "rate_in_FN"], orient="row"))
print("\n== class x bucket: true rate vs V9 mean p3 (calibration), acceptance, errors")
t = h.group_by("cls", "bucket").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate"), pl.col("p3").mean().round(3).alias("mean_p3"),
                                    pl.col("acc").mean().round(3).alias("acc_rate"), (pl.col("acc") & ~pl.col("y")).sum().alias("FP"), (~pl.col("acc") & pl.col("y")).sum().alias("FN"))
print(t.filter(pl.col("n") >= 200).sort("cls", "bucket"))
for c in ("US", "India", "France"):
    x = annotate(cands(pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"])), "test")
    ns = x["s1"].n_unique() / 1000
    print(f"\n== TEST {c}: per 1k S1 with candidates — candidates / accepted by class x bucket")
    print(x.group_by("cls", "bucket").agg((pl.len() / ns).round(1).alias("cand_per1k"), (pl.col("acc").sum() / ns).round(1).alias("acc_per1k"),
                                          pl.col("acc").mean().round(3).alias("acc_rate"), pl.col("p3").mean().round(3).alias("mean_p3")).filter(pl.col("cand_per1k") >= 0.5).sort("cls", "bucket"))
    x.select("s1", "m", "p3", "acc", "cls", "bucket", "nops", *OPS).write_parquet(f"work/copy_channel_test_{c}.parquet"); del x
