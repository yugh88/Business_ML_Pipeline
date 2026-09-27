"""Audit each normalization step: equality / similarity gain on TRUE pairs vs collision gain on HARD NEGATIVES
(dev_cands y=0, i.e. retrieved-but-wrong pairs). Sample: 150k positives + 150k hard negatives."""
import sys, re, polars as pl, numpy as np, json, unicodedata
sys.path.insert(0, "scripts"); import memguard
sys.path.insert(0, "scripts")
from norm import base_clean, alnum, anyascii, DOMAIN_RE, has_indic
import norm2
from rapidfuzz import process, fuzz
d = pl.read_parquet("work/dev_cands.parquet", columns=["s1", "m", "y"])
d = pl.concat([d.filter(pl.col("y")).sample(150000, seed=1), d.filter(~pl.col("y")).sample(150000, seed=1)])
raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet") for s in (1, 2, 3)])
d = d.join(raw.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"), "country").collect(), on="s1") \
     .join(raw.select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")).collect(), on="m")
y = d["y"].to_numpy(); ctry = d["country"].to_list()
N1, N2, A1, A2 = d["n1"].to_list(), d["n2"].to_list(), d["a1"].to_list(), d["a2"].to_list()
EXTRA_DROP = {"services", "service", "enterprises", "enterprise", "group", "center", "centre"}
HON = re.compile(r"^((shri|sri|smt|mr|mrs|ms|dr|m s|messrs) )+")
TAIL = re.compile(r" (id [0-9]+|[0-9]{6,}|com|c0m|in|net|org)( |$).*$")
def steps_name(s):
    out = {}
    out["N0_lower_ws"] = " ".join(s.lower().split())
    b = base_clean(s); out["N1_clean_alnum"] = alnum(b)
    out["N2_anyascii"] = alnum(anyascii(b))
    nn, dom, ind = norm2.name_norm(s); out["N3_norm2(dict,domain,leet)"] = nn
    core = norm2.name_core(nn); out["N4_legal_drop"] = core
    out["N5_sorted"] = " ".join(sorted(core.split()))
    c2 = TAIL.sub("", HON.sub("", core)).strip(); c2 = c2[:-3] if c2.endswith("com") and len(c2) > 5 else c2
    out["N6_honorific_tail"] = " ".join(sorted(c2.split()))
    out["N7_drop_generic_words"] = " ".join(sorted(t for t in c2.split() if t not in EXTRA_DROP)) or out["N6_honorific_tail"]
    out["N8_nospace"] = out["N7_drop_generic_words"].replace(" ", "")
    return out
def steps_addr(s, c):
    out = {}
    out["A0_lower_ws"] = " ".join(s.lower().split())
    out["A1_alnum_anyascii"] = alnum(anyascii(base_clean(s))) if s else ""
    a, nums, st = norm2.addr_norm(s, c); out["A2_norm2(abbr,zeros,houseprefix)"] = a.replace(" , ", " ")
    toks = [t for t in out["A2_norm2(abbr,zeros,houseprefix)"].split() if t not in ("null", "n", "a", "na")]
    out["A3_drop_placeholders"] = " ".join(toks)
    out["A4_token_set"] = " ".join(sorted(set(toks)))
    return out, nums, st
ns1 = [steps_name(x) for x in N1]; ns2 = [steps_name(x) for x in N2]
res = {"name": {}, "addr": {}}
for k in ns1[0]:
    a = [r[k] for r in ns1]; b = [r[k] for r in ns2]
    eq = np.array([x == z and x != "" for x, z in zip(a, b)])
    tsr = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=4) if k != "N8_nospace" else process.cpdist(a, b, scorer=fuzz.ratio, workers=4)
    res["name"][k] = dict(eq_pos=eq[y].mean(), eq_neg=eq[~y].mean(), sim_pos=float(np.mean(tsr[y])), sim_neg=float(np.mean(tsr[~y])),
                          sep=float(np.mean(tsr[y]) - np.mean(tsr[~y])), pos_ge90=float((tsr[y] >= 90).mean()), neg_ge90=float((tsr[~y] >= 90).mean()))
    print(f"{k:32s} eq pos={eq[y].mean():.4f} neg={eq[~y].mean():.4f} | sim pos={np.mean(tsr[y]):.1f} neg={np.mean(tsr[~y]):.1f} | >=90 pos={(tsr[y]>=90).mean():.4f} neg={(tsr[~y]>=90).mean():.4f}", flush=True)
as1 = [steps_addr(x, c) for x, c in zip(A1, ctry)]; as2 = [steps_addr(x, c) for x, c in zip(A2, ctry)]
has2 = np.array([x != "" for x in A2])
for k in as1[0][0]:
    a = [r[0][k] for r in as1]; b = [r[0][k] for r in as2]
    tsr = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=4)
    jac = np.array([len(set(x.split()) & set(z.split())) / max(1, len(set(x.split()) | set(z.split()))) for x, z in zip(a, b)])
    m = has2
    res["addr"][k] = dict(tsr_pos=float(tsr[y & m].mean()), tsr_neg=float(tsr[~y & m].mean()), jac_pos=float(jac[y & m].mean()), jac_neg=float(jac[~y & m].mean()))
    print(f"{k:32s} tsr pos={tsr[y&m].mean():.1f} neg={tsr[~y&m].mean():.1f} | jaccard pos={jac[y&m].mean():.3f} neg={jac[~y&m].mean():.3f}", flush=True)
# numbers & state agreement
n1 = [set(r[1].split()) for r in as1]; n2 = [set(r[1].split()) for r in as2]
num_any = np.array([bool(x & z) for x, z in zip(n1, n2)]); both_nums = np.array([bool(x) and bool(z) for x, z in zip(n1, n2)])
st_eq = np.array([r1[2] == r2[2] and r1[2] != "" for r1, r2 in zip(as1, as2)]); st_known = np.array([r1[2] != "" and r2[2] != "" for r1, r2 in zip(as1, as2)])
st_dis = st_known & ~st_eq
for nm, v, msk in [("share_number|both_have", num_any, both_nums), ("state_equal|both_known", st_eq, st_known), ("state_disagree(all)", st_dis, np.ones_like(y))]:
    print(f"{nm:32s} pos={v[y & msk].mean():.4f} neg={v[~y & msk].mean():.4f}")
    res[nm] = dict(pos=float(v[y & msk].mean()), neg=float(v[~y & msk].mean()))
json.dump(res, open("analysis/out/41_norm_audit.json", "w"), indent=1, default=float)
