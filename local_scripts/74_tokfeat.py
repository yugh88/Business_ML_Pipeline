"""Stage-2b: stage-2 + sibling CONSENSUS features (house number, street, anchor-copy similarity).
Stage-2 collective features on the dev set. OOF stage-1 (5 folds by S1) -> group/sibling aggregates -> stage-2 LightGBM.
Reports holdout macro-F0.5 for stage-1 vs stage-2 and per segment."""
import sys; sys.path.insert(0, "scripts"); import memguard
import polars as pl, numpy as np, json, time
from devlib import *
from retrieval2 import core2_expr
t0 = time.time()
d, gt = load_dev()
pn = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", core2_expr().alias("pc2"), pl.col("a").alias("pa"),
        pl.col("nums").str.split(" ").list.first().alias("pnum1"), pl.col("a").str.extract(r"([0-9]+ [a-z]{3,})", 1).alias("pstreet")) for s in (2, 3)]) \
       .join(d.lazy().select(pl.col("m").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
d = d.join(pn, left_on="m", right_on="entity_id").with_columns(pl.col("pc2").str.replace_all(" ", "").alias("pns"))
print("loaded", d.height, f"{time.time()-t0:.0f}s", flush=True)
d = d.join(pl.read_parquet("work/dev_oof_p1.parquet").select("s1", "m", "p1"), on=["s1", "m"])  # OOF stage-1 from 61_stage2.py
# ---- number-perturbation type features (analysis/out/72) ----
nm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", pl.col("nums").str.split(" ").list.first().fill_null("").alias("n1")) for s in (1, 2, 3)]).collect()
x = d.select("s1", "m").join(nm, left_on="s1", right_on="entity_id").join(nm.rename({"n1": "n2"}), left_on="m", right_on="entity_id")
KIND = {"missing": 0, "equal": 1, "prefix_trunc": 2, "suffix_trunc": 3, "1digit_first": 4, "1digit_last": 5, "1digit_mid": 6, "transpose": 7, "same_len_multi": 8, "len_change_other": 9}
def kind(a, b):
    if not a or not b: return 0
    if a == b: return 1
    if b.startswith(a) or a.startswith(b): return 2
    if b.endswith(a) or a.endswith(b): return 3
    if len(a) == len(b):
        dp = [i for i in range(len(a)) if a[i] != b[i]]
        if len(dp) == 1: return 4 if dp[0] == 0 else (5 if dp[0] == len(a) - 1 else 6)
        if len(dp) == 2 and dp[1] == dp[0] + 1 and a[dp[0]] == b[dp[1]] and a[dp[1]] == b[dp[0]]: return 7
        return 8
    return 9
A, Bn = x["n1"].to_list(), x["n2"].to_list()
x = x.with_columns(pl.Series("num_kind", [kind(a, b) for a, b in zip(A, Bn)], dtype=pl.Int8),
                   pl.Series("num_logdiff", [float(np.log1p(abs(int(a[:9]) - int(b[:9])))) if a and b else -1.0 for a, b in zip(A, Bn)], dtype=pl.Float32)).select("s1", "m", "num_kind", "num_logdiff")
d = d.join(x, on=["s1", "m"], how="left")
# ---- injected/missing name-token features ----
GEN = {"center", "centre", "services", "service", "group", "co", "company", "enterprises", "enterprise", "corp", "corporation", "inc",
       "llc", "ltd", "limited", "private", "pvt", "the", "and", "sri", "shri", "mr", "smt", "dr", "ms"}
nn = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "n") for s in (1, 2, 3)]).collect()
y = d.select("s1", "m").join(nn, left_on="s1", right_on="entity_id").join(nn.rename({"n": "n2"}), left_on="m", right_on="entity_id")
def tokfeat(a, b):
    ta, tb = a.split(), b.split(); sa, sb = set(ta), set(tb)
    extra, miss = sb - sa, sa - sb
    return (len(extra), len(miss), int(bool(extra) and extra <= GEN), int(bool(miss) and miss <= GEN), int(len(tb) != len(sb)))
F = [tokfeat(a, b) for a, b in zip(y["n"].to_list(), y["n2"].to_list())]
y = y.select("s1", "m").with_columns(*[pl.Series(nm_, [f[i] for f in F], dtype=pl.Int8) for i, nm_ in enumerate(["tok_extra", "tok_missing", "extra_generic", "missing_generic", "tok_repeat"])])
d = d.join(y, on=["s1", "m"], how="left")
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
for key, nm in [("pc2", "sib_name"), ("pa", "sib_addr"), ("pns", "sib_ns"), ("pnum1", "sib_num"), ("pstreet", "sib_street")]:
    g = ["s1", key]
    d = d.with_columns(*top2("p1", g), pl.len().over(g).alias(f"{nm}_n"))
    mx, sd = f"p1_max_{'_'.join(g)}", f"p1_2nd_{'_'.join(g)}"
    d = d.with_columns(pl.when(pl.col(f"{nm}_n") <= 1).then(-1.0).when(pl.col("p1") == pl.col(mx)).then(pl.col(sd)).otherwise(pl.col(mx)).alias(f"{nm}_max"))
    if key in ("pa", "pnum1", "pstreet"):
        d = d.with_columns(pl.when(pl.col(key).is_null() | (pl.col(key) == "")).then(-1.0).otherwise(pl.col(f"{nm}_max")).alias(f"{nm}_max"))
# anchor copy = the S1's highest-p1 OTHER candidate; similarity of this candidate to it
from rapidfuzz import process, fuzz
d = d.with_columns(pl.col("p1").rank("ordinal", descending=True).over("s1").alias("_r"))
anc = d.filter(pl.col("_r") <= 2).select("s1", "_r", "m", "p1", "pc2", "pa", "pnum1").pivot(on="_r", index="s1", values=["m", "p1", "pc2", "pa", "pnum1"])
d = d.join(anc, on="s1", how="left")
isa = (d["m"] == d["m_1"]).to_numpy()
def pick(c): return np.where(isa, d[f"{c}_2"].fill_null("").to_numpy(), d[f"{c}_1"].fill_null("").to_numpy())
anc_n, anc_a, anc_num = pick("pc2"), pick("pa"), pick("pnum1")
anc_p = np.where(isa, d["p1_2"].fill_null(0).to_numpy(), d["p1_1"].fill_null(0).to_numpy())
d = d.with_columns(pl.Series("anc_p", anc_p),
    pl.Series("anc_name_ratio", process.cpdist(d["pc2"].to_list(), anc_n.tolist(), scorer=fuzz.ratio, workers=4), dtype=pl.Float32),
    pl.Series("anc_addr_tsr", process.cpdist(d["pa"].to_list(), anc_a.tolist(), scorer=fuzz.token_set_ratio, workers=4), dtype=pl.Float32),
    pl.Series("anc_num_eq", (d["pnum1"].fill_null("").to_numpy() == anc_num) & (anc_num != ""), dtype=pl.Int8))
S2 = ["p1", "p1_max_s1", "s1_best_other", "p1_rank_s1", "s1_n_p50", "s1_sum_p1", "m_best_other_s1", "m_ncand_dev",
      "sib_name_max", "sib_name_n", "sib_addr_max", "sib_addr_n", "sib_ns_max", "sib_ns_n",
      "sib_num_max", "sib_num_n", "sib_street_max", "sib_street_n", "anc_p", "anc_name_ratio", "anc_addr_tsr", "anc_num_eq"]
S2A = S2[:14]
tr, va, ho = [d.filter(pl.col("split") == s) for s in ("train", "valid", "hold")]
gv, gh = gt.filter(pl.col("split") == "valid"), gt.filter(pl.col("split") == "hold")
R = {}
NK = ["num_kind", "num_logdiff"]; TK = ["tok_extra", "tok_missing", "extra_generic", "missing_generic", "tok_repeat"]
for name, cols in [("stage2b+numkind", BASE + S2 + NK), ("stage2b+numkind+tok", BASE + S2 + NK + TK)]:
    m = fit(tr, va, cols); pv, ph = m.predict(X(va, cols)), m.predict(X(ho, cols))
    for pol in ("none",):
        fv, t = best_thr(va, pv, gv, pol); fh, det = macro_f05(ho, ph, t, gh, pol)
        R[f"{name}_{pol}"] = dict(valid=fv, thr=t, hold=fh, **det)
        print(f"{name:7s} {pol:8s} valid={fv:.4f} thr={t:.2f} HOLD={fh:.4f} {det}", flush=True)
    if False:
        imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda z: -z[1]); R["imp"] = [(a, float(b)) for a, b in imp[:25]]
        print("top gain:", [(a, round(b / imp[0][1], 3)) for a, b in imp[:15]])
        ho2 = ho.with_columns(pl.Series("p", ph))
        seg = pl.when(pl.col("a2_empty") == 1).then(pl.lit("addr_empty")).when(pl.col("brand2") == 1).then(pl.lit("brand")).when(pl.col("dom2") == 1).then(pl.lit("domain")) \
                .when(pl.col("n_tsr") < 50).then(pl.lit("weakname")).when(pl.col("a_tsr") < 70).then(pl.lit("weakaddr")).otherwise(pl.lit("normal"))
        print(ho2.with_columns(seg.alias("seg"), (pl.col("p") >= R["stage2b_none"]["thr"]).alias("pred")).group_by("seg").agg(
            pl.col("pred").filter(pl.col("y")).mean().alias("recall"), (pl.col("pred") & ~pl.col("y")).sum().alias("fp"), pl.col("y").sum().alias("pos")).sort("pos", descending=True))
json.dump(R, open("analysis/out/74_tokfeat.json", "w"), indent=1, default=float)

# ho.with_columns(pl.Series("p2", ph)).select("s1","m","y","p2","a2_empty","brand2","dom2","n_tsr","a_tsr","name_freq_s1","rrank","country","src").write_parquet("work/dev_hold_stage2b.parquet")
print(memguard.report(), f"{time.time()-t0:.0f}s")
