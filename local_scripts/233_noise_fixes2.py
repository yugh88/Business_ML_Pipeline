"""More targeted noise fixes, validated on the labelled holdout (V11 final logic baseline):
 R3b OCR-only names  : raw names differ ONLY by OCR digits (after 0->o 1->l 5->s 8->b 3->e 4->a they are equal), same address
 R4  empty + unique  : pool address empty, identical core name, S1 core name UNIQUE among S1 (cannot belong to another S1)
 R5  hidden twins    : ACCEPTED pairs whose pool number was lost in parsing and the raw number = S1 number + 1..99 -> REMOVE"""
import sys, re, json
exec(open("scripts/232_noise_fixes.py").read().split("gt = duckdb.connect()")[0].replace('if __name__', 'if False'))
def feats(split, x):
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", pl.col("nums").fill_null("")) for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    raw = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")) for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    uniq = pl.scan_parquet(f"work/{split}_s1_norm.parquet").group_by("ncore").len().collect()
    U = set(uniq.filter(pl.col("len") == 1)["ncore"].to_list())
    N = {r[0]: r[1:] for r in nm.iter_rows()}; R = {r[0]: r[1:] for r in raw.iter_rows()}
    o = []
    for a, b in zip(x["s1"].to_list(), x["m"].to_list()):
        if a not in N or b not in N: o.append((False, False, False)); continue
        (c1, a1, u1), (c2, a2, u2) = N[a], N[b]; (rn1, ra1), (rn2, ra2) = R.get(a, ("", "")), R.get(b, ("", ""))
        ocr = (rn1 != rn2) and re.sub(r"[^a-z]", "", rn1.lower().translate(OCRMAP)) == re.sub(r"[^a-z]", "", rn2.lower().translate(OCRMAP)) and bool(re.search(r"[0158]", rn2))
        same_addr = addr_rel(a1, a2, u1, u2) in ("A0_identical", "A1_same_num") and bool(u1) and bool(u2)
        r4 = (ra2.strip() == "") and c1 == c2 and c1 in U
        m = RAWNUM.search(ra2); f1 = (u1.split() or [""])[0]
        r5 = c1 == c2 and not u2 and bool(m) and f1.isdigit() and 1 <= int(m.group(1)[:9]) - int(f1[:9]) <= 99
        o.append((ocr and same_addr, r4, r5))
    return x.with_columns(*[pl.Series(k, [v[i] for v in o]) for i, k in enumerate(("R3b", "R4", "R5"))])
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti").filter(pl.col("p1") >= TAU).join(t1, on="s1")
acc_h = pl.concat([decode(h.filter(pl.col("country") == c), ALPHA[c]) for c in ("US", "India")]).with_columns(pl.lit(True).alias("acc"))
hx = feats("train", h.join(acc_h, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False)))
base = gt.join(hx.filter("acc").group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
taken = hx.filter("acc").select("m").unique()
print("HOLDOUT validation:")
for rule, add in (("R3b", True), ("R4", True), ("R5", False)):
    if add:
        ad = hx.filter(~pl.col("acc") & pl.col(rule)).join(taken, on="m", how="anti").sort("p3", descending=True).unique(subset="m", keep="first")
        z = base.join(ad.group_by("s1").agg(pl.col("y").sum().alias("atp"), (~pl.col("y")).sum().alias("afp")), on="s1", how="left").fill_null(0)
        d = F(z["tp"] + z["atp"], z["fp"] + z["afp"], z["n"]) - F(z["tp"], z["fp"], z["n"]); n_, prec = ad.height, ad["y"].mean() or 0
    else:
        rm = hx.filter(pl.col("acc") & pl.col(rule))
        z = base.join(rm.group_by("s1").agg(pl.col("y").sum().alias("rtp"), (~pl.col("y")).sum().alias("rfp")), on="s1", how="left").fill_null(0)
        d = F(z["tp"] - z["rtp"], z["fp"] - z["rfp"], z["n"]) - F(z["tp"], z["fp"], z["n"]); n_, prec = rm.height, 1 - (rm["y"].mean() or 0)
    res = {c: float(d[(z["country"] == c).to_numpy()].mean()) for c in ("US", "India")}
    print(f"  {rule:4s} ({'add' if add else 'remove'}): {n_ / gt.height * 1000:5.2f}/1k S1, correct-share {prec:.3f} -> holdout F US {res['US']:+.5f} India {res['India']:+.5f}", flush=True)
nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
for c in ("US", "India", "France"):
    x = pl.read_parquet(f"{S3D}/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= TAU)
    a = decode(x, ALPHA[c]).with_columns(pl.lit(True).alias("acc"))
    tx = feats("test", x.join(a, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False)))
    tx.filter(pl.col("R3b") | pl.col("R4") | pl.col("R5")).select("s1", "m", "p3", "acc", "R3b", "R4", "R5").write_parquet(f"work/noisefix2_{c}.parquet")
    tk = tx.filter("acc").select("m").unique()
    print(f"TEST {c:6s}: R3b add {tx.filter(~pl.col('acc') & pl.col('R3b')).join(tk, on='m', how='anti').height / nt[c] * 1000:.2f}/1k, R4 add {tx.filter(~pl.col('acc') & pl.col('R4')).join(tk, on='m', how='anti').height / nt[c] * 1000:.2f}/1k, R5 remove {tx.filter(pl.col('acc') & pl.col('R5')).height / nt[c] * 1000:.2f}/1k", flush=True)
