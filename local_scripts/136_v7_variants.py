"""Holdout F of V7 (R1-R4) and V7b (V7 + reject collectively-promoted records with changed house numbers),
and build output_v7b. Promoted = accepted by p2x2 policy but not by pairwise p1 policy (analysis/out/134-135)."""
import sys, json, os, shutil; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/130_rules_v7.py").read().split("gt = duckdb")[0])
D279 = set(json.load(open("analysis/aws_v5/v6_descriptor_rule.json"))["train_derived"])
RULES["R3_fr_sib_extra"] = lambda x: (x["country"] == "France") & x["E"].list.eval(pl.element().is_in(list(FR_SIB))).list.any()
RULES["R4_desc_usin"] = lambda x: (x["country"] != "France") & x["E"].list.eval(pl.element().is_in(list(D192 if HOLD else D279))).list.any()
D192 = set(pl.read_parquet("work/descriptor_words_train.parquet")["t"].to_list())
cp1 = dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)
NUMCHG = ["+1..9", "|d|10..99", "+100", "-100"]
def build(d, sp):
    a2 = decide(prep(d, dec["tau"], "p2x2"), dec)
    a1 = decide(prep(d, 0.05, "p1"), cp1).with_columns(pl.lit(True).alias("acc1"))
    x = a2.join(a1, on=["s1", "m"], how="left").with_columns(pl.col("acc1").fill_null(False))
    return x
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
HOLD = True
for f in [0, 8, 9]:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
    x = build(d.drop("country"), "train").join(d.select("s1", "m", "y", "country"), on=["s1", "m"]); del d
    x = enrich(x, "train"); s1f = gt.filter(pl.col("fold") == f)
    r7 = __import__('functools').reduce(lambda a, b: a | b, [RULES[k](x) for k in RULES])
    prom = (~x["acc1"]) & x["off"].is_in(NUMCHG)
    def sc(rm):
        g = s1f.join(x.filter(~rm).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
        return f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean()
    print(f"fold {f}: V5 {sc(pl.Series([False]*x.height)):.5f}  V7 {sc(r7):.5f}  V7b {sc(r7 | prom):.5f}  (V7b removes TP {int(((r7|prom) & x['y']).sum())} FP {int(((r7|prom) & ~x['y']).sum())})", flush=True)
HOLD = False
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1t = dict(s1c.group_by("country").len().rows())
keep = []
summ = {}
for c in ["US", "India", "France"]:
    d = pl.scan_parquet("work/v5_p2/test/*.parquet").filter(pl.col("country") == c).select("s1", "m", "p1", "pc2", "pa", "p2x2").collect()
    x = enrich(build(d, "test").with_columns(pl.lit(c).alias("country")), "test"); del d
    r7 = __import__('functools').reduce(lambda a, b: a | b, [RULES[k](x) for k in RULES]); prom = (~x["acc1"]) & x["off"].is_in(NUMCHG)
    n = nS1t[c] / 1000
    summ[c] = dict(v5=x.height, v7_removed=int(r7.sum()), v7b_removed=int((r7 | prom).sum()))
    print(f"TEST {c}: V5 pairs {x.height} ({x.height/nS1t[c]:.3f}/S1)  V7 removes {r7.sum()/n:.1f}/1k  V7b removes {(r7|prom).sum()/n:.1f}/1k", flush=True)
    keep.append(x.filter(~(r7 | prom)).select("s1", "m"))
keep = pl.concat(keep)
os.makedirs("output_v7b", exist_ok=True)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = keep.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).write_csv("output_v7b/matching_results.tsv", separator="\t", quote_style="never")
shutil.copy("output_v5/candidate_pairs.tsv", "output_v7b/candidate_pairs.tsv")
json.dump({"base": "output_v5 decision recomputed", "summary": summ, "n_pairs": keep.height}, open("output_v7b/v7b_info.json", "w"), indent=1)
print("V7b pairs:", keep.height, round(keep.height / allS1.height, 4))
