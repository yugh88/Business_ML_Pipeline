"""V7 rules (R1 twin same-source, R2 twin 2-source, R3 France siblings, R4 US/India descriptors) on top of ANY scored run:
holdout F0.5 (augmented + clean views) with/without each rule, and test removals per 1k S1. Writes <out>_rules/ variant.
usage: 158_rules_on.py <p2_dir> <decision.json> <matching_results.tsv> <out_dir>"""
import sys, json, os, shutil, functools; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/130_rules_v7.py").read().split("gt = duckdb.connect()")[0])
p2dir, decf, mres, outdir = sys.argv[1:5]
dec = json.load(open(decf)); V = dec["variant"]
D279 = set(json.load(open("analysis/aws_v5/v6_descriptor_rule.json"))["train_derived"])
RULES["R3_fr_sib_extra"] = lambda x: (x["country"] == "France") & x["E"].list.eval(pl.element().is_in(list(FR_SIB))).list.any()
RULES["R4_desc_usin"] = lambda x: (x["country"] != "France") & x["E"].list.eval(pl.element().is_in(list(D279))).list.any()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
COMBOS = {"none": [], "R1R2": ["R1_twin_same_src", "R2_twin_2src"], "R4": ["R4_desc_usin"], "R1R2R4": ["R1_twin_same_src", "R2_twin_2src", "R4_desc_usin"]}
for f in [0, 8, 9]:
    d = pl.scan_parquet(f"{p2dir}/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", V).collect()
    s1f = gt.filter(pl.col("fold") == f)
    line = f"fold {f}:"
    for view, dd in (("aug", d), ("clean", d.join(SYN, on="m", how="anti"))):
        pr = decide(prep(dd, dec["tau"], V), dec).select("s1", "m").join(dd.select("s1", "m", "y"), on=["s1", "m"]).join(t1, on="s1")
        x = enrich(pr, "train")
        for k, names in COMBOS.items():
            m = functools.reduce(lambda a, b: a | b, [RULES[n](x) for n in names]) if names else pl.Series([False] * x.height)
            g = s1f.join(x.filter(~m).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
            line += f" {view}/{k} {f05(g['tp'].to_numpy().astype(float), g['fp'].to_numpy().astype(float), g['n'].to_numpy().astype(float)).mean():.5f}"
    print(line, flush=True)
p = duckdb.connect().execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{mres}', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1 = dict(s1c.group_by("country").len().rows())
te = enrich(p.join(s1c, on="s1"), "test")
flags = te.with_columns(*[RULES[k](te).alias(k) for k in RULES]).select("s1", "m", "country", *RULES.keys())
for c in ["US", "India", "France"]:
    x = flags.filter(pl.col("country") == c); n = nS1[c] / 1000
    print(f"TEST {c}: " + " ".join(f"{k} -{x[k].sum()/n:.1f}/1k" for k in RULES))
flags.write_parquet(os.path.join(os.path.dirname(outdir.rstrip("/")) or ".", "work_rule_flags.parquet") if False else "work/rule_flags_last.parquet")
print("flags saved to work/rule_flags_last.parquet")
