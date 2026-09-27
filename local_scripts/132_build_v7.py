"""V7 = V5 predictions minus generator-signature false merges (analysis/out/115-131):
 R1 twin: house number = S1's + 1..9 while the SAME source also has an accepted record with the S1's own number
 R2 twin: +1..9 group spanning both sources while the S1's own number is supported by an accepted record
 R3 France sibling descriptors (address behaviour of siblings): holding/participations/international(e)/distribution/snc
 R4 US/India train descriptor words (match rate <5% as extra word, n>=100; V6 set) — France excluded
   (V6's French analog list is dropped: groupe/france/developpement/fils/associes behave like COPY noise in France)."""
import sys, json, os, shutil; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
exec(open("scripts/130_rules_v7.py").read().split("gt = duckdb")[0])
D279 = set(json.load(open("analysis/aws_v5/v6_descriptor_rule.json"))["train_derived"])
RULES["R3_fr_sib_extra"] = lambda x: (x["country"] == "France") & x["E"].list.eval(pl.element().is_in(list(FR_SIB))).list.any()
RULES["R4_desc_usin"] = lambda x: (x["country"] != "France") & x["E"].list.eval(pl.element().is_in(list(D279))).list.any()
con = duckdb.connect()
p5 = con.execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v5/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
te = enrich(p5.join(s1c, on="s1"), "test")
flags = te.with_columns(*[RULES[k](te).alias(k) for k in RULES]).select("s1", "m", "country", *RULES.keys())
flags = flags.with_columns(pl.any_horizontal(*RULES.keys()).alias("drop"))
nS1 = dict(s1c.group_by("country").len().rows())
summ = {c: {k: int(flags.filter(pl.col("country") == c)[k].sum()) for k in list(RULES) + ["drop"]} for c in nS1}
print(json.dumps(summ, indent=1))
keep = flags.filter(~pl.col("drop")).select("s1", "m")
assert keep.height + int(flags["drop"].sum()) == p5.height
os.makedirs("output_v7", exist_ok=True)
agg = keep.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids"))
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
out = allS1.join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
out.write_csv("output_v7/matching_results.tsv", separator="\t", quote_style="never")
shutil.copy("output_v5/candidate_pairs.tsv", "output_v7/candidate_pairs.tsv")
json.dump({"base": "output_v5", "removed": summ, "n_pairs_v5": p5.height, "n_pairs_v7": keep.height,
           "rules": __doc__}, open("output_v7/v7_info.json", "w"), indent=1)
print("V5 pairs", p5.height, "-> V7 pairs", keep.height, " per S1:", round(keep.height / allS1.height, 4))
