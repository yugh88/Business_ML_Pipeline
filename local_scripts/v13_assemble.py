"""V13 = V12 - French identical-name same-address accepted pairs with p3 < 0.8 in S1 with set size >= 3 and >= 1 strong (p3 >= 0.9)
copy (label-free address-fingerprint evidence, scripts 236/237). Exact code used for the submission."""
import sys, os, shutil, duckdb
exec(open("scripts/236_fr_threshold.py").read().split("rows = []")[0])
x = x.with_columns((pl.col("p3") >= 0.9).cast(pl.Int32).sum().over("s1").alias("n_strong"))
rm = x.filter((pl.col("cls") == "identical@same") & (pl.col("p3") < 0.8) & (pl.col("k") >= 3) & (pl.col("n_strong") >= 1)).select("s1", "m")
q2 = lambda p: duckdb.connect().execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
V12 = q2("output_v12/matching_results.tsv"); pred = V12.join(rm, on=["s1", "m"], how="anti")
os.makedirs("output_v13", exist_ok=True)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = pred.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id").write_csv("output_v13/matching_results.tsv", separator="\t", quote_style="never")
shutil.copy("output_v12/candidate_pairs.tsv", "output_v13/candidate_pairs.tsv")
print(f"V13: removed {rm.height} French pairs on {rm['s1'].n_unique()} S1; pairs {pred.height}")
