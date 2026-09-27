"""Write a submission from per-country decision files (206: work/mix_acc_{country}_{alpha_d}.parquet).
usage: 207_build_final.py <out_dir> <alpha_US> <alpha_India> <alpha_France>
matching_results.tsv: every test S1 (sorted), matched IDs sorted and comma-joined, empty when none (same format as V9).
candidate_pairs.tsv: V9's candidate set (unchanged: the decoder only re-decides within it)."""
import sys, os, shutil
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
out, a_us, a_in, a_fr = sys.argv[1], *sys.argv[2:5]
os.makedirs(out, exist_ok=True)
pred = pl.concat([pl.read_parquet(f"work/mix_acc_{c}_{float(a)}.parquet").select("s1", "m") for c, a in (("US", a_us), ("India", a_in), ("France", a_fr))])
cand = pl.read_csv("output_v9/candidate_pairs.tsv", separator="\t", quote_char=None).rename({"source1_entity_id": "s1"}) \
         .with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(",")).explode("candidate_entity_ids").rename({"candidate_entity_ids": "m"})
missing = pred.join(cand, on=["s1", "m"], how="anti").height
assert missing == 0, f"{missing} matches outside the candidate set"
assert pred.select(pl.len()).item() == pred.unique().height, "duplicate pairs"
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = pred.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id") \
     .write_csv(os.path.join(out, "matching_results.tsv"), separator="\t", quote_style="never")
shutil.copy("output_v9/candidate_pairs.tsv", os.path.join(out, "candidate_pairs.tsv"))
print(f"wrote {out}: {pred.height} pairs for {allS1.height} S1 ({1000 * pred.height / allS1.height:.1f}/1k); alphas US {a_us} India {a_in} France {a_fr}")
