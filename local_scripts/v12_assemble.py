"""V12 = V11 + validated noise fixes (exact code used for the submission): R2 house-number formats (232) and R3b OCR-digit-only
names (233) among V11-rejected tau-set pairs whose pool record is unassigned; best p3 per record."""
import polars as pl, duckdb, os, shutil
con = duckdb.connect()
q = lambda p: con.execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
V11 = q("output_v11/matching_results.tsv")
adds = []
for c in ("US", "India", "France"):
    r2 = pl.read_parquet(f"work/noisefix_{c}.parquet").filter("R2").select("s1", "m", "p3").with_columns(pl.lit("R2").alias("rule"))
    r3 = pl.read_parquet(f"work/noisefix2_{c}.parquet").filter(pl.col("R3b") & ~pl.col("acc")).select("s1", "m", "p3").with_columns(pl.lit("R3b").alias("rule"))
    a = pl.concat([r2, r3]).join(V11, on=["s1", "m"], how="anti").join(V11.select("m").unique(), on="m", how="anti")
    a = a.sort("p3", descending=True).unique(subset="m", keep="first")
    print(f"{c}: R2 +{a.filter(pl.col('rule')=='R2').height}, R3b +{a.filter(pl.col('rule')=='R3b').height}")
    adds.append(a.select("s1", "m"))
add = pl.concat(adds); pred = pl.concat([V11, add])
os.makedirs("output_v12", exist_ok=True)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = pred.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id").write_csv("output_v12/matching_results.tsv", separator="\t", quote_style="never")
shutil.copy("output_v11/candidate_pairs.tsv", "output_v12/candidate_pairs.tsv")
print("V12 pairs", pred.height, "(V11", V11.height, "+", add.height, ")")
