"""V5 label-free realism checks: (1) how 'tempting' simulated orphans are vs test unmatched records;
(2) per-country test ambiguity/contest rates for V5 vs V3."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local, polars as pl
pairs = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"])
drop = pl.read_parquet("work/v5_dropped_s1.parquet")
orph = pairs.join(drop, on="s1", how="semi").join(drop.select("s1", "why"), on="s1").select("m", "why")
def bins(d):
    return d.with_columns(pl.when(pl.col("best") < 0.05).then(pl.lit("a<0.05")).when(pl.col("best") < 0.7).then(pl.lit("b 0.05-0.7"))
                          .otherwise(pl.lit("c>=0.7")).alias("bin"))
tr = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("p1") >= 1e-4).select("m", "y", "p2x2").collect() \
       .group_by("m").agg(pl.col("p2x2").max().alias("best"), pl.col("y").any().alias("matched"))
tr = tr.join(orph, on="m", how="left").with_columns(pl.when(pl.col("matched")).then(pl.lit("true_copy"))
        .when(pl.col("why") == "hard").then(pl.lit("sim_orphan_HARD")).when(pl.col("why") == "random").then(pl.lit("sim_orphan_random"))
        .otherwise(pl.lit("orig_distractor")).alias("kind"))
acc = pl.read_csv("output_v5_auto/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8}) \
        .select(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
te = pl.scan_parquet("work/v5_p2/test/*.parquet").filter(pl.col("p1") >= 1e-4).select("m", "p2x2").collect().group_by("m").agg(pl.col("p2x2").max().alias("best"))
te = te.join(acc.with_columns(pl.lit(True).alias("pred")), on="m", how="left").filter(pl.col("pred").is_null()).with_columns(pl.lit("test_unmatched").alias("kind"))
allk = pl.concat([bins(tr).select("kind", "bin"), bins(te).select("kind", "bin")])
print("(1) best candidate probability (V5 retrained XGB) by record kind — share per bin:")
print(allk.group_by("kind", "bin").len().with_columns((pl.col("len") / pl.col("len").sum().over("kind")).round(4).alias("share"))
      .pivot(on="bin", index="kind", values="share").sort("kind"))
print("   reference V3: test_unmatched band 0.05-0.7 = 0.228 ; V3 random sim orphans = 0.059")
te1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
def stats(glob, col, label):
    d = pl.scan_parquet(glob).filter(pl.col("p1") >= 0.02).select("s1", "m", pl.col(col).alias("p")).collect()
    d = d.with_columns(pl.col("p").max().over("m").alias("mmax"))
    per = d.group_by("s1").agg(((pl.col("p") >= 0.2) & (pl.col("p") < 0.8)).sum().alias("namb"), (pl.col("p") >= 0.7).sum().alias("n70"),
                               ((pl.col("p") >= 0.5) & (pl.col("p") < pl.col("mmax"))).sum().alias("ncont"))
    g = te1.join(per, on="s1", how="left").fill_null(0)
    print(label); print(g.group_by("country").agg((pl.col("namb") > 0).mean().alias("S1_ambig"), pl.col("n70").mean().alias("p>=.7/S1"),
                                                  (pl.col("ncont") > 0).mean().alias("S1_contested")).sort("country"))
print("(2) TEST per-country rates")
stats("work/v3_p2/test/*.parquet", "p2x", "V3 (p2x):")
stats("work/v5_p2/test/*.parquet", "p2x2", "V5 (p2x2):")
