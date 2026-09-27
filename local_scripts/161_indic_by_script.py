"""Per-script (Telugu etc.) copy rate, V5 holdout recall/precision, and test acceptance (V7)."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
BLOCKS = [("devanagari", r"[\x{0900}-\x{097F}]"), ("bengali", r"[\x{0980}-\x{09FF}]"), ("gurmukhi", r"[\x{0A00}-\x{0A7F}]"), ("gujarati", r"[\x{0A80}-\x{0AFF}]"),
          ("tamil", r"[\x{0B80}-\x{0BFF}]"), ("telugu", r"[\x{0C00}-\x{0C7F}]"), ("kannada", r"[\x{0C80}-\x{0CFF}]"), ("malayalam", r"[\x{0D00}-\x{0D7F}]")]
def script_of(sp, ids):
    e = pl.lit("latin_name")
    for nm, rx in reversed(BLOCKS):
        e = pl.when(pl.col("business_name").str.contains(rx)).then(pl.lit(nm)).otherwise(e)
    q = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi")
    return q.select("entity_id", e.alias("script")).collect()
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("country") == "India")).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
pr = decide(prep(d, dec["tau"], "p2x2"), dec).with_columns(pl.lit(True).alias("pred"))
truth = pl.read_parquet("work/train_pairs.parquet").join(d.select("s1").unique(), on="s1", how="semi").with_columns(pl.lit(True).alias("y_"))
allp = truth.join(pr, on=["s1", "m"], how="full", coalesce=True).with_columns(pl.col("y_").fill_null(False), pl.col("pred").fill_null(False))
allp = allp.join(script_of("train", allp.select(pl.col("m").alias("entity_id")).unique()).rename({"entity_id": "m"}), on="m", how="left")
tr = allp.group_by("script").agg(pl.col("y_").sum().alias("true_pairs"), (pl.col("y_") & pl.col("pred")).sum().alias("tp"), (~pl.col("y_") & pl.col("pred")).sum().alias("fp")) \
         .with_columns((pl.col("tp") / pl.col("true_pairs")).round(4).alias("recall"), (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).round(4).alias("precision"))
p7 = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v7/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
tpool = pl.concat([pl.scan_parquet(f"work/test_s{s}.parquet").filter(pl.col("country") == "India").select("entity_id") for s in (2, 3)]).collect()
ts = script_of("test", tpool).join(p7.select(pl.col("m").alias("entity_id")).unique().with_columns(pl.lit(True).alias("pred")), on="entity_id", how="left").with_columns(pl.col("pred").fill_null(False))
te = ts.group_by("script").agg(pl.len().alias("test_records"), pl.col("pred").mean().round(4).alias("test_pred_rate"))
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(200)
print(tr.join(te, on="script", how="full", coalesce=True).sort("test_records", descending=True))
# Telugu examples: S1 vs matched Telugu copy (train)
tel = allp.filter((pl.col("script") == "telugu") & pl.col("y_") & pl.col("pred")).head(6)
ids = pl.concat([tel.select(pl.col("s1").alias("entity_id")), tel.select(pl.col("m").alias("entity_id"))])
raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
nrm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "n") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
R = {r[0]: r[1] for r in raw.select("entity_id", "business_name").iter_rows()}; N = dict(nrm.iter_rows())
print("\nTelugu examples (train, correctly matched): S1 name | raw copy name -> normalized")
for s1, m in tel.select("s1", "m").iter_rows(): print(f"  {R[s1]} | {R[m]} -> {N[m]}")
