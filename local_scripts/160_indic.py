"""Indian-script records (Devanagari etc.): share, true-copy rate, V5 recall / precision (train holdout) and test acceptance."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
IND = r"[\x{0900}-\x{0DFF}]"
DEV = r"[\x{0900}-\x{097F}]"
def script_tab(sp, ids):
    q = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi")
    return q.select("entity_id", pl.when(pl.col("business_name").str.contains(DEV)).then(pl.lit("name_devanagari"))
                    .when(pl.col("business_name").str.contains(IND)).then(pl.lit("name_other_indic"))
                    .when(pl.col("business_address").fill_null("").str.contains(IND)).then(pl.lit("addr_only_indic")).otherwise(pl.lit("latin")).alias("script")).collect()
# ---- train: all India pool records, true-copy status
pool = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet").filter(pl.col("country") == "India").select("entity_id") for s in (2, 3)]).collect()
st = script_tab("train", pool)
tp = pl.read_parquet("work/train_pairs.parquet", columns=["m"]).unique().with_columns(pl.lit(True).alias("is_copy"))
st = st.join(tp.rename({"m": "entity_id"}), on="entity_id", how="left").with_columns(pl.col("is_copy").fill_null(False))
print("TRAIN India pool records by script: n, share, true-copy (matched to some S1) rate")
print(st.group_by("script").agg(pl.len(), pl.col("is_copy").mean().round(4).alias("copy_rate")).with_columns((pl.col("len") / pl.col("len").sum()).round(4).alias("share")).sort("len", descending=True).rows())
# ---- V5 holdout recall / precision by script (India)
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("country") == "India")).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
pr = decide(prep(d, dec["tau"], "p2x2"), dec).with_columns(pl.lit(True).alias("pred"))
hold_s1 = d.select("s1").unique()
truth = pl.read_parquet("work/train_pairs.parquet").join(hold_s1, on="s1", how="semi").with_columns(pl.lit(True).alias("y_"))
allp = truth.join(pr, on=["s1", "m"], how="full", coalesce=True).with_columns(pl.col("y_").fill_null(False), pl.col("pred").fill_null(False))
allp = allp.join(script_tab("train", allp.select(pl.col("m").alias("entity_id")).unique()).rename({"entity_id": "m"}), on="m", how="left")
print("\nV5 holdout (India): recall on true pairs / precision of predictions, by script of the pool record")
print(allp.group_by("script").agg(pl.col("y_").sum().alias("true_pairs"), (pl.col("y_") & pl.col("pred")).sum().alias("tp"), (~pl.col("y_") & pl.col("pred")).sum().alias("fp"))
      .with_columns((pl.col("tp") / pl.col("true_pairs")).round(4).alias("recall"), (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).round(4).alias("precision")).sort("true_pairs", descending=True).rows())
# where are the missed Indic copies lost? candidates / p1 / decision
fn = allp.filter(pl.col("y_") & ~pl.col("pred")).join(d.select("s1", "m", "p1", "p2x2"), on=["s1", "m"], how="left")
print("\nmissed true pairs by script and reason:")
print(fn.with_columns(pl.when(pl.col("p1").is_null()).then(pl.lit("not_cand")).when(pl.col("p1") < dec["tau"]).then(pl.lit("p1<tau")).otherwise(pl.lit("scored_rejected")).alias("why"))
      .group_by("script", "why").len().sort("script", "len", descending=[False, True]).rows())
# ---- test: prediction share by script vs pool share
p5 = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v7/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
tpool = pl.concat([pl.scan_parquet(f"work/test_s{s}.parquet").filter(pl.col("country") == "India").select("entity_id") for s in (2, 3)]).collect()
ts = script_tab("test", tpool).join(p5.select(pl.col("m").alias("entity_id")).unique().with_columns(pl.lit(True).alias("pred")), on="entity_id", how="left").with_columns(pl.col("pred").fill_null(False))
print("\nTEST India pool records by script: n, share, predicted-as-match rate (V7)  [compare train copy_rate]")
print(ts.group_by("script").agg(pl.len(), pl.col("pred").mean().round(4).alias("pred_rate")).with_columns((pl.col("len") / pl.col("len").sum()).round(4).alias("share")).sort("len", descending=True).rows())
