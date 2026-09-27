"""Copy-noise signature: the extra ~22% test pool records carry ~no copy-noise forms. Measure 'noise-form' rates by class:
TRAIN pool: true copies vs copies of other S1 vs distractors; TEST pool: accepted by V9 vs not accepted. And the fraction of
records whose raw text looks S1-canonical (Title Case name + S1-style address with full state name)."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, duckdb
DBA = r"(?i)\b(dba|d/b/a|fka|f/k/a|aka|a/k/a|t/a|doing business as|formerly|trading as)\b"
con = duckdb.connect()


def props(q):
    return q.with_columns(
        (pl.col("business_address").fill_null("") == "").alias("addr_empty"), pl.col("f_domain").alias("domain"),
        pl.col("business_name").str.contains(DBA).alias("dba"), (pl.col("nums") == "").alias("no_num"),
        pl.col("business_name").str.contains(r"[^\x00-\x7F]").alias("nonascii"),
        (pl.col("business_name") == pl.col("business_name").str.to_uppercase()).alias("upper"),
        pl.col("business_name").str.contains(r"  |[\[\]#@]|\(|-\s|\s-").alias("punct_noise"))


FLAGS = ["addr_empty", "domain", "dba", "no_num", "nonascii", "upper", "punct_noise"]
pairs = pl.read_parquet("work/train_pairs.parquet").select("s1", "m")
drop = pl.read_parquet("work/v5_dropped_s1.parquet").select("s1")
cls_tr = pairs.join(drop, on="s1", how="anti").select(pl.col("m").alias("entity_id")).with_columns(pl.lit("true_copy(kept S1)").alias("cls")) \
    .vstack(pairs.join(drop, on="s1", how="semi").select(pl.col("m").alias("entity_id")).with_columns(pl.lit("orphan_copy(dropped S1)").alias("cls")))
for sp in ("train", "test"):
    q = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet").join(pl.scan_parquet(f"work/{sp}_s{s}_norm.parquet").select("entity_id", "nums", "f_domain"), on="entity_id")
                   .with_columns(pl.lit(f"S{s}").alias("src")) for s in (2, 3)])
    x = props(q).select(["entity_id", "country", "src"] + FLAGS).collect()
    if sp == "train":
        x = x.join(cls_tr, on="entity_id", how="left").with_columns(pl.col("cls").fill_null("distractor"))
    else:
        acc = con.execute("select unnest(string_split(matched_entity_ids, ',')) entity_id from read_csv('output_v9/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') "
                          "where matched_entity_ids is not null and matched_entity_ids<>''").pl().unique()
        x = x.join(acc.with_columns(pl.lit("V9_accepted").alias("cls")), on="entity_id", how="left").with_columns(pl.col("cls").fill_null("V9_not_accepted"))
    t = x.group_by("country", "src", "cls").agg([pl.len().alias("n")] + [pl.col(f).mean().round(4) for f in FLAGS]).sort("country", "src", "cls")
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
    print(f"==== {sp}"); [print("  ", r) for r in t.rows()]
