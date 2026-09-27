"""In the UNCERTAIN band (0.1 <= p2x2 < 0.95), does a same-source fingerprint separate true from false?
 fp_addr : raw address string == raw address of an ACCEPTED same-source candidate of the same S1
 fp_tok  : raw first digit-token (as written, e.g. '04613/13', 'B3/7-1-28/3/1', 'No-a-244') == that of an accepted same-source candidate
 fp_tok_s1: raw first digit-token == S1's raw first digit-token"""
import sys, json, re; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
TOK = r"([^\s,]*\d[^\s,]*)"
parts = []
for f in [0, 8]:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    acc = decide(prep(d, dec["tau"], "p2x2"), dec).with_columns(pl.lit(True).alias("acc"))
    x = d.filter(pl.col("p1") >= dec["tau"]).join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    ids = x.select(pl.col("m").alias("entity_id")).unique()
    raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect() \
            .select(pl.col("entity_id").alias("m"), pl.col("business_address").fill_null("").str.to_lowercase().alias("ra"))
    s1r = pl.scan_parquet("work/train_s1.parquet").join(x.select(pl.col("s1").alias("entity_id")).unique().lazy(), on="entity_id", how="semi").collect() \
            .select(pl.col("entity_id").alias("s1"), pl.col("business_address").fill_null("").str.to_lowercase().str.extract(TOK, 1).fill_null("").alias("tok1"))
    x = x.join(raw, on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src"), pl.col("ra").str.extract(TOK, 1).fill_null("").alias("tok")).join(s1r, on="s1")
    A = x.filter(pl.col("acc")).select("s1", "src", pl.col("m").alias("m2"), pl.col("ra").alias("ra2"), pl.col("tok").alias("tok2"))
    band = x.filter((pl.col("p2x2") >= 0.1) & (pl.col("p2x2") < 0.95))
    ja = band.join(A, on=["s1", "src"], how="left").filter(pl.col("m2").is_null() | (pl.col("m2") != pl.col("m")))
    fa = ja.group_by("s1", "m").agg(((pl.col("ra2") == pl.col("ra")) & (pl.col("ra") != "")).any().fill_null(False).alias("fp_addr"),
                                    ((pl.col("tok2") == pl.col("tok")) & (pl.col("tok") != "")).any().fill_null(False).alias("fp_tok"))
    band = band.join(fa, on=["s1", "m"], how="left").with_columns(pl.col("fp_addr").fill_null(False), pl.col("fp_tok").fill_null(False),
                                                                  ((pl.col("tok") == pl.col("tok1")) & (pl.col("tok") != "")).alias("fp_tok_s1"))
    parts.append(band)
b = pl.concat(parts)
pl.Config.set_tbl_rows(40)
print("uncertain band candidates:", b.height, "true rate", round(b["y"].mean(), 4), "accepted share", round(b["acc"].mean(), 4))
print(b.group_by("fp_addr", "fp_tok", "fp_tok_s1").agg(pl.len(), pl.col("y").mean().round(3).alias("true_rate"), pl.col("acc").mean().round(3).alias("acc_rate"),
      (pl.col("y") & ~pl.col("acc")).sum().alias("FN"), (~pl.col("y") & pl.col("acc")).sum().alias("FP")).sort("len", descending=True))
