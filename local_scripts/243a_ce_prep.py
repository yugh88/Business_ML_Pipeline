"""CE v2 data prep (separate process so the MPS trainer stays small): pair texts "name ; address" for
 - training: ALL hard pairs (V8 stage-2 p2x2 in 0.02-0.98) + 200k easy pairs of TRAIN folds 1-6 (incl. V8 synthetic look-alikes)
 - scoring: uncertain band (p1 >= 0.05, 0.02 <= p3 <= 0.98) of V11s3 train folds 0/7/8/9 (real records) and test US/India/France
-> work/ce2_train_texts.parquet (a, b, y), work/ce2_score_texts.parquet (part, s1, m, a, b)."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl

d = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([1, 2, 3, 4, 5, 6]) & (pl.col("p1") >= 0.05)).select("s1", "m", "y", "p2x2").collect()
hard = d.filter(pl.col("p2x2").is_between(0.02, 0.98)); easy = d.filter(~pl.col("p2x2").is_between(0.02, 0.98)).sample(n=200000, seed=2)
tr = pl.concat([hard, easy]).select("s1", "m", "y"); del d, hard, easy


def texts(pairs, split, aug):
    ids = pl.concat([pairs.select(pl.col("s1").alias("entity_id")), pairs.select(pl.col("m").alias("entity_id"))]).unique()
    srcs = [pl.scan_parquet(f"work/{split}_s{s}.parquet") for s in (1, 2, 3)] + ([pl.scan_parquet(f"work/aug_v8_{s}.parquet") for s in ("s2", "s3")] if aug else [])
    raw = pl.concat([s.select("entity_id", (pl.col("business_name").fill_null("") + " ; " + pl.col("business_address").fill_null("")).alias("t")) for s in srcs]) \
            .join(ids.lazy(), on="entity_id", how="semi").collect()
    return pairs.join(raw.rename({"entity_id": "s1", "t": "a"}), on="s1", how="left").join(raw.rename({"entity_id": "m", "t": "b"}), on="m", how="left") \
                .with_columns(pl.col("a").fill_null(""), pl.col("b").fill_null(""))


t = texts(tr, "train", True)
assert t.height == tr.height
t.select("a", "b", pl.col("y").cast(pl.Float32)).write_parquet("work/ce2_train_texts.parquet"); print("train texts", t.height, "positive", round(t["y"].mean(), 3)); del t, tr
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
band = lambda x: x.filter((pl.col("p1") >= 0.05) & pl.col("p3").is_between(0.02, 0.98)).select("s1", "m")
parts = [texts(band(pl.concat([pl.read_parquet(f"output_v11s3/p3/train/fold{f}.parquet", columns=["s1", "m", "p1", "p3"]) for f in (0, 7, 8, 9)]).join(SYN, on="m", how="anti")), "train", False)
         .with_columns(pl.lit("train").alias("part"))]
for c in ("US", "India", "France"):
    parts.append(texts(band(pl.read_parquet(f"output_v11s3/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"])), "test", False).with_columns(pl.lit(c).alias("part")))
s = pl.concat(parts).select("part", "s1", "m", "a", "b")
s.write_parquet("work/ce2_score_texts.parquet"); print("score texts", dict(s.group_by("part").len().rows()))
