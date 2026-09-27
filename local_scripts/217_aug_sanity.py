"""Sanity check of the V10 augmentation (local, before any AWS spend): normalize the synthetic records, classify each against its
origin S1 with the same relation taxonomy used on test (213), and compare the category mix with the test excess profile.
Writes work/aug_v10_norm.parquet (needed later by stage-3)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from normlib import normalize_record
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])     # annotate(), offset(), klass()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True); pl.Config.set_fmt_str_lengths(70)
raw = pl.concat([pl.read_parquet(f"work/aug_v10_{s}.parquet") for s in ("s2", "s3")])
rows = []
for e, n, a, c in raw.iter_rows():
    r = normalize_record(n, a, c)
    rows.append((e, r["ncore"], r["a"], r["nums"]))
nm = pl.DataFrame(rows, schema=["entity_id", "ncore", "a", "nums"], orient="row"); nm.write_parquet("work/aug_v10_norm.parquet")
mt = pl.read_parquet("work/aug_v10_meta.parquet")
x = annotate(mt.select("s1", pl.col("entity_id").alias("m"), "kind", "num", "name_op"), "train", "work/aug_v10_norm.parquet")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
x = x.join(t1, on="s1")
ns = dict(t1.group_by("country").len().rows())
print("V10 synthetic records per 1k train S1, by kind -> relation class / offset (top 3 each):")
for (c, k), g in sorted(x.group_by("country", "kind"), key=lambda z: z[0]):
    top = g.group_by("cls", "off").len().sort("len", descending=True).head(3)
    print(f"  {c:6s} {k:8s} {1000 * g.height / ns[c]:6.1f}/1k  " + " | ".join(f"{r[0]}/{r[1]} {100 * r[2] / g.height:.0f}%" for r in top.rows()))
ex = pl.read_parquet("work/excess_profile.parquet").filter(pl.col("excess_cands") >= 2)
gen = x.group_by("country", "cls", "off").len().with_columns((pl.col("len") / pl.col("country").replace_strict(ns) * 1000).round(1).alias("gen_per1k")).drop("len")
print("\nTest excess categories (candidates/1k beyond train-as-trained) vs V10 generated records/1k (only ~25-30% reach the cascade set):")
print(ex.select("country", "cls", "off", pl.col("excess_cands").round(1)).join(gen, on=["country", "cls", "off"], how="left").fill_null(0).sort("country", "excess_cands", descending=[False, True]))
s1raw = pl.read_parquet("work/train_s1.parquet").rename({"entity_id": "s1"})
smp = mt.join(raw, on="entity_id").join(s1raw.select("s1", pl.col("business_name").alias("s1n"), pl.col("business_address").alias("s1a")), on="s1")
for k in ("fwr", "swap", "light", "numx", "namenum", "empty"):
    for r in smp.filter(pl.col("kind") == k).sample(n=2, seed=5).iter_rows(named=True):
        print(f"[{k}/{r['num']}/{r['name_op']}] S1 {r['s1n']!r} | {r['s1a']!r}\n      syn {r['business_name']!r} | {r['business_address']!r}")
