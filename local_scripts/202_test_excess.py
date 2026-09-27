"""Test-side decision budget (label-free): per country x class x noise bucket, compare test candidate / accepted density per 1k S1
with the holdout's (true copies, accepted TP, accepted FP). Excess accepted over the holdout TP density = candidate false merges
(assumes test copy generation per S1 = train, supported by clean categories). Value = excess x typical FP cost in that segment."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
g = pl.read_parquet("work/error_budget_s1.parquet")
nh = dict(g.group_by("country").len().rows())
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "y", "acc", "cls", "bucket", "p3"]).join(t1, on="s1")
H = h.group_by("country", "cls", "bucket").agg(pl.len().alias("n"), pl.col("y").sum().alias("T"), (pl.col("acc") & pl.col("y")).sum().alias("TPacc"),
                                                (pl.col("acc") & ~pl.col("y")).sum().alias("FPacc"))
H = H.with_columns(*[(pl.col(c) / pl.col("country").replace_strict(nh) * 1000).alias(c + "_h") for c in ("n", "T", "TPacc", "FPacc")])
rows = []
s1t = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"])
nt = dict(s1t.group_by("country").len().rows())
for c in ("US", "India"):
    x = pl.read_parquet(f"work/copy_channel_test_{c}.parquet", columns=["s1", "acc", "cls", "bucket", "p3"])
    t = x.group_by("cls", "bucket").agg((pl.len() / nt[c] * 1000).alias("n_t"), (pl.col("acc").sum() / nt[c] * 1000).alias("acc_t"),
                                         pl.col("p3").filter(pl.col("acc")).mean().alias("p3acc_t")).with_columns(pl.lit(c).alias("country"))
    rows.append(t)
T = pl.concat(rows).join(H, on=["country", "cls", "bucket"], how="left").fill_null(0)
T = T.with_columns((pl.col("n_t") - pl.col("n_h")).alias("extra_cands"), (pl.col("acc_t") - pl.col("TPacc_h") - pl.col("FPacc_h")).alias("extra_acc"),
                   (pl.col("acc_t") - pl.col("TPacc_h")).alias("acc_over_TP"))
out = T.select("country", "cls", "bucket", *[pl.col(k).round(1) for k in ("n_h", "n_t", "extra_cands", "TPacc_h", "FPacc_h", "acc_t", "extra_acc")], pl.col("p3acc_t").round(3)) \
       .filter((pl.col("n_t") >= 3) | (pl.col("n_h") >= 3)).sort("country", "extra_acc", descending=[False, True])
print("per 1k S1: holdout candidates n_h, test candidates n_t, holdout accepted TP/FP, test accepted acc_t, extra_acc = acc_t - holdout accepted (TP+FP)")
print(out)
for c in ("US", "India"):
    z = T.filter(pl.col("country") == c)
    print(f"{c}: candidates/1k holdout {z['n_h'].sum():.0f} test {z['n_t'].sum():.0f}; accepted/1k holdout {(z['TPacc_h'] + z['FPacc_h']).sum():.0f} test {z['acc_t'].sum():.0f};"
          f" positive extra_acc sum {z.filter(pl.col('extra_acc') > 0)['extra_acc'].sum():.1f}/1k, negative {z.filter(pl.col('extra_acc') < 0)['extra_acc'].sum():.1f}/1k;"
          f" plain-bucket extra_acc {z.filter(pl.col('bucket') == 'plain')['extra_acc'].sum():.1f}, noisy {z.filter(pl.col('bucket') == 'noisy')['extra_acc'].sum():.1f}")
