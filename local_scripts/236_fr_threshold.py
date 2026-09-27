"""France-specific conservative threshold (teammate's tau_FR idea) evaluated on OUR scores, label-free:
for French V12-accepted same-address pairs with p3 < t, estimate the true-copy share per class from the address fingerprint mixture
(train holdout baselines for true/false records, scaled to France's copy baseline) and value each removal with the exact per-S1 F change
(k = V12 set size: removing a TP loses 1-F(k-1,0,k), removing an FP gains 1-F(k-1,1,k-1))."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


BASE = {"identical@same": (0.036, 0.238), "light_edit@same": (0.168, 0.237), "swap_extra@same": (0.14, 0.205), "first_word_replaced@same": (0.14, 0.205),
        "no_overlap@same": (0.131, 0.232), "desc@same": (0.01, 0.205)}           # (share if FALSE, share if TRUE) from train holdout (191, 153)
SCALE = 0.222 / 0.238
q = lambda p: duckdb.connect().execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                                       "where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1f = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").rename({"entity_id": "s1"}).select("s1")
V = q("output_v12/matching_results.tsv").join(s1f, on="s1", how="semi")
p3 = pl.read_parquet("output_v11s3/p3/test/France.parquet", columns=["s1", "m", "p3"])
x = V.join(p3, on=["s1", "m"], how="left").with_columns(pl.col("p3").fill_null(1.0))       # rule-added pairs (no p3 row) kept as certain
ids = V.select(pl.col("m").alias("entity_id")).unique()
raw = pl.concat([pl.scan_parquet(f"work/test_s{s}.parquet").select("entity_id", pl.col("business_address").fill_null("").str.to_lowercase().alias("ra")) for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
x = x.join(raw.rename({"entity_id": "m"}), on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
strong = (pl.col("p3") >= 0.9).cast(pl.Int32)
x = x.with_columns(strong.alias("_st")).with_columns(((pl.col("_st").sum().over(["s1", "src", "ra"]) - pl.col("_st")) > 0).alias("shared"), pl.len().over("s1").alias("k"))
x = annotate(x, "test")
x = x.filter(pl.col("ra") != "")
NS = s1f.height / 1000
k = x["k"].to_numpy()
x = x.with_columns(pl.Series("loss_tp", 1 - F(k - 1, 0, k)), pl.Series("gain_fp", 1 - F(k - 1, 1, k - 1)))
rows = []
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    tot, n_rm = 0.0, 0
    for cls, (sf, st) in BASE.items():
        z = x.filter((pl.col("cls") == cls) & (pl.col("p3") < t))
        if z.height == 0: continue
        sh = z["shared"].mean(); ptrue = float(np.clip((sh - sf) / (st * SCALE - sf), 0, 1))
        val = float(((1 - ptrue) * z["gain_fp"] - ptrue * z["loss_tp"]).sum())
        rows.append((t, cls, z.height, round(z.height / NS, 2), round(sh, 3), round(ptrue, 2), round(val / NS, 3)))
        tot += val; n_rm += z.height
    print(f"t={t}: removes {n_rm / NS:.1f}/1k French S1 (same-address classes) -> estimated France F change {tot / (NS * 1000):+.5f}  (LB {tot / 1732544:+.5f})")
print(pl.DataFrame(rows, schema=["t", "cls", "removed", "per1k", "fingerprint", "est_true_share", "value_per1k"], orient="row"))
