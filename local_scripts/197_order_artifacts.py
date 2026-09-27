"""Diagnostic only (not a feature): do record IDs or file row order carry generation-order information?
Train truth pairs vs random S1-pool pairs vs hard negatives (V9 holdout false candidates). If |delta| of ID numbers or row
positions is far smaller for true pairs, the data has an ordering artifact (a leak) — reported, never used."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np


def ids(split, s):
    return pl.read_parquet(f"work/{split}_s{s}.parquet", columns=["entity_id"]).with_row_index("row").with_columns(
        pl.col("entity_id").str.split("-").list.last().cast(pl.Int64).alias("num"))


S1 = ids("train", 1).rename({"entity_id": "s1", "row": "r1", "num": "n1"})
P = pl.concat([ids("train", s).with_columns(pl.lit(f"S{s}").alias("src")) for s in (2, 3)]).rename({"entity_id": "m", "row": "r2", "num": "n2"})
tr = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"]).join(S1, on="s1").join(P, on="m")
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "cls"]).filter(~pl.col("y")).join(S1, on="s1").join(P, on="m")
rng = np.random.default_rng(0)
rnd = pl.DataFrame({"i": rng.integers(0, S1.height, 500_000), "j": rng.integers(0, P.height, 500_000)})
rnd = rnd.with_columns(S1["r1"].gather(rnd["i"]).alias("r1"), S1["n1"].gather(rnd["i"]).alias("n1"),
                       P["r2"].gather(rnd["j"]).alias("r2"), P["n2"].gather(rnd["j"]).alias("n2"), P["src"].gather(rnd["j"]).alias("src"))
n_s1, n_p = S1.height, P.group_by("src").len()
print("S1 rows", n_s1, "| pool rows by source", n_p.rows(), "| id ranges S1", S1["n1"].min(), S1["n1"].max(), "pool", P["n2"].min(), P["n2"].max())
for name, d in (("TRUE pairs", tr), ("HARD negatives (V9 holdout false cands)", h), ("RANDOM pairs", rnd)):
    for src in ("S2", "S3"):
        x = d.filter(pl.col("src") == src)
        rr = (x["r2"].cast(pl.Float64) / x["r2"].max() - x["r1"].cast(pl.Float64) / n_s1).abs()
        nn = (x["n2"] - x["n1"]).abs().cast(pl.Float64)
        print(f"{name:42s} {src}: n={x.height:>8}  |rel. row gap| median {rr.median():.4f} p05 {rr.quantile(0.05):.4f}  "
              f"corr(row) {np.corrcoef(x['r1'].to_numpy(), x['r2'].to_numpy())[0, 1]:+.4f}  |id gap| median {nn.median():.3g} p05 {nn.quantile(0.05):.3g}  "
              f"corr(id) {np.corrcoef(x['n1'].to_numpy(), x['n2'].to_numpy())[0, 1]:+.4f}")
# within the pool: are copies of the same S1 adjacent in file order?
g = tr.group_by("s1", "src").agg(pl.col("r2").sort()).filter(pl.col("r2").list.len() >= 2).with_columns(pl.col("r2").list.diff().list.drop_nulls().list.min().alias("gap"))
print("copies of the same S1 in the same source: min row gap median", g["gap"].median(), "share adjacent (gap 1)", round(float((g["gap"] == 1).mean()), 4), "of", g.height)
