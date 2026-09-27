"""Eyeball: random S1 with their top candidates (p2x2), predicted flag, raw text. Usage: 114_look.py split country n seed"""
import sys, polars as pl, duckdb
sp, ctry, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
buckets = sorted(__import__("glob").glob(f"work/v5_p2/{sp}/*.parquet"))[:4]
d = pl.scan_parquet(buckets).filter((pl.col("country") == ctry) & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "p1", "p2x2").collect()
s1s = d.select("s1").unique().sample(n, seed=seed)
d = d.join(s1s, on="s1")
out = "output_v6/matching_results.tsv" if sp == "test" else None
ids = pl.concat([d.select(pl.col("s1").alias("entity_id")), d.select(pl.col("m").alias("entity_id"))]).unique()
raw = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
R = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
if sp == "test":
    pred = duckdb.connect().execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{out}', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
    P = set(zip(pred["s1"], pred["m"]))
for s1 in s1s["s1"].to_list():
    print("=" * 140); print(f"{s1}  | {R[s1][0]} | {R[s1][1]}")
    for r in d.filter(pl.col("s1") == s1).sort("p2x2", descending=True).head(10).iter_rows(named=True):
        flag = ("PRED " if (s1, r["m"]) in P else "     ") if sp == "test" else ("TRUE " if r["y"] else "     ")
        print(f"   {flag}{r['p2x2']:.3f} {r['m']:14s} | {R[r['m']][0]} | {R[r['m']][1]}")
