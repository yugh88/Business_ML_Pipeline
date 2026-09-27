"""Print S1s having a PREDICTED pair with signed house offset in a bucket. Usage: split country bucket n seed"""
import sys, json; sys.path.insert(0, "aws/scripts")
import polars as pl, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
sp, ctry, B, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
buckets = sorted(__import__("glob").glob(f"work/v5_p2/{sp}/*.parquet"))[:24]
d = pl.scan_parquet(buckets).filter((pl.col("country") == ctry) & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
if sp == "test":
    pred = con.execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v6/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl().join(d.select("s1").unique(), on="s1")
else:
    sys.path.insert(0, "scripts"); from tune_decision import prep, decide
    dec = json.load(open("analysis/aws_v5/decision_v5.json")); d = d.filter(pl.col("fold").is_in([0, 8, 9]))
    pred = decide(prep(d, dec["tau"], "p2x2"), dec).select("s1", "m")
    if len(sys.argv) > 6: pred = pred.join(d.filter(~pl.col("y")).select("s1", "m"), on=["s1", "m"])    # FPs only
P = set(zip(pred["s1"], pred["m"]))
hits = attach(pred, sp).filter(pl.col("off") == B)
hits = hits.sample(min(n, hits.height), seed=seed)
ids = pl.concat([d.select(pl.col("s1").alias("entity_id")), d.select(pl.col("m").alias("entity_id"))]).unique()
raw = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
R = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
for s1, mm in zip(hits["s1"], hits["m"]):
    print("=" * 150); print(f"{s1}  | {R[s1][0]} | {R[s1][1]}")
    for r in d.filter(pl.col("s1") == s1).sort("p2x2", descending=True).head(12).iter_rows(named=True):
        flag = ("PRED " if (s1, r["m"]) in P else "     ") + ("TRUE " if (sp == "train" and r["y"]) else "")
        print(f"   {flag:10s}{r['p2x2']:.3f} {r['m']:13s} | {R[r['m']][0]} | {R[r['m']][1]}{' <==' if r['m'] == mm else ''}")
