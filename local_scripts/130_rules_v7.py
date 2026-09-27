"""Candidate V7 rules on top of V5 predictions. Train holdout cost (exact F) + test removals per 1k S1 + symmetry check."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
DESC_TRAIN = set(pl.read_parquet("work/descriptor_words_train.parquet")["t"].to_list())
FR_SIB = {"holding", "participations", "international", "internationale", "distribution", "snc"}
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
def enrich(pr, sp):
    pr = attach(pr, sp).with_columns(pl.col("m").str.slice(0, 2).alias("src"))
    ids1 = pr.select(pl.col("s1").alias("entity_id")).unique(); ids2 = pr.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{sp}_s1_norm.parquet").select("entity_id", "ncore").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}_norm.parquet").select("entity_id", "ncore") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    pr = pr.join(n1.rename({"entity_id": "s1", "ncore": "c1"}), on="s1").join(n2.rename({"entity_id": "m", "ncore": "c2"}), on="m")
    pr = pr.with_columns(pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).alias("E"), pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" ")).alias("M"))
    i1, i2 = pl.col("h1").cast(pl.Int64, strict=False), pl.col("h2").cast(pl.Int64, strict=False)
    pr = pr.with_columns(((i2 - i1 >= 1) & (i2 - i1 <= 9)).fill_null(False).alias("plusk"), ((pl.col("h2") == pl.col("h1")) & (pl.col("h1") != "")).alias("ish1"))
    pr = pr.with_columns(pl.col("ish1").cast(pl.Int32).sum().over("s1").alias("A_all"), pl.col("ish1").cast(pl.Int32).sum().over(["s1", "src"]).alias("A_src"),
                         pl.when(pl.col("plusk")).then(pl.col("src")).otherwise(None).n_unique().over(["s1", "h2"]).alias("grp_nsrc_raw"))
    pr = pr.with_columns(pl.col("src").filter(pl.col("plusk")).n_unique().over(["s1", "h2"]).alias("grp_nsrc"))
    return pr
RULES = {
    "R1_twin_same_src": lambda x: x["plusk"] & (x["A_src"] >= 1),
    "R2_twin_2src": lambda x: x["plusk"] & (x["A_all"] >= 1) & (x["grp_nsrc"] >= 2),
    "R3_fr_sib_extra": lambda x: (x["country"] == "France") & (x["M"].list.len() == 0) & x["E"].list.eval(pl.element().is_in(list(FR_SIB))).list.any(),
    "R4_desc_usin": lambda x: (x["country"] != "France") & x["E"].list.eval(pl.element().is_in(list(DESC_TRAIN))).list.any(),
}
COMBOS = {"R1": ["R1_twin_same_src"], "R1+R2": ["R1_twin_same_src", "R2_twin_2src"], "R4": ["R4_desc_usin"], "R1+R2+R4": ["R1_twin_same_src", "R2_twin_2src", "R4_desc_usin"],
          "ALL": ["R1_twin_same_src", "R2_twin_2src", "R3_fr_sib_extra", "R4_desc_usin"]}
def mask(x, names):
    m = pl.Series([False] * x.height)
    for n in names: m = m | RULES[n](x)
    return m
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
for f in []:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
    pr = decide(prep(d.drop("country"), dec["tau"], "p2x2"), dec).select("s1", "m").join(d.select("s1", "m", "y", "country"), on=["s1", "m"]); del d
    pr = enrich(pr, "train"); s1f = gt.filter(pl.col("fold") == f)
    def score(keep):
        x = pr.filter(keep).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
        g = s1f.join(x, on="s1", how="left").fill_null(0)
        return float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean())
    line = f"fold {f}: V5 {score(pl.lit(True)):.5f}"
    for k, names in COMBOS.items():
        m = mask(pr, names); line += f" | {k} {score(~m):.5f} (-TP {int((m & pr['y']).sum())} -FP {int((m & ~pr['y']).sum())})"
    print(line, flush=True)
p5 = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v5/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1t = dict(s1c.group_by("country").len().rows())
te = enrich(p5.join(s1c, on="s1"), "test")
for c in ["US", "India", "France"]:
    x = te.filter(pl.col("country") == c); n = nS1t[c] / 1000
    print(f"TEST {c}: " + " | ".join(f"{k} removes {v(x).sum()/n:.1f}/1k" for k, v in RULES.items()) + " | " + " | ".join(f"{k}: {mask(x, v).sum()/n:.1f}/1k" for k, v in COMBOS.items()))
te.with_columns(*[RULES[k](te).alias(k) for k in RULES]).select("s1", "m", "country", *RULES.keys()).write_parquet("work/test_v5_rules.parquet")
