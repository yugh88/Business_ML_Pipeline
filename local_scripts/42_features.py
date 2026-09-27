"""Pairwise features on dev candidates (work/dev_cands.parquet) -> work/dev_feats.parquet.
Token rarity from S1 (per country) -> work/tok_df_name.parquet, work/tok_df_addr.parquet (built once)."""
import sys, os, re, time, polars as pl, numpy as np
sys.path.insert(0, "scripts"); import memguard
sys.path.insert(0, "scripts")
from retrieval2 import core2_expr
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
t0 = time.time()
GEN = ["services", "service", "enterprises", "enterprise"]
def c2(col="ncore"):
    return core2_expr(col)
def atoks(col):
    return pl.col(col).str.replace_all(",", " ").str.split(" ").list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(["null", "n", "a", "na"]))).list.unique()
if not os.path.exists("work/tok_df_name.parquet"):
    s1 = pl.scan_parquet("work/train_s1_norm.parquet")
    s1.select("country", c2().alias("c")).with_columns(pl.col("c").str.split(" ").list.unique()).explode("c").group_by("country", "c").len() \
      .rename({"c": "t", "len": "df"}).collect().write_parquet("work/tok_df_name.parquet")
    s1.select("country", atoks("a").alias("t")).explode("t").group_by("country", "t").len().rename({"len": "df"}).collect().write_parquet("work/tok_df_addr.parquet")
    s1.select("country", c2().alias("c")).group_by("country", "c").len().rename({"len": "name_freq_s1"}).collect().write_parquet("work/name_freq_s1.parquet")
NS1 = {"US": 1323633, "India": 883188}
dfn = pl.read_parquet("work/tok_df_name.parquet"); dfa = pl.read_parquet("work/tok_df_addr.parquet")
nf = pl.read_parquet("work/name_freq_s1.parquet")
d = pl.read_parquet("work/dev_cands.parquet").with_row_index("pid")
cols = ["entity_id", "n", "ncore", "a", "nums", "state", "f_domain", "f_indic", "f_brand", "f_junk"]
s1 = pl.scan_parquet("work/train_s1_norm.parquet").select(cols + ["country"]).join(d.lazy().select(pl.col("s1").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
pool = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select(cols) for s in (2, 3)]).join(d.lazy().select(pl.col("m").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
d = d.join(s1.rename({c: c + "_1" for c in cols if c != "entity_id"}), left_on="s1", right_on="entity_id") \
     .join(pool.rename({c: c + "_2" for c in cols if c != "entity_id"}), left_on="m", right_on="entity_id")
d = d.with_columns(c2("ncore_1").alias("c1"), c2("ncore_2").alias("c2"), pl.col("m").str.slice(0, 2).alias("src"))
d = d.join(nf, left_on=["country", "c1"], right_on=["country", "c"], how="left").with_columns(pl.col("name_freq_s1").fill_null(1))
print("joined", d.height, f"{time.time()-t0:.0f}s", flush=True)
c1, c2l = d["c1"].to_list(), d["c2"].to_list()
ns1 = [x.replace(" ", "") for x in c1]; ns2 = [x.replace(" ", "") for x in c2l]
a1 = d["a_1"].str.replace_all(" , ", " ").to_list(); a2 = d["a_2"].str.replace_all(" , ", " ").to_list()
W = 4
F = {
 "n_tsr": process.cpdist(c1, c2l, scorer=fuzz.token_set_ratio, workers=W),
 "n_tsort": process.cpdist(c1, c2l, scorer=fuzz.token_sort_ratio, workers=W),
 "n_ratio": process.cpdist(c1, c2l, scorer=fuzz.ratio, workers=W),
 "n_pr": process.cpdist(c1, c2l, scorer=fuzz.partial_ratio, workers=W),
 "ns_jw": process.cpdist(ns1, ns2, scorer=JaroWinkler.normalized_similarity, workers=W),
 "ns_ratio": process.cpdist(ns1, ns2, scorer=fuzz.ratio, workers=W),
 "ns_pr": process.cpdist(ns1, ns2, scorer=fuzz.partial_ratio, workers=W),
 "a_tsr": process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=W),
 "a_tsort": process.cpdist(a1, a2, scorer=fuzz.token_sort_ratio, workers=W),
 "a_pr": process.cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=W),
}
d = d.with_columns([pl.Series(k, v, dtype=pl.Float32) for k, v in F.items()]); del F, c1, c2l, ns1, ns2, a1, a2
print("rapidfuzz", f"{time.time()-t0:.0f}s", flush=True)
# token overlap + rarity-weighted overlap (idf from S1 of same country)
def overlap(side1, side2, dft, pre):
    t1 = d.select("pid", "country", side1.alias("t")).explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
    t2 = d.select("pid", side2.alias("t")).explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
    t1 = t1.unique().join(dft, on=["country", "t"], how="left").with_columns(pl.col("df").fill_null(1))
    t1 = t1.with_columns((pl.col("country").replace_strict(NS1, default=1_000_000).cast(pl.Float64) / pl.col("df")).log().alias("idf"))
    sh = t1.join(t2.with_columns(pl.lit(True).alias("in2")), on=["pid", "t"], how="left").with_columns(pl.col("in2").fill_null(False))
    n2 = t2.group_by("pid").len().rename({"len": f"{pre}_ntok2"})
    g = sh.group_by("pid").agg(pl.len().alias(f"{pre}_ntok1"), pl.col("in2").sum().alias(f"{pre}_shared"),
        (pl.col("idf") * pl.col("in2")).sum().alias("_si"), pl.col("idf").sum().alias("_ti"),
        pl.col("idf").filter(pl.col("in2")).max().alias(f"{pre}_max_idf_shared"), pl.col("df").min().alias(f"{pre}_min_df1"),
        pl.col("idf").filter(~pl.col("in2")).max().alias(f"{pre}_max_idf_missing"))
    g = g.join(n2, on="pid", how="left").with_columns((pl.col("_si") / pl.col("_ti")).alias(f"{pre}_idf_cov"),
        (pl.col(f"{pre}_shared") / (pl.col(f"{pre}_ntok1") + pl.col(f"{pre}_ntok2").fill_null(0) - pl.col(f"{pre}_shared"))).alias(f"{pre}_jac")).drop("_si", "_ti")
    return g
split = lambda c: pl.col(c).str.split(" ").list.unique()
d = d.join(overlap(split("c1"), split("c2"), dfn, "n"), on="pid", how="left")
d = d.join(overlap(atoks("a_1"), atoks("a_2"), dfa, "a"), on="pid", how="left")
print("overlap", f"{time.time()-t0:.0f}s", flush=True)
num = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
d = d.with_columns(
    (pl.col("n_1") == pl.col("n_2")).alias("n_eq"), (pl.col("c1") == pl.col("c2")).alias("core_eq"),
    (pl.col("c1").str.split(" ").list.sort() == pl.col("c2").str.split(" ").list.sort()).alias("core_sorted_eq"),
    (pl.col("a_1") == pl.col("a_2")).alias("a_eq"), (pl.col("a_2") == "").alias("a2_empty"),
    num("nums_1").list.set_intersection(num("nums_2")).list.len().alias("num_shared"),
    num("nums_1").list.len().alias("num_n1"), num("nums_2").list.len().alias("num_n2"),
    (pl.col("nums_1").str.split(" ").list.first() == pl.col("nums_2").str.split(" ").list.first()).alias("num_first_eq"),
    pl.when((pl.col("state_1") == "") | (pl.col("state_2") == "")).then(0).when(pl.col("state_1") == pl.col("state_2")).then(1).otherwise(-1).alias("state_agree"),
    (pl.col("c1").str.len_chars().cast(pl.Float32) / pl.col("c2").str.len_chars().clip(1)).alias("n_len_ratio"),
    pl.col("f_domain_2").alias("dom2"), pl.col("f_indic_2").alias("indic2"), pl.col("f_brand_2").alias("brand2"), pl.col("f_junk_2").alias("junk2"),
    pl.col("rrank").fill_null(99).alias("rrank"), pl.col("rscore").fill_null(0).alias("rscore"),
)
d = d.with_columns((pl.col("rscore") - pl.col("m_best")).alias("r_margin_best"), (pl.col("m_best") - pl.col("m_2nd").fill_null(0)).alias("m_gap12"),
    ((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0) & (pl.col("num_shared") == 0)).alias("num_conflict"))
# per-S1 context: rank of this candidate among the S1's candidates by reverse score / name tsr; number of core-equal candidates
d = d.with_columns(pl.col("rscore").rank("ordinal", descending=True).over("s1").alias("s1_rank_rscore"),
    pl.col("core_eq").sum().over("s1").alias("s1_n_core_eq"), pl.len().over("s1").alias("s1_ncand"),
    (pl.col("n_tsr") + pl.col("a_tsr")).rank("ordinal", descending=True).over("s1").alias("s1_rank_sum"))
keep = [c for c in d.columns if c not in ("n_1", "n_2", "ncore_1", "ncore_2", "a_1", "a_2", "nums_1", "nums_2", "state_1", "state_2", "c1", "c2",
        "f_domain_1", "f_indic_1", "f_brand_1", "f_junk_1", "f_domain_2", "f_indic_2", "f_brand_2", "f_junk_2")]
d.select(keep).write_parquet("work/dev_feats.parquet")
print("done", d.height, len(keep), f"{time.time()-t0:.0f}s")
print(memguard.report())
