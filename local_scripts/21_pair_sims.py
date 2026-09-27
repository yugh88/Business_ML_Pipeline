"""Similarity profile of every TRUE pair (7.6M). Chunked by S1 hash bucket; single process (rapidfuzz threads only).
Output: work/true_pair_sims.parquet"""
import polars as pl, numpy as np, time, os
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler
NB = 10
cols = ["entity_id", "country", "n", "ncore", "a", "nums", "state", "f_domain", "f_indic", "f_brand", "f_junk"]
s1 = pl.scan_parquet("work/train_s1_norm.parquet").select(cols)
pool = pl.concat([pl.scan_parquet("work/train_s2_norm.parquet").select(cols[:1] + cols[2:]),
                  pl.scan_parquet("work/train_s3_norm.parquet").select(cols[:1] + cols[2:])])
pairs = pl.scan_parquet("work/train_pairs.parquet")
parts = []
for b in range(NB):
    t = time.time()
    p = pairs.filter(pl.col("s1").hash(7) % NB == b)
    d = (p.join(s1, left_on="s1", right_on="entity_id")
          .join(pool, left_on="m", right_on="entity_id", suffix="_2").collect())
    n1, n2 = d["n"].to_list(), d["n_2"].to_list(); c1, c2 = d["ncore"].to_list(), d["ncore_2"].to_list()
    a1, a2 = d["a"].to_list(), d["a_2"].to_list()
    ns1 = [x.replace(" ", "") for x in c1]; ns2 = [x.replace(" ", "") for x in c2]
    out = d.select("s1", "m", "src", "country",
        (pl.col("n") == pl.col("n_2")).alias("n_eq"), (pl.col("ncore") == pl.col("ncore_2")).alias("core_eq"),
        (pl.col("a") == pl.col("a_2")).alias("a_eq"), (pl.col("a_2") == "").alias("a2_empty"),
        (pl.col("state") == pl.col("state_2")).alias("state_eq"), (pl.col("state_2") == "").alias("state2_empty"),
        pl.col("f_domain_2").alias("dom2"), pl.col("f_indic_2").alias("indic2"), pl.col("f_brand_2").alias("brand2"), pl.col("f_junk_2").alias("junk2"),
        pl.col("ncore").str.split(" ").list.set_intersection(pl.col("ncore_2").str.split(" ")).list.len().alias("core_tok_inter"),
        pl.col("ncore").str.split(" ").list.len().alias("core_toks1"), pl.col("ncore_2").str.split(" ").list.len().alias("core_toks2"),
        pl.col("nums").str.split(" ").list.set_intersection(pl.col("nums_2").str.split(" ")).list.eval(pl.element().filter(pl.element() != "")).list.len().alias("num_inter"),
        (pl.col("nums") != "").alias("nums1_any"), (pl.col("nums_2") != "").alias("nums2_any"),
        pl.col("a").str.replace_all(",", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.set_intersection(
            pl.col("a_2").str.replace_all(",", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))).list.len().alias("addr_tok_inter"),
    ).with_columns(
        pl.Series("name_tsr", process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=4), dtype=pl.Float32),
        pl.Series("name_tsort", process.cpdist(c1, c2, scorer=fuzz.token_sort_ratio, workers=4), dtype=pl.Float32),
        pl.Series("name_jw_nospace", process.cpdist(ns1, ns2, scorer=JaroWinkler.normalized_similarity, workers=4), dtype=pl.Float32),
        pl.Series("name_pr_nospace", process.cpdist(ns1, ns2, scorer=fuzz.partial_ratio, workers=4), dtype=pl.Float32),
        pl.Series("addr_tsr", process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=4), dtype=pl.Float32),
    )
    parts.append(out)
    print(b, out.height, f"{time.time()-t:.0f}s", flush=True)
    del d, n1, n2, c1, c2, a1, a2, ns1, ns2
pl.concat(parts).write_parquet("work/true_pair_sims.parquet")
