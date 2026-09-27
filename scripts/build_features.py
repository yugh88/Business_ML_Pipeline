"""Stage features: pairwise features per S1-bucket (all candidates of an S1 live in one bucket).
Ported from scripts/42_features.py + devlib number features; adds legal-suffix agreement and blocking indicators.
Workers are spawned processes (fresh polars), each bounded by POLARS_MAX_THREADS."""
import os, sys, glob, time, multiprocessing as mp

FEATURES = [
    # name
    "n_tsr", "n_tsort", "n_ratio", "n_pr", "ns_jw", "ns_ratio", "ns_pr", "n_eq", "core_eq", "core_sorted_eq", "n_jac", "n_idf_cov",
    "n_max_idf_shared", "n_min_df1", "n_max_idf_missing", "n_ntok1", "n_ntok2", "n_shared", "n_len_ratio", "name_freq_s1",
    "legal_agree", "dom2", "indic2", "brand2", "junk2",
    # address
    "a_tsr", "a_tsort", "a_pr", "a_eq", "a2_empty", "a_jac", "a_idf_cov", "a_max_idf_shared", "a_max_idf_missing", "a_ntok1", "a_ntok2",
    "a_shared", "num_shared", "num_n1", "num_n2", "num_first_eq", "num_conflict", "state_agree", "num_min_ed", "num_first_ed",
    # retrieval / competition
    "rscore", "rrank", "m_best", "m_gap12", "r_margin_best", "c3score", "c3rank", "fscore", "frank", "in_block",
    "s1_rank_rscore", "s1_n_core_eq", "s1_ncand", "s1_rank_sum",
    # source
    "is_s3",
    # v3 additions (featx.py): number-change type, injected/missing name tokens
    "num_kind", "num_logdiff", "tok_extra", "tok_missing", "extra_generic", "missing_generic", "tok_repeat",
    # v8 additions (featx.py): signed house-number difference and twin-offset flag
    "num_sdiff", "num_plusk", "extra_desc", "extra_noise",
]
KEEP = ["s1", "m", "y", "fold", "country", "pc2", "pa", "pns", "pnum1", "pstreet"] + FEATURES


def _bucket(split, b):
    import polars as pl, numpy as np
    from rapidfuzz import process, fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    from lib import WORK, P, write_parquet_atomic
    from normlib import LEGAL
    out = f"features/{split}/bucket_{b:04d}.parquet"
    if os.path.exists(P(out)):
        return out
    d = pl.read_parquet(os.path.join(WORK, "candidates", split, f"bucket={b}", "*.parquet")).with_row_index("pid")
    if d.height == 0:
        return None
    cols = ["entity_id", "country", "n", "core2", "ns", "a", "nums", "state", "pnum1", "pstreet", "f_domain", "f_indic", "f_brand", "f_junk"]
    ng = lambda s: os.path.join(WORK, "normalized", f"{split}_s{s}", "*.parquet")
    s1 = pl.scan_parquet(ng(1)).select(cols).join(d.lazy().select(pl.col("s1").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
    pool = pl.concat([pl.scan_parquet(ng(s)).select([c for c in cols if c != "country"]) for s in (2, 3)]) \
             .join(d.lazy().select(pl.col("m").alias("entity_id")).unique(), on="entity_id", how="semi").collect()
    d = d.join(s1.rename({c: c + "_1" for c in cols if c != "entity_id"}), left_on="s1", right_on="entity_id") \
         .join(pool.rename({c: c + "_2" for c in cols if c not in ("entity_id", "country")}), left_on="m", right_on="entity_id")
    d = d.rename({"country_1": "country"})
    nf = pl.read_parquet(P(f"indexes/{split}_name_freq.parquet"))
    d = d.join(nf, left_on=["country", "core2_1"], right_on=["country", "core2"], how="left").with_columns(pl.col("name_freq_s1").fill_null(1))
    c1, c2 = d["core2_1"].to_list(), d["core2_2"].to_list(); ns1, ns2 = d["ns_1"].to_list(), d["ns_2"].to_list()
    a1 = d["a_1"].str.replace_all(" , ", " ").to_list(); a2 = d["a_2"].str.replace_all(" , ", " ").to_list()
    W = int(os.environ.get("RF_WORKERS", 2))
    F = {"n_tsr": process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=W), "n_tsort": process.cpdist(c1, c2, scorer=fuzz.token_sort_ratio, workers=W),
         "n_ratio": process.cpdist(c1, c2, scorer=fuzz.ratio, workers=W), "n_pr": process.cpdist(c1, c2, scorer=fuzz.partial_ratio, workers=W),
         "ns_jw": process.cpdist(ns1, ns2, scorer=JaroWinkler.normalized_similarity, workers=W), "ns_ratio": process.cpdist(ns1, ns2, scorer=fuzz.ratio, workers=W),
         "ns_pr": process.cpdist(ns1, ns2, scorer=fuzz.partial_ratio, workers=W), "a_tsr": process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=W),
         "a_tsort": process.cpdist(a1, a2, scorer=fuzz.token_sort_ratio, workers=W), "a_pr": process.cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=W)}
    d = d.with_columns([pl.Series(k, v, dtype=pl.Float32) for k, v in F.items()]); del F, c1, c2, ns1, ns2, a1, a2
    nss = pl.read_parquet(P(f"indexes/{split}_n_s1.parquet"))

    def overlap(side1, side2, dft, pre):
        t1 = d.select("pid", "country", side1.alias("t")).explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
        t2 = d.select("pid", side2.alias("t")).explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
        t1 = t1.join(dft, on=["country", "t"], how="left").with_columns(pl.col("df").fill_null(1)).join(nss, on="country", how="left") \
               .with_columns((pl.col("n_s1").cast(pl.Float64) / pl.col("df")).log().alias("idf"))
        sh = t1.join(t2.with_columns(pl.lit(True).alias("in2")), on=["pid", "t"], how="left").with_columns(pl.col("in2").fill_null(False))
        n2 = t2.group_by("pid").len().rename({"len": f"{pre}_ntok2"})
        g = sh.group_by("pid").agg(pl.len().alias(f"{pre}_ntok1"), pl.col("in2").sum().alias(f"{pre}_shared"),
            (pl.col("idf") * pl.col("in2")).sum().alias("_si"), pl.col("idf").sum().alias("_ti"),
            pl.col("idf").filter(pl.col("in2")).max().alias(f"{pre}_max_idf_shared"), pl.col("df").min().alias(f"{pre}_min_df1"),
            pl.col("idf").filter(~pl.col("in2")).max().alias(f"{pre}_max_idf_missing"))
        return g.join(n2, on="pid", how="left").with_columns((pl.col("_si") / pl.col("_ti")).alias(f"{pre}_idf_cov"),
            (pl.col(f"{pre}_shared") / (pl.col(f"{pre}_ntok1") + pl.col(f"{pre}_ntok2").fill_null(0) - pl.col(f"{pre}_shared"))).alias(f"{pre}_jac")).drop("_si", "_ti")
    atok = lambda c: pl.col(c).str.replace_all(",", " ").str.split(" ").list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(["null", "n", "a", "na"]))).list.unique()
    d = d.join(overlap(pl.col("core2_1").str.split(" ").list.unique(), pl.col("core2_2").str.split(" ").list.unique(),
                       pl.read_parquet(P(f"indexes/{split}_tok_df_name.parquet")), "n"), on="pid", how="left")
    d = d.join(overlap(atok("a_1"), atok("a_2"), pl.read_parquet(P(f"indexes/{split}_tok_df_addr.parquet")), "a"), on="pid", how="left")

    def close(a, b):
        A, B_ = a.split(), b.split()
        if not A or not B_: return -1
        best = 9
        for x in A:
            for z in B_:
                dd = Levenshtein.distance(x, z)
                if dd and (x.startswith(z) or z.startswith(x)): dd = 1
                best = min(best, dd)
        return best
    N1, N2 = d["nums_1"].to_list(), d["nums_2"].to_list()
    legal = lambda s: frozenset(t for t in s.split() if t in LEGAL)
    L1, L2 = [legal(x) for x in d["n_1"].to_list()], [legal(x) for x in d["n_2"].to_list()]
    num = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    d = d.with_columns(
        pl.Series("num_min_ed", [close(a, b) for a, b in zip(N1, N2)], dtype=pl.Int16),
        pl.Series("num_first_ed", [Levenshtein.distance(a.split()[0], b.split()[0]) if a and b else -1 for a, b in zip(N1, N2)], dtype=pl.Int16),
        pl.Series("legal_agree", [0 if not (x and y) else (1 if x == y else -1) for x, y in zip(L1, L2)], dtype=pl.Int8),
        (pl.col("n_1") == pl.col("n_2")).alias("n_eq"), (pl.col("core2_1") == pl.col("core2_2")).alias("core_eq"),
        (pl.col("core2_1").str.split(" ").list.sort() == pl.col("core2_2").str.split(" ").list.sort()).alias("core_sorted_eq"),
        (pl.col("a_1") == pl.col("a_2")).alias("a_eq"), (pl.col("a_2") == "").alias("a2_empty"),
        num("nums_1").list.set_intersection(num("nums_2")).list.len().alias("num_shared"),
        num("nums_1").list.len().alias("num_n1"), num("nums_2").list.len().alias("num_n2"),
        (pl.col("pnum1_1") == pl.col("pnum1_2")).alias("num_first_eq"),
        pl.when((pl.col("state_1") == "") | (pl.col("state_2") == "")).then(0).when(pl.col("state_1") == pl.col("state_2")).then(1).otherwise(-1).alias("state_agree"),
        (pl.col("core2_1").str.len_chars().cast(pl.Float32) / pl.col("core2_2").str.len_chars().clip(1)).alias("n_len_ratio"),
        pl.col("f_domain_2").alias("dom2"), pl.col("f_indic_2").alias("indic2"), pl.col("f_brand_2").alias("brand2"), pl.col("f_junk_2").alias("junk2"),
        pl.col("rrank").fill_null(99), pl.col("rscore").fill_null(0), pl.col("c3rank").fill_null(99), pl.col("c3score").fill_null(0),
        pl.col("frank").fill_null(99), pl.col("fscore").fill_null(0), (pl.col("m").str.slice(0, 2) == "S3").cast(pl.Int8).alias("is_s3"),
        pl.col("core2_2").alias("pc2"), pl.col("a_2").alias("pa"), pl.col("ns_2").alias("pns"), pl.col("pnum1_2").alias("pnum1"), pl.col("pstreet_2").alias("pstreet"))
    d = d.with_columns((pl.col("rscore") - pl.col("m_best")).alias("r_margin_best"), (pl.col("m_best") - pl.col("m_2nd")).alias("m_gap12"),
                       ((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0) & (pl.col("num_shared") == 0)).alias("num_conflict"))
    d = d.with_columns(pl.col("rscore").rank("ordinal", descending=True).over("s1").alias("s1_rank_rscore"),
                       pl.col("core_eq").sum().over("s1").alias("s1_n_core_eq"), pl.len().over("s1").alias("s1_ncand"),
                       (pl.col("n_tsr") + pl.col("a_tsr")).rank("ordinal", descending=True).over("s1").alias("s1_rank_sum"))
    from featx import add_extra
    d = add_extra(d)
    write_parquet_atomic(d.select(KEEP), out)
    return out


def _worker(args):
    split, buckets, env = args
    os.environ.update(env)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return [_bucket(split, b) for b in buckets]


def main():
    from lib import CFG, P, log, done, mark_done, s3_push
    import polars as pl
    for split in CFG["splits"]:
        if done(f"features_{split}"):
            continue
        NB = CFG["features"]["buckets"][split]; W = CFG["features"]["workers"]
        env = {"POLARS_MAX_THREADS": str(max(1, (os.cpu_count() or 4) // W)), "RF_WORKERS": "2"}
        todo = [b for b in range(NB) if not os.path.exists(P(f"features/{split}/bucket_{b:04d}.parquet"))]
        t = time.time(); log(f"features {split}: {len(todo)} buckets, {W} workers")
        if W <= 1:
            for b in todo: _bucket(split, b)
        else:
            with mp.get_context("spawn").Pool(W, maxtasksperchild=4) as pool:
                for i, _ in enumerate(pool.imap_unordered(_worker, [(split, [b], env) for b in todo])):
                    if i % 10 == 0: log(f"  features {split} {i}/{len(todo)} {time.time()-t:.0f}s")
        s3_push(f"features/{split}", recursive=True)
        mark_done(f"features_{split}", {"buckets": NB})


if __name__ == "__main__":
    main()
