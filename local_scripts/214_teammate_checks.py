"""Teammate ideas, decisive holdout checks.
(A) Retrieval (BM25 + 4-gram + FAISS): profile of the true pairs OUR retrieval misses (p1<1e-4): name/address relation,
    empty address, how many other S1 share the exact core name (top-k cutoff problem), and whether the pool name is an
    alias with no shared characters (what a dense/char retriever could add).
(B) Graph closure for single-source S1 (all accepted copies in one source): if a candidate from the OTHER source equals an
    accepted copy on normalized name+address (or is its best other-source neighbour), accept it. Holdout precision / F, test count."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


st = pl.read_parquet("work/error_budget_pairs.parquet")                     # every holdout true pair and its stage
g = pl.read_parquet("work/error_budget_s1.parquet")
miss = st.filter(pl.col("stage") == "1_not_retrieved(p1<1e-4)").select("s1", "m")
ids = pl.concat([miss.select(pl.col("s1").alias("entity_id")), miss.select(pl.col("m").alias("entity_id"))]).unique()
nm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
N = {r[0]: r[1:] for r in nm.iter_rows()}
core_cnt = pl.scan_parquet("work/train_s1_norm.parquet").group_by("ncore").len().collect()
CC = dict(core_cnt.iter_rows())
rows = []
for a, b in zip(miss["s1"].to_list(), miss["m"].to_list()):
    na, nb = N[a], N[b]
    rows.append((name_rel(na[0], nb[0]), "empty" if not nb[1] else ("same" if addr_rel(na[1], nb[1], na[2], nb[2]) in ("A0_identical", "A1_same_num") else "diff"),
                 CC.get(na[0], 1), fuzz.ratio(na[0].replace(" ", ""), nb[0].replace(" ", "")) / 100))
x = pl.DataFrame(rows, schema=["nrel", "addr", "s1_same_core", "char_sim"], orient="row")
print(f"(A) holdout true pairs NOT retrieved: {x.height} ({x.height / st.height:.3%} of true pairs)")
print("   by pool address:", x.group_by("addr").len().with_columns((pl.col("len") / x.height).round(3)).sort("len", descending=True).rows())
print("   by name relation:", x.group_by("nrel").len().with_columns((pl.col("len") / x.height).round(3)).sort("len", descending=True).rows())
print("   identical core name but missed:", x.filter(pl.col("nrel") == "N0_identical").height, " of which the core name is shared by >=5 S1:",
      x.filter((pl.col("nrel") == "N0_identical") & (pl.col("s1_same_core") >= 5)).height)
print("   no shared word AND char similarity < 0.5 (only a semantic/dense retriever could link):", x.filter((pl.col("nrel") == "N7_no_overlap") & (pl.col("char_sim") < 0.5)).height)
# ---------------- (B) closure
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc", "cls"]).with_columns(pl.col("m").str.slice(0, 2).alias("src"))
acc = h.filter("acc")
ss = acc.group_by("s1").agg(pl.col("src").n_unique().alias("nsrc"), pl.col("src").first().alias("only"))
single = ss.filter(pl.col("nsrc") == 1)
print(f"\n(B) holdout S1 with accepted copies in ONE source only: {single.height} ({single.height / g.height:.1%} of S1)")
gt_pairs = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"]).join(single.select("s1"), on="s1", how="semi").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
other_true = gt_pairs.join(single, on="s1").filter(pl.col("src") != pl.col("only"))
print(f"   of these, S1 that truly have a copy in the OTHER source: {other_true['s1'].n_unique()} ({other_true['s1'].n_unique() / single.height:.1%}); such copies: {other_true.height}")
cand = h.join(single, on="s1").filter((pl.col("src") != pl.col("only")) & ~pl.col("acc"))
ids = pl.concat([cand.select(pl.col("m").alias("entity_id")), acc.select(pl.col("m").alias("entity_id"))]).unique()
nm2 = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "ncore", "a") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
K = dict(zip(nm2["entity_id"], (nm2["ncore"] + "|" + nm2["a"]).to_list())); NC = dict(zip(nm2["entity_id"], nm2["ncore"].to_list()))
accset = acc.group_by("s1").agg(pl.col("m"))
AM = dict(zip(accset["s1"].to_list(), accset["m"].to_list()))
def link(s1, m):
    best = 0.0
    for r in AM.get(s1, []):
        if K.get(r) == K.get(m): return 1.0
        best = max(best, fuzz.token_set_ratio(NC.get(r, ""), NC.get(m, "")) / 100)
    return best
cand = cand.with_columns(pl.Series("link", [link(a, b) for a, b in zip(cand["s1"].to_list(), cand["m"].to_list())]))
N0 = g.height
for rule, flt in (("exact name+addr equal to an accepted copy", pl.col("link") >= 1.0), ("name token-set >= 0.9 to an accepted copy", pl.col("link") >= 0.9),
                  ("best other-source candidate with p3 >= 0.3", (pl.col("p3") >= 0.3) & (pl.col("p3") == pl.col("p3").max().over("s1")))):
    add = cand.filter(flt)
    z = g.join(add.group_by("s1").agg(pl.col("y").sum().alias("atp"), (~pl.col("y")).sum().alias("afp")), on="s1", how="left").fill_null(0)
    d = F(z["tp"] + z["atp"], z["fp"] + z["afp"], z["n"]).mean() - F(z["tp"], z["fp"], z["n"]).mean()
    print(f"   closure rule [{rule}]: adds {add.height} (true {int(add['y'].sum())}, precision {add['y'].mean() if add.height else 0:.3f}) -> holdout F {d:+.5f}")
