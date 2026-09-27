"""Generator mechanism audit M1/M2: at the SAME address, a rewritten name (no overlap, or first word replaced) — is its
true-copy rate determined by the TYPE of the new words (synthetic brand from the syllable generator / domain / initials / real word)?
(1) train truth over all pool records vs best S1, (2) V9 holdout errors by type, (3) test density + V9 acceptance by type."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from normlib import is_brand_token
from tune_decision import prep, decide
from rapidfuzz.distance import Levenshtein
d9 = json.load(open("output_v9/decision_s3.json"))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)


def new_word_type(c1, c2, dom):
    a, b = c1.split(), c2.split()
    E = [t for t in b if t not in a]
    if not E: return "none"
    if dom: return "domain"
    if len("".join(b)) <= 4 and "".join(b) == "".join(w[0] for w in a)[:len("".join(b))]: return "initials"
    if all(is_brand_token(t) for t in E): return "synthetic_brand"
    if any(is_brand_token(t) for t in E): return "mixed_brand"
    M = [t for t in a if t not in b]
    if M and all(min(Levenshtein.distance(e, m) for m in M) <= 2 for e in E): return "typo"
    return "real_word"


def kind(nrel, c1, c2):
    if nrel == "N7_no_overlap": return "no_overlap"
    a, b = c1.split(), c2.split()
    if nrel == "N6_word_swap" and a and a[0] not in b: return "first_word_replaced"
    return None


def annotate(pairs, split):
    ids = pl.concat([pairs.select(pl.col("s1").alias("entity_id")), pairs.select(pl.col("m").alias("entity_id"))]).unique()
    srcs = [pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore") for s in (1, 2, 3)]
    nm = dict(pl.concat(srcs).join(ids.lazy(), on="entity_id", how="semi").collect().iter_rows())
    fd = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "f_domain") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    FD = dict(fd.iter_rows())
    k = [kind(r, nm.get(a, ""), nm.get(b, "")) for r, a, b in zip(pairs["nrel"], pairs["s1"], pairs["m"])]
    t = [new_word_type(nm.get(a, ""), nm.get(b, ""), FD.get(b, False)) if kk else None for kk, a, b in zip(k, pairs["s1"], pairs["m"])]
    return pairs.with_columns(pl.Series("kind", k), pl.Series("wtype", t)).filter(pl.col("kind").is_not_null())


# (1) train truth: all pool records vs best S1, same address
tr = pl.read_parquet("work/rel_train.parquet", columns=["s1", "m", "nrel", "arel", "truth"]).filter(pl.col("arel").is_in(["A0_identical", "A1_same_num"]) & pl.col("nrel").is_in(["N7_no_overlap", "N6_word_swap"]))
tr = annotate(tr, "train")
print("TRAIN (all pool records, same address): true-copy rate by rewrite kind x new-word type")
print(tr.group_by("kind", "wtype").agg(pl.len(), (pl.col("truth") == "true_copy").mean().round(3).alias("true_rate"), (pl.col("truth") == "distractor").mean().round(3).alias("distractor")).sort("kind", "len", descending=[False, True]))
# (2) V9 holdout: accepted / errors by type
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
def cand(p3, split):
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    x = p3.filter(pl.col("p1") >= d9["tau"]).join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    n = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in n.iter_rows()}
    x = x.with_columns(pl.Series("nrel", [name_rel(N[a][0], N[b][0]) for a, b in zip(x["s1"], x["m"])]),
                       pl.Series("arel", [addr_rel(N[a][1], N[b][1], N[a][2], N[b][2]) for a, b in zip(x["s1"], x["m"])]))
    x = x.filter(pl.col("arel").is_in(["A0_identical", "A1_same_num"]) & pl.col("nrel").is_in(["N7_no_overlap", "N6_word_swap"]))
    return annotate(x, split)
h = cand(pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet") for f in (0, 8, 9)]).join(SYN, on="m", how="anti"), "train")
print("\nV9 HOLDOUT (cascade candidates, same address, rewritten names): acceptance, precision, recall by type")
print(h.group_by("kind", "wtype").agg(pl.len().alias("cands"), pl.col("y").mean().round(3).alias("true_rate"), pl.col("acc").mean().round(3).alias("acc_rate"),
                                      (pl.col("acc") & ~pl.col("y")).sum().alias("FP"), (~pl.col("acc") & pl.col("y")).sum().alias("FN")).sort("kind", "cands", descending=[False, True]))
for c in ("US", "India", "France"):
    x = cand(pl.read_parquet(f"output_v9/p3/test/{c}.parquet"), "test")
    n = {"US": 663106, "India": 809986, "France": 259452}[c] / 1000
    print(f"TEST {c}:", x.group_by("kind", "wtype").agg((pl.len() / n).round(1).alias("cands_per1k"), pl.col("acc").mean().round(3).alias("acc_rate")).sort("kind", "cands_per1k", descending=[False, True]).rows())
