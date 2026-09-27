"""French dotted legal-form fix (post-hoc, decision level; normalization on AWS is unchanged so reused test features stay consistent).
Adds French same-address candidates that the model rejected when the ONLY name difference is a dotted/OCR legal-form residue
validated as copy-like by the address fingerprint (198): S.A.S./S.A. -> 's', E.U.R.L. -> 'e r u', E.I. vs EI -> 'e ei i',
S.C.I. vs SCI -> 'c i s sci', OCR '5arl'. Mixed / look-alike patterns ('+r s', '-sci', '+s u', '-ei') are NOT added.
A pool record already accepted for another S1 is never added (exclusivity).
usage: 220_fr_legal_rule.py <stage3_out_dir_with_p3> <in_matching_results.tsv> <out_dir>"""
import sys, os, json, shutil
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, duckdb
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
S3DIR, IN_TSV, OUT = sys.argv[1:4]
OK = {("s",), ("e", "r", "u"), ("e", "ei", "i"), ("c", "i", "s", "sci"), ("5arl",)}
import re
FORMS = ["SELARL", "SASU", "SARL", "EURL", "SAS", "SCI", "SNC", "SA", "EI"]
def legal_forms(raw):
    """legal forms in a raw name after collapsing dotted / spaced single letters (S.A.S. -> SAS, E. U. R. L. -> EURL, 5ARL -> SARL)."""
    t = re.sub(r"\b((?:[A-Za-z]\s*\.\s*){2,}[A-Za-z]?)\.?", lambda m: re.sub(r"[\s.]", "", m.group(1)), raw or "").upper().replace("5ARL", "SARL")
    toks = set(re.findall(r"[A-Z]+", t))
    return {f for f in FORMS if f in toks}
def nums(raw):
    return set(re.findall(r"\d+", raw or ""))
acc = duckdb.connect().execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{IN_TSV}', delim='\t', header=true, "
                               "all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
c = pl.read_parquet(os.path.join(S3DIR, "p3", "test", "France.parquet"), columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= 0.05)
c = c.join(acc, on=["s1", "m"], how="anti").join(acc.select("m").unique(), on="m", how="anti")          # rejected, record free
ids = pl.concat([c.select(pl.col("s1").alias("entity_id")), c.select(pl.col("m").alias("entity_id"))]).unique()
nm = pl.concat([pl.scan_parquet(f"work/test_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
N = {r[0]: r[1:] for r in nm.iter_rows()}
RAW = {e: (n, a) for e, n, a in pl.concat([pl.scan_parquet(f"work/test_s{s}.parquet").select("entity_id", "business_name", "business_address") for s in (1, 2, 3)])
        .join(ids.lazy(), on="entity_id", how="semi").collect().iter_rows()}
keep = []
for a, b in zip(c["s1"].to_list(), c["m"].to_list()):
    na, nb = N.get(a), N.get(b)
    if not na or not nb or not nb[1]:
        keep.append(False); continue
    t1, t2 = set(na[0].split()), set(nb[0].split())
    d = tuple(sorted((t1 - t2) | (t2 - t1)))
    ra, rb = RAW.get(a, ("", "")), RAW.get(b, ("", ""))
    fa, fb = legal_forms(ra[0]), legal_forms(rb[0])
    same_form = bool(fa) and fa == fb                                   # dotted variant of the SAME legal form (no SARL->SA swaps)
    num_ok = not nums(rb[1]) or bool(nums(rb[1]) & nums(ra[1]))         # a pool house number must appear in the S1 address
    keep.append(d in OK and same_form and num_ok and addr_rel(na[1], nb[1], na[2], nb[2]) in ("A0_identical", "A1_same_num"))
add = c.filter(pl.Series(keep)).sort("p3", descending=True).unique(subset="m", keep="first").select("s1", "m")
print(f"French legal-form rule: adds {add.height} pairs to {add['s1'].n_unique()} S1 (candidates checked {c.height})")
pred = pl.concat([acc, add])
os.makedirs(OUT, exist_ok=True)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = pred.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id") \
     .write_csv(os.path.join(OUT, "matching_results.tsv"), separator="\t", quote_style="never")
json.dump(dict(added=add.height, s1=add["s1"].n_unique()), open(os.path.join(OUT, "fr_legal_rule.json"), "w"))
