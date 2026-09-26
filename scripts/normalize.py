"""Stage normalize: v3 normalization (normlib) for every split/source, chunked, multi-process (pure Python workers)."""
import os, multiprocessing as mp, pyarrow as pa, pyarrow.parquet as pq
from lib import CFG, P, log, done, mark_done, s3_push
from normlib import normalize_record

COLS = ["n", "ncore", "core2", "ns", "a", "nums", "state", "pnum1", "pstreet", "f_domain", "f_indic", "f_brand", "f_junk"]

def _chunk(args):
    src, rg, out = args
    if os.path.exists(out):
        return out
    t = pq.ParquetFile(src).read_row_group(rg, columns=["entity_id", "business_name", "business_address", "country"])
    ids, nm, ad, ct = (t.column(c).to_pylist() for c in ("entity_id", "business_name", "business_address", "country"))
    recs = [normalize_record(n or "", a or "", c or "") for n, a, c in zip(nm, ad, ct)]
    cols = {"entity_id": ids, "country": ct}
    for c in COLS:
        cols[c] = [r[c] for r in recs]
    pq.write_table(pa.table(cols), out + ".tmp", compression="zstd"); os.replace(out + ".tmp", out)
    return out

def main():
    if done("normalize"):
        return
    jobs = []
    for split in CFG["splits"]:
        for s in (1, 2, 3):
            src = P("input", f"{split}_s{s}.parquet")
            d = P("normalized", f"{split}_s{s}", "x"); d = os.path.dirname(d)
            for rg in range(pq.ParquetFile(src).num_row_groups):
                jobs.append((src, rg, os.path.join(d, f"part_{rg:04d}.parquet")))
    log("normalize jobs", len(jobs))
    w = CFG["normalize"]["workers"]
    if w <= 1:
        for j in jobs: _chunk(j)
    else:
        with mp.get_context("fork").Pool(w) as pool:
            for i, _ in enumerate(pool.imap_unordered(_chunk, jobs)):
                if i % 50 == 0: log("  normalized", i, "/", len(jobs))
    s3_push("normalized", recursive=True)
    mark_done("normalize")

if __name__ == "__main__":
    main()
