"""Sparse TF-IDF top-k retrieval with document-frequency caps, parallelised over query chunks."""
import numpy as np
import polars as pl
import scipy.sparse as sp
from multiprocessing import Pool

_P = None  # pool matrix (CSR, transposed lazily) shared with forked workers
_K = None


def tokens_expr(col, kind):
    c = pl.col(col)
    if kind == "word":
        return c.str.replace_all(",", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    if kind.startswith("char"):
        n = int(kind[4:])
        # character n-grams over the space-free string (handles concatenated handles/domains)
        return c.str.replace_all(r"[ ,]", "").map_elements(
            lambda s: list({s[i:i + n] for i in range(max(1, len(s) - n + 1))}) if s else [],
            return_dtype=pl.List(pl.Utf8))
    raise ValueError(kind)


def build(query_df, pool_df, col, kind, df_cap, min_df=1):
    """Return (Q, P) L2-normalised TF-IDF CSR matrices sharing a pool-derived vocabulary."""
    pt = pool_df.select(pl.int_range(pl.len()).alias("r"), tokens_expr(col, kind).alias("t")).explode("t").drop_nulls()
    qt = query_df.select(pl.int_range(pl.len()).alias("r"), tokens_expr(col, kind).alias("t")).explode("t").drop_nulls()
    N = pool_df.height
    voc = pt.group_by("t").len().filter((pl.col("len") <= df_cap) & (pl.col("len") >= min_df))
    voc = voc.with_columns(pl.int_range(pl.len()).alias("tid"), (np.log(N / pl.col("len")) + 1).alias("idf"))
    mats = []
    for tab, nrows in ((qt, query_df.height), (pt, N)):
        j = tab.join(voc, on="t", how="inner")
        m = sp.csr_matrix((j["idf"].to_numpy().astype(np.float32), (j["r"].to_numpy(), j["tid"].to_numpy())),
                          shape=(nrows, voc.height))
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1
        mats.append(sp.diags(1 / norms).dot(m).tocsr())
    return mats[0], mats[1]


def _topk_chunk(Qc):
    R = (Qc @ _P).tocsr()
    rows, cols, vals = [], [], []
    for i in range(R.shape[0]):
        s, e = R.indptr[i], R.indptr[i + 1]
        if s == e:
            continue
        d = R.data[s:e]
        idx = R.indices[s:e]
        if e - s > _K:
            sel = np.argpartition(-d, _K)[:_K]
            d, idx = d[sel], idx[sel]
        rows.append(np.full(len(d), i, dtype=np.int32))
        cols.append(idx.astype(np.int32))
        vals.append(d.astype(np.float32))
    if not rows:
        return np.array([], np.int32), np.array([], np.int32), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def topk(Q, P, k, chunk=2000, procs=10):
    """Top-k pool rows per query row by cosine. Returns (qrow, prow, score) arrays."""
    global _P, _K
    _P = P.T.tocsc()
    _K = k
    chunks = [Q[i:i + chunk] for i in range(0, Q.shape[0], chunk)]
    offs = list(range(0, Q.shape[0], chunk))
    with Pool(procs) as pool:
        res = pool.map(_topk_chunk, chunks)
    q = np.concatenate([r[0] + o for r, o in zip(res, offs)])
    p = np.concatenate([r[1] for r in res])
    s = np.concatenate([r[2] for r in res])
    return q, p, s
