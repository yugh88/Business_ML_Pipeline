"""Bounded sparse TF-IDF top-k retrieval (from scripts/retrieval2.py, measured safe locally).
Index built once in the parent; query shards processed by forked workers (copy-on-write index),
each worker capped with RLIMIT_AS and a bounded product size. Workers use numpy/pyarrow only (no polars after fork)."""
import os, time, numpy as np, scipy.sparse as sp, pyarrow as pa, pyarrow.parquet as pq, psutil, multiprocessing as mp
from sklearn.feature_extraction.text import HashingVectorizer

NF = 1 << 23


def texts(core2, a, view):
    """core2/a: python lists. view: na | c3"""
    if view == "na":
        return [" ".join("n_" + t for t in c.split()) + " " + x.replace(",", " ") for c, x in zip(core2, a)]
    if view == "c3":
        return [c.replace(" ", "") for c in core2]
    raise ValueError(view)


def _hv(view):
    if view == "c3":
        return HashingVectorizer(analyzer="char", ngram_range=(3, 3), n_features=NF, alternate_sign=False, norm=None, binary=True, lowercase=False)
    return HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=NF, alternate_sign=False, norm=None, binary=True, lowercase=False)


def vec(txts, view, chunk=500_000):
    hv = _hv(view)
    if not txts:
        return sp.csr_matrix((0, NF), dtype=np.float32)
    return sp.vstack([hv.transform(txts[i:i + chunk]) for i in range(0, len(txts), chunk)]).tocsr().astype(np.float32)


def weight(X, idf, keep):
    X = X.copy()
    X.data *= idf[X.indices]
    X.data[~keep[X.indices]] = 0
    X.eliminate_zeros()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()); nrm[nrm == 0] = 1
    return sp.diags(1 / nrm).dot(X).tocsr()


def build_index(X_raw, cap):
    N = X_raw.shape[0]
    df = np.bincount(X_raw.indices, minlength=NF)
    idf = np.zeros(NF, np.float32); nz = df > 0
    idf[nz] = np.log(N / df[nz]) + 1
    keep = (df > 0) & (df <= cap)
    return weight(X_raw, idf, keep).T.tocsr(), idf, keep, (df * keep).astype(np.float32)


def topk(Q, PT, dfk, K, max_out):
    est = np.asarray(Q.astype(bool).astype(np.float32) @ dfk).ravel()
    qs, ps, ss = [], [], []
    i, n = 0, Q.shape[0]
    while i < n:
        j, acc = i, 0.0
        while j < n and (acc + est[j] <= max_out or j == i):
            acc += est[j]; j += 1
        R = (Q[i:j] @ PT).tocsr()
        for r in range(R.shape[0]):
            s, e = R.indptr[r], R.indptr[r + 1]
            if s == e:
                continue
            d = R.data[s:e]; idx = R.indices[s:e]
            if e - s > K:
                sel = np.argpartition(-d, K)[:K]; d, idx = d[sel], idx[sel]
            o = np.argsort(-d)
            qs.append(np.full(len(d), i + r, np.int32)); ps.append(idx[o].astype(np.int32)); ss.append(d[o])
        del R
        i = j
    if not qs:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    return np.concatenate(qs), np.concatenate(ps), np.concatenate(ss)


_G = {}  # shared (copy-on-write) state for forked workers


def _work(args):
    shard, lo, hi, out = args
    if os.path.exists(out):
        return out, 0, 0.0
    g = _G
    if g["mem_gb"]:
        import resource
        base = psutil.Process(os.getpid()).memory_info().vms
        lim = int(base + g["mem_gb"] * 1024 ** 3)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (ValueError, OSError):
            pass
    t = time.time()
    Q = weight(vec(texts(g["q_core2"][lo:hi], g["q_a"][lo:hi], g["view"]), g["view"]), g["idf"], g["keep"])
    qi, pi, sc = topk(Q, g["PT"], g["dfk"], g["K"], g["max_out"])
    rank = np.zeros(len(qi), np.int16)
    if len(qi):
        starts = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
        lens = np.diff(np.r_[starts, len(qi)])
        rank = (np.arange(len(qi)) - np.repeat(starts, lens) + 1).astype(np.int16)
    q_ids = np.asarray(g["q_ids"][lo:hi], dtype=object)
    i_ids = g["i_ids"]
    tab = pa.table({g["qcol"]: pa.array(q_ids[qi].tolist(), pa.string()), g["icol"]: pa.array([i_ids[k] for k in pi], pa.string()),
                    "score": pa.array(sc, pa.float32()), "rank": pa.array(rank, pa.int16())})
    pq.write_table(tab, out + ".tmp", compression="zstd"); os.replace(out + ".tmp", out)
    return out, len(qi), time.time() - t


def run(view, index_ids, index_core2, index_a, query_ids, query_core2, query_a, cap, K, out_dir, qcol, icol,
        shard_rows, workers, max_out, mem_gb, log=print):
    """Retrieve top-K index rows for every query row; write shards to out_dir. Idempotent."""
    os.makedirs(out_dir, exist_ok=True)
    t = time.time()
    PT, idf, keep, dfk = build_index(vec(texts(index_core2, index_a, view), view), cap)
    _G.clear()
    _G.update(PT=PT, idf=idf, keep=keep, dfk=dfk, K=K, max_out=max_out, view=view, mem_gb=mem_gb,
              q_ids=query_ids, q_core2=query_core2, q_a=query_a, i_ids=index_ids, qcol=qcol, icol=icol)
    jobs = [(k, lo, min(lo + shard_rows, len(query_ids)), os.path.join(out_dir, f"shard_{k:05d}.parquet"))
            for k, lo in enumerate(range(0, len(query_ids), shard_rows))]
    log(f"  {view} index={len(index_ids)} PT.nnz={PT.nnz/1e6:.0f}M queries={len(query_ids)} shards={len(jobs)} build={time.time()-t:.0f}s")
    n = 0
    if workers <= 1:
        _G["mem_gb"] = 0
        for j in jobs:
            n += _work(j)[1]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for out, rows, secs in pool.imap_unordered(_work, jobs):
                n += rows
    _G.clear()
    log(f"  {view} done rows={n} total={time.time()-t:.0f}s")
