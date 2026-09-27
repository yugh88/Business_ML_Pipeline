"""Memory-safe sparse TF-IDF top-k retrieval (replacement for retrieval.py, which froze the machine).
Single process. Stateless hashing (no vocabulary dict). Query chunks sized by estimated matmul output.
"""
import numpy as np, scipy.sparse as sp, polars as pl, psutil, os, time
from sklearn.feature_extraction.text import HashingVectorizer

NF = 1 << 23
HONOR = r"^((shri|sri|smt|mr|mrs|ms|dr|m s|messrs) )+"
TAIL = r" (id [0-9]+|[0-9]{6,}|com|c0m|in|net|org)( |$).*$"


def core2_expr(col="ncore"):
    return pl.col(col).str.replace(HONOR, "").str.replace(TAIL, "").str.replace(r"com$", "").str.strip_chars()


def rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def texts(df, view):
    """Return list of strings to vectorize for a view."""
    if view == "name":
        return df.select(core2_expr()).to_series().to_list()
    if view == "addr":
        return df["a"].str.replace_all(",", " ").to_list()
    if view == "na":
        return df.select(pl.concat_str([core2_expr().str.replace_all(r"(\S+)", "n_$1"), pl.lit(" "),
                                        pl.col("a").str.replace_all(",", " ")])).to_series().to_list()
    if view in ("c3", "c4"):
        return df.select(core2_expr().str.replace_all(" ", "")).to_series().to_list()
    raise ValueError(view)


def vectorizer(view):
    if view in ("c3", "c4"):
        n = int(view[1])
        return HashingVectorizer(analyzer="char", ngram_range=(n, n), n_features=NF, alternate_sign=False, norm=None, binary=True, lowercase=False)
    return HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=NF, alternate_sign=False, norm=None, binary=True, lowercase=False)


def vec(txts, view, chunk=500_000):
    hv = vectorizer(view)
    return sp.vstack([hv.transform(txts[i:i + chunk]) for i in range(0, len(txts), chunk)]).tocsr().astype(np.float32)


def weight(X, idf, keep):
    X = X[:, :]  # copy
    X.data *= idf[X.indices]
    X.data[~keep[X.indices]] = 0
    X.eliminate_zeros()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()); nrm[nrm == 0] = 1
    return sp.diags(1 / nrm).dot(X).tocsr()


def build_pool(P_raw, cap):
    N = P_raw.shape[0]
    df = np.bincount(P_raw.indices, minlength=NF)
    idf = np.zeros(NF, np.float32); nz = df > 0
    idf[nz] = np.log(N / df[nz]) + 1
    keep = (df > 0) & (df <= cap)
    P = weight(P_raw, idf, keep)
    return P.T.tocsr(), idf, keep, df  # PT: features x pool


def topk(Q, PT, df, keep, K, max_out=25_000_000, work_budget=None):
    """Yield (qrow, prow, score) arrays; chunks sized so the product result stays below max_out nonzeros."""
    est = np.asarray(Q.astype(bool).astype(np.float32) @ (df * keep).astype(np.float32)).ravel()  # per-query upper bound
    tot = est.sum()
    if work_budget and tot > work_budget:
        raise RuntimeError(f"estimated work {tot:.3g} > budget {work_budget:.3g}")
    order_q, order_p, order_s = [], [], []
    i = 0; n = Q.shape[0]
    while i < n:
        j = i; acc = 0
        while j < n and (acc + est[j] <= max_out or j == i):
            acc += est[j]; j += 1
        R = (Q[i:j] @ PT).tocsr()
        for r in range(R.shape[0]):
            s, e = R.indptr[r], R.indptr[r + 1]
            if s == e: continue
            d = R.data[s:e]; idx = R.indices[s:e]
            if e - s > K:
                sel = np.argpartition(-d, K)[:K]; d = d[sel]; idx = idx[sel]
            order_q.append(np.full(len(d), i + r, np.int32)); order_p.append(idx.astype(np.int32)); order_s.append(d)
        del R
        i = j
    if not order_q:
        return np.array([], np.int32), np.array([], np.int32), np.array([], np.float32), tot
    return np.concatenate(order_q), np.concatenate(order_p), np.concatenate(order_s), tot
