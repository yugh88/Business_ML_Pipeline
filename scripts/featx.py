"""Extra pairwise features validated on dev (analysis/out/72-74; dev holdout 0.9707 -> 0.9786):
  num_kind / num_logdiff : HOW the first house number differs (truncation vs digit edit vs other)
  tok_*                  : injected / missing name tokens and whether they are generic (center, services, legal...)
  num_sdiff / num_plusk  : V8 — SIGNED number difference; generator twins sit at S1 number + {1,2,3,4,5,7,9} (never minus)
  extra_desc/extra_noise : V8 — injected name word is a known sibling descriptor / only known copy-noise words (word_classes.json)
Input columns: n_1, n_2 (normalized full names), pnum1_1, pnum1_2 (first house numbers)."""
import os, json
import numpy as np, polars as pl

EXTRA = ["num_kind", "num_logdiff", "tok_extra", "tok_missing", "extra_generic", "missing_generic", "tok_repeat", "num_sdiff", "num_plusk", "extra_desc", "extra_noise"]
TWIN_K = {1, 2, 3, 4, 5, 7, 9}   # generator twin offsets: pool number = S1 number + k (analysis/out/119, 140)
GEN = {"center", "centre", "services", "service", "co", "company", "corp", "corporation", "inc",   # v8: group/enterprises removed (descriptors)
       "llc", "ltd", "limited", "private", "pvt", "the", "and", "sri", "shri", "mr", "smt", "dr", "ms"}


def _kind(a, b):
    if not a or not b: return 0
    if a == b: return 1
    if b.startswith(a) or a.startswith(b): return 2
    if b.endswith(a) or a.endswith(b): return 3
    if len(a) == len(b):
        dp = [i for i in range(len(a)) if a[i] != b[i]]
        if len(dp) == 1: return 4 if dp[0] == 0 else (5 if dp[0] == len(a) - 1 else 6)
        if len(dp) == 2 and dp[1] == dp[0] + 1 and a[dp[0]] == b[dp[1]] and a[dp[1]] == b[dp[0]]: return 7
        return 8
    return 9


_WC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "word_classes.json")))
DESC = set(_WC["desc"])                                        # sibling descriptors (train match rate < 5% as extra word)
NOISE = set(_WC["noise"]) | {"llc", "inc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "plc", "llp",
                             "lp", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu", "the", "and", "of", "l", "opc"}   # copy noise


def _wc(a, b):
    extra = set(b.split()) - set(a.split())
    return int(bool(extra & DESC)), int(bool(extra) and extra <= NOISE)


def _tok(a, b):
    ta, tb = a.split(), b.split(); sa, sb = set(ta), set(tb)
    extra, miss = sb - sa, sa - sb
    return (len(extra), len(miss), int(bool(extra) and extra <= GEN), int(bool(miss) and miss <= GEN), int(len(tb) != len(sb)))


def _sdiff(a, b):
    """signed pool-minus-S1 first house number (clipped); -999 when either is missing / non-numeric."""
    if not a or not b or not a.isdigit() or not b.isdigit():
        return -999
    return max(-100, min(100, int(b[:9]) - int(a[:9])))


def add_extra(d):
    A, B = d["pnum1_1"].fill_null("").to_list(), d["pnum1_2"].fill_null("").to_list()
    N1, N2 = d["n_1"].fill_null("").to_list(), d["n_2"].fill_null("").to_list()
    T = [_tok(a, b) for a, b in zip(N1, N2)]
    W = [_wc(a, b) for a, b in zip(N1, N2)]
    return d.with_columns(
        pl.Series("num_kind", [_kind(a, b) for a, b in zip(A, B)], dtype=pl.Int8),
        pl.Series("num_logdiff", [float(np.log1p(abs(int(a[:9]) - int(b[:9])))) if a and b else -1.0 for a, b in zip(A, B)], dtype=pl.Float32),
        *[pl.Series(nm, [t[i] for t in T], dtype=pl.Int8) for i, nm in enumerate(["tok_extra", "tok_missing", "extra_generic", "missing_generic", "tok_repeat"])],
        pl.Series("num_sdiff", [_sdiff(a, b) for a, b in zip(A, B)], dtype=pl.Int16)).with_columns(
        pl.col("num_sdiff").is_in(list(TWIN_K)).cast(pl.Int8).alias("num_plusk"),
        pl.Series("extra_desc", [w[0] for w in W], dtype=pl.Int8), pl.Series("extra_noise", [w[1] for w in W], dtype=pl.Int8))
