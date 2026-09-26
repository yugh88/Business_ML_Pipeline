"""Collective (stage-2) features from stage-1 probabilities (validated in analysis/14: dev holdout 0.9625 -> 0.9707).
Group features per S1 are exact within a bucket (all candidates of an S1 share a bucket); per-pool-record competition
uses the GLOBAL table mstat(m, mx, sd) = best / 2nd-best stage-1 probability of m over all S1."""
import polars as pl, numpy as np
from rapidfuzz import process, fuzz

SIB_KEYS = [("pc2", "name"), ("pa", "addr"), ("pns", "ns"), ("pnum1", "num"), ("pstreet", "street")]


def _top2(col, grp, tag):
    return [pl.col(col).max().over(grp).alias(f"_{tag}_mx"),
            pl.col(col).sort(descending=True).get(1, null_on_oob=True).over(grp).fill_null(0).alias(f"_{tag}_2")]


def _excl(col, tag):
    return pl.when(pl.col(col) == pl.col(f"_{tag}_mx")).then(pl.col(f"_{tag}_2")).otherwise(pl.col(f"_{tag}_mx"))


def add_collective(d, mstat, pcol="p1", pre="c1", workers=2):
    d = d.join(mstat.rename({"mx": "_m_mx", "sd": "_m_2"}), on="m", how="left")
    d = d.with_columns(*_top2(pcol, ["s1"], "s"),
                       pl.col(pcol).rank("ordinal", descending=True).over("s1").alias(f"{pre}_rank_s1"),
                       (pl.col(pcol) > 0.5).sum().over("s1").alias(f"{pre}_s1_n50"), pl.col(pcol).sum().over("s1").alias(f"{pre}_s1_sum"))
    d = d.with_columns(pl.col("_s_mx").alias(f"{pre}_s1_max"), _excl(pcol, "s").alias(f"{pre}_s1_other"), _excl(pcol, "m").alias(f"{pre}_m_other"))
    out = [f"{pre}_rank_s1", f"{pre}_s1_n50", f"{pre}_s1_sum", f"{pre}_s1_max", f"{pre}_s1_other", f"{pre}_m_other"]
    for key, nm in SIB_KEYS:
        g = ["s1", key]
        d = d.with_columns(*_top2(pcol, g, "k"), pl.len().over(g).alias(f"{pre}_sib_{nm}_n"))
        d = d.with_columns(pl.when(pl.col(key).is_null() | (pl.col(key) == "") | (pl.col(f"{pre}_sib_{nm}_n") <= 1)).then(-1.0)
                           .otherwise(_excl(pcol, "k")).alias(f"{pre}_sib_{nm}_max"))
        out += [f"{pre}_sib_{nm}_n", f"{pre}_sib_{nm}_max"]
    d = d.with_columns(pl.col(pcol).rank("ordinal", descending=True).over("s1").alias("_r"))
    anc = d.filter(pl.col("_r") <= 2).select("s1", "_r", "m", pcol, "pc2", "pa", "pnum1").rename({pcol: "_p", "m": "_am"}) \
           .pivot(on="_r", index="s1", values=["_am", "_p", "pc2", "pa", "pnum1"])
    for c in ("_am_2", "_p_2", "pc2_2", "pa_2", "pnum1_2"):
        if c not in anc.columns:
            anc = anc.with_columns(pl.lit(None, dtype=pl.Float32 if c == "_p_2" else pl.Utf8).alias(c))
    d = d.join(anc, on="s1", how="left")
    isa = (d["m"] == d["_am_1"]).fill_null(False).to_numpy()
    pick = lambda c: np.where(isa, d[f"{c}_2"].fill_null("").to_numpy(), d[f"{c}_1"].fill_null("").to_numpy())
    an, aa, anum = pick("pc2"), pick("pa"), pick("pnum1")
    ap = np.where(isa, d["_p_2"].fill_null(0).to_numpy(), d["_p_1"].fill_null(0).to_numpy())
    d = d.with_columns(pl.Series(f"{pre}_anc_p", ap, dtype=pl.Float32),
        pl.Series(f"{pre}_anc_name", process.cpdist(d["pc2"].fill_null("").to_list(), an.tolist(), scorer=fuzz.ratio, workers=workers), dtype=pl.Float32),
        pl.Series(f"{pre}_anc_addr", process.cpdist(d["pa"].fill_null("").to_list(), aa.tolist(), scorer=fuzz.token_set_ratio, workers=workers), dtype=pl.Float32),
        pl.Series(f"{pre}_anc_num", (d["pnum1"].fill_null("").to_numpy() == anum) & (anum != ""), dtype=pl.Int8))
    out += [f"{pre}_anc_p", f"{pre}_anc_name", f"{pre}_anc_addr", f"{pre}_anc_num"]
    return d.select(["s1", "m"] + out), out


CONS_KEYS = [("pnum1", "num"), ("pstreet", "street"), ("pc2", "name"), ("pa", "addr")]


def add_consensus(d, pcol="p1", pre="k1"):
    """Probability-weighted vote share of this candidate's value among the OTHER candidates of the same S1
    (analysis/12 F2: true copies agree with each other, near-twins do not). Also the share held by S1's own number."""
    tot = pl.col(pcol).sum().over("s1")
    out = []
    for key, nm in CONS_KEYS:
        has = (pl.col(key).is_not_null() & (pl.col(key) != ""))
        d = d.with_columns(pl.when(has).then(pl.col(pcol)).otherwise(0.0).sum().over("s1").alias("_known"),
                           pl.col(pcol).sum().over(["s1", key]).alias("_same"))
        d = d.with_columns(pl.when(~has | (pl.col("_known") - pl.col(pcol) <= 1e-6)).then(-1.0)
                           .otherwise((pl.col("_same") - pl.col(pcol)) / (pl.col("_known") - pl.col(pcol))).alias(f"{pre}_vote_{nm}"))
        out.append(f"{pre}_vote_{nm}")
    # share of candidate probability mass whose first number equals S1's own first number
    d = d.with_columns(pl.when(pl.col("s1num1") != "").then(
            pl.when(pl.col("pnum1") == pl.col("s1num1")).then(pl.col(pcol)).otherwise(0.0).sum().over("s1") / tot.clip(1e-6)).otherwise(-1.0).alias(f"{pre}_s1num_share"))
    out.append(f"{pre}_s1num_share")
    return d.select(["s1", "m"] + out), out
