"""Collective (stage-2+) features computed from a probability column over the candidate graph.
Requires columns: s1, m, <pcol>, pc2 (pool core name), pa (pool normalized address), pns (space-free name),
pnum1 (pool first house number), pstreet (pool '<num> <word>' street key)."""
import polars as pl, numpy as np
from rapidfuzz import process, fuzz

SIB_KEYS = [("pc2", "name"), ("pa", "addr"), ("pns", "ns"), ("pnum1", "num"), ("pstreet", "street")]

def _top2(col, grp, tag):
    return [pl.col(col).max().over(grp).alias(f"_{tag}_mx"),
            pl.col(col).sort(descending=True).get(1, null_on_oob=True).over(grp).fill_null(0).alias(f"_{tag}_2")]

def _excl(col, tag):  # max over group excluding self (via top-2 trick)
    return pl.when(pl.col(col) == pl.col(f"_{tag}_mx")).then(pl.col(f"_{tag}_2")).otherwise(pl.col(f"_{tag}_mx"))

def add_collective(d, pcol, pre, workers=4):
    d = d.with_columns(*_top2(pcol, ["s1"], "s"), *_top2(pcol, ["m"], "m"),
                       pl.col(pcol).rank("ordinal", descending=True).over("s1").alias(f"{pre}_rank_s1"),
                       (pl.col(pcol) > 0.5).sum().over("s1").alias(f"{pre}_s1_n50"), pl.col(pcol).sum().over("s1").alias(f"{pre}_s1_sum"),
                       pl.len().over("m").alias(f"{pre}_m_ncand"))
    d = d.with_columns(pl.col("_s_mx").alias(f"{pre}_s1_max"), _excl(pcol, "s").alias(f"{pre}_s1_other"), _excl(pcol, "m").alias(f"{pre}_m_other"))
    out = [f"{pre}_rank_s1", f"{pre}_s1_n50", f"{pre}_s1_sum", f"{pre}_m_ncand", f"{pre}_s1_max", f"{pre}_s1_other", f"{pre}_m_other"]
    for key, nm in SIB_KEYS:
        g = ["s1", key]
        d = d.with_columns(*_top2(pcol, g, "k"), pl.len().over(g).alias(f"{pre}_sib_{nm}_n"))
        d = d.with_columns(pl.when(pl.col(key).is_null() | (pl.col(key) == "") | (pl.col(f"{pre}_sib_{nm}_n") <= 1)).then(-1.0)
                           .otherwise(_excl(pcol, "k")).alias(f"{pre}_sib_{nm}_max"))
        out += [f"{pre}_sib_{nm}_n", f"{pre}_sib_{nm}_max"]
    # anchor = highest-probability OTHER candidate of the same S1
    d = d.with_columns(pl.col(pcol).rank("ordinal", descending=True).over("s1").alias("_r"))
    anc = d.filter(pl.col("_r") <= 2).select("s1", "_r", "m", pcol, "pc2", "pa", "pnum1").rename({pcol: "_p"}) \
           .pivot(on="_r", index="s1", values=["m", "_p", "pc2", "pa", "pnum1"])
    d = d.join(anc, on="s1", how="left")
    isa = (d["m"] == d["m_1"]).fill_null(False).to_numpy()
    pick = lambda c: np.where(isa, d[f"{c}_2"].fill_null("").to_numpy(), d[f"{c}_1"].fill_null("").to_numpy())
    an, aa, anum = pick("pc2"), pick("pa"), pick("pnum1")
    ap = np.where(isa, d["_p_2"].fill_null(0).to_numpy(), d["_p_1"].fill_null(0).to_numpy())
    d = d.with_columns(pl.Series(f"{pre}_anc_p", ap, dtype=pl.Float32),
        pl.Series(f"{pre}_anc_name", process.cpdist(d["pc2"].fill_null("").to_list(), an.tolist(), scorer=fuzz.ratio, workers=workers), dtype=pl.Float32),
        pl.Series(f"{pre}_anc_addr", process.cpdist(d["pa"].fill_null("").to_list(), aa.tolist(), scorer=fuzz.token_set_ratio, workers=workers), dtype=pl.Float32),
        pl.Series(f"{pre}_anc_num", (d["pnum1"].fill_null("").to_numpy() == anum) & (anum != ""), dtype=pl.Int8))
    out += [f"{pre}_anc_p", f"{pre}_anc_name", f"{pre}_anc_addr", f"{pre}_anc_num"]
    drop = [c for c in d.columns if c.startswith("_") or c in ("m_1", "m_2", "pc2_1", "pc2_2", "pa_1", "pa_2", "pnum1_1", "pnum1_2")]
    return d.drop(drop), out
