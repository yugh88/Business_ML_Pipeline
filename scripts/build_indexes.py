"""Stage indexes: per-split token statistics from that split's S1 (reference source; no labels):
name-token df, address-token df, core2 name frequency, S1 size per country."""
import os, polars as pl
from lib import CFG, P, WORK, log, done, mark_done, write_parquet_atomic, s3_push

def atoks(col):
    return pl.col(col).str.replace_all(",", " ").str.split(" ").list.eval(
        pl.element().filter((pl.element() != "") & ~pl.element().is_in(["null", "n", "a", "na"]))).list.unique()

def main():
    if done("indexes"):
        return
    for split in CFG["splits"]:
        s1 = pl.scan_parquet(os.path.join(WORK, "normalized", f"{split}_s1", "*.parquet"))
        write_parquet_atomic(s1.select("country", pl.col("core2").str.split(" ").list.unique().alias("t")).explode("t")
                             .group_by("country", "t").len().rename({"len": "df"}).collect(), f"indexes/{split}_tok_df_name.parquet")
        write_parquet_atomic(s1.select("country", atoks("a").alias("t")).explode("t").group_by("country", "t").len()
                             .rename({"len": "df"}).collect(), f"indexes/{split}_tok_df_addr.parquet")
        write_parquet_atomic(s1.group_by("country", "core2").len().rename({"len": "name_freq_s1"}).collect(), f"indexes/{split}_name_freq.parquet")
        write_parquet_atomic(s1.group_by("country").len().rename({"len": "n_s1"}).collect(), f"indexes/{split}_n_s1.parquet")
        log("indexes", split)
    s3_push("indexes", recursive=True)
    mark_done("indexes")

if __name__ == "__main__":
    main()
