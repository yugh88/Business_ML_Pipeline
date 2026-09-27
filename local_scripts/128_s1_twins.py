"""Do twin ENTITIES (same name, same street, house number +1..9) exist inside S1 itself? train vs test."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
for sp in ["test"]:
    s = pl.read_parquet(f"work/{sp}_s1_norm.parquet", columns=["entity_id", "country", "ncore", "a", "nums"])
    s = s.with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("h"))
    s = s.with_columns(pl.col("h").cast(pl.Int64, strict=False).alias("hi"),
                       pl.col("a").str.replace(r"^\D*?\d+\S*\s*", "").alias("street_key"))   # address minus first number token
    s = s.filter(pl.col("hi").is_not_null())
    g = s.select("entity_id", "country", "ncore", "street_key", "hi")
    j = g.join(g, on=["country", "ncore", "street_key"], suffix="_b").filter((pl.col("hi_b") - pl.col("hi") >= 1) & (pl.col("hi_b") - pl.col("hi") <= 9))
    j2 = j
    print(sp, "S1 with numbers:", g.height, "| twin S1 pairs same name+street(+1..9):", j.height, f"({1000*j.height/g.height:.2f}/1k)",
          "| same name any street +1..9:", j2.height, f"({1000*j2.height/g.height:.2f}/1k)")
    print("   by country:", j.group_by("country").len().rows())
