"""France: same-address candidates whose name differs only by frequent tokens (196 'stop-collapse'). Per difference token:
count, V9 acceptance, and the label-free address FINGERPRINT (raw address equals another strong accepted same-source
candidate's raw address: copies ~20%, sibling entities ~1%) for accepted vs rejected pairs. Baselines: French identical-name
same-number (copies) and holding/participations (siblings). Tells whether V9's French rejections are right."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)
x = pl.read_parquet("work/copy_channel_test_France.parquet", columns=["s1", "m", "p3", "acc", "cls"])
ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
nm = pl.concat([pl.scan_parquet(f"work/test_s{s}_norm.parquet").select("entity_id", "ncore") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
raw = pl.concat([pl.scan_parquet(f"work/test_s{s}.parquet").select("entity_id", pl.col("business_address").fill_null("").str.to_lowercase().alias("ra"), "business_name")
                 for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
x = x.join(nm.rename({"entity_id": "s1", "ncore": "c1"}), on="s1").join(nm.rename({"entity_id": "m", "ncore": "c2"}), on="m") \
     .join(raw.select(pl.col("entity_id").alias("m"), "ra", pl.col("business_name").alias("n2")), on="m") \
     .join(raw.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1")), on="s1")
a, b = pl.col("c1").str.split(" "), pl.col("c2").str.split(" ")
x = x.with_columns(b.list.set_difference(a).list.sort().alias("E"), a.list.set_difference(b).list.sort().alias("M"), pl.col("m").str.slice(0, 2).alias("src"))
st = ((pl.col("p3") >= 0.9) & pl.col("acc")).cast(pl.Int32)
x = x.with_columns(st.alias("_st")).with_columns(((pl.col("_st").sum().over(["s1", "src", "ra"]) - pl.col("_st")) > 0 & (pl.col("ra") != "")).alias("fp_shared"),
                                                (pl.col("acc").cast(pl.Int32).sum().over("s1") - pl.col("acc").cast(pl.Int32)).alias("other_acc"))
x = x.with_columns(pl.when(pl.col("M").list.len() == 0).then(pl.lit("+") + pl.col("E").list.join(" "))
                     .when(pl.col("E").list.len() == 0).then(pl.lit("-") + pl.col("M").list.join(" "))
                     .otherwise(pl.col("M").list.join(" ") + pl.lit(" -> ") + pl.col("E").list.join(" ")).alias("diff"))
same = pl.col("cls").str.ends_with("@same")
base_copy = float(x.filter(same & (pl.col("cls") == "identical@same") & pl.col("acc") & (pl.col("ra") != ""))["fp_shared"].mean())
sib = x.filter(same & pl.col("diff").is_in(["+holding", "+participations", "+international", "+internationale", "+distribution", "+snc"]) & (pl.col("ra") != ""))
print(f"BASELINES fingerprint-shared: French identical-name same-address accepted = {base_copy:.3f};  sibling words (+holding/+participations/…) = "
      f"{sib['fp_shared'].mean():.3f} (n={sib.height}, V9 acc {sib['acc'].mean():.3f})")
t = x.filter(same & (pl.col("cls") != "identical@same") & (pl.col("ra") != "")).group_by("diff").agg(
    pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc"), pl.col("p3").mean().round(3).alias("p3"),
    pl.col("fp_shared").filter(pl.col("acc")).mean().round(3).alias("shared_acc"), pl.col("fp_shared").filter(~pl.col("acc")).mean().round(3).alias("shared_rej"),
    (~pl.col("acc")).sum().alias("n_rej"), (pl.col("other_acc").filter(~pl.col("acc")) > 0).mean().round(3).alias("rej_S1_has_other_acc"))
print(t.sort("n_rej", descending=True).head(45))
x.write_parquet("work/fr_residual.parquet")
for d in ("+s", "+sci", "-sci", "+e r u", "+ei", "+developpement", "+groupe", "+france"):
    z = x.filter(same & (pl.col("diff") == d) & ~pl.col("acc")).head(4)
    for r in z.select("n1", "n2", "p3", "other_acc").rows():
        print(f"  [{d}] rejected p3={r[2]:.2f} other_acc={r[3]}:  {r[0]!r}  vs  {r[1]!r}")
