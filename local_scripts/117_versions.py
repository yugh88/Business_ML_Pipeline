"""Generator forensics on TRAIN: house-number 'versions' of true copies per (S1, source), and relation of twins' numbers."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
pairs = pl.read_parquet("work/train_pairs.parquet")                                   # s1, m, src (true pairs)
s1s = pl.read_parquet("work/train_s1_norm.parquet", columns=["entity_id", "country", "nums", "a"])
s1s = s1s.filter(pl.col("entity_id").hash(7) % 8 == 0)                                  # 1/8 of S1 for speed (exact counts not needed)
pool = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "nums", "a") for s in (2, 3)])
p = pairs.join(s1s.select(pl.col("entity_id").alias("s1"), pl.col("nums").alias("n1"), "country"), on="s1")
p = p.join(pool.collect().rename({"entity_id": "m", "nums": "n2", "a": "a2"}), on="m")
hn = lambda c: pl.col(c).str.split(" ").list.first().fill_null("")
p = p.with_columns(hn("n1").alias("h1"), hn("n2").alias("h2"))
p = p.with_columns(pl.when(pl.col("h2") == "").then(pl.lit("empty")).when(pl.col("h2") == pl.col("h1")).then(pl.lit("same")).otherwise(pl.lit("diff")).alias("hk"))
print("true copies: house number vs S1 (share):", p.group_by("hk").len().with_columns((pl.col("len") / pl.col("len").sum()).round(4)).rows())
# versions per (s1, src)
v = p.filter(pl.col("h2") != "").group_by("s1", "src").agg(pl.col("h2").n_unique().alias("nver"), pl.len().alias("ncop"), (pl.col("hk") == "diff").sum().alias("ndiff"),
                                                             pl.col("h2").filter(pl.col("hk") == "diff").n_unique().alias("ndiffver"))
print("\n(S1,source) groups with >=2 numbered copies: distribution of #distinct numbers")
print(v.filter(pl.col("ncop") >= 2).group_by("nver").len().sort("nver").with_columns((pl.col("len") / pl.col("len").sum()).round(4)).rows())
print("\namong groups with >=2 numbered copies AND >=1 changed number: #distinct changed numbers")
print(v.filter((pl.col("ncop") >= 2) & (pl.col("ndiff") >= 1)).group_by("ndiffver").len().sort("ndiffver").with_columns((pl.col("len") / pl.col("len").sum()).round(4)).rows())
print("\nchanged-number copies: share that are the ONLY changed copy in their (S1,source) with >=2 numbered copies:")
w = p.filter(pl.col("hk") == "diff").join(v, on=["s1", "src"])
w2 = w.filter(pl.col("ncop") >= 2)
print("  same changed number shared by >=2 copies of that source:", (w2.group_by("s1", "src", "h2").len().filter(pl.col("len") >= 2)["len"].sum() / w2.height))
# cross-source: does S2's changed number ever equal S3's changed number?
cs = p.filter(pl.col("hk") == "diff").group_by("s1", "h2").agg(pl.col("src").n_unique().alias("nsrc"))
print("  changed numbers appearing in BOTH S2 and S3 (per s1,number):", cs.group_by("nsrc").len().rows())
# per source: share of copies with changed number
print("\nshare of numbered copies with changed number, by source & country:")
print(p.filter(pl.col("h2") != "").group_by("country", "src").agg((pl.col("hk") == "diff").mean().round(4)).sort("country", "src").rows())
# kinds of change
def kind(a, b):
    if a == "" or b == "": return "empty"
    if a == b: return "same"
    if b.lstrip("0") == a.lstrip("0"): return "zero_pad"
    if b.startswith(a) or a.startswith(b): return "prefix/extend"
    if b.endswith(a) or a.endswith(b): return "suffix/trunc_first"
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            pos = "first" if d[0] == 0 else ("last" if d[0] == len(a) - 1 else "mid")
            try: off = abs(int(b) - int(a))
            except: off = -1
            return f"1digit_{pos}" + ("_off1" if off == 1 else ("_off<=5" if 0 < off <= 5 else ""))
        if len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]: return "transpose"
    try:
        off = abs(int(b) - int(a)); return "numeric_off<=10" if off <= 10 else ("numeric_off<=100" if off <= 100 else "other")
    except: return "other_alnum"
q = p.filter(pl.col("hk") == "diff").sample(n=min(200000, p.filter(pl.col("hk") == "diff").height), seed=1)
q = q.with_columns(pl.Series("kind", [kind(a, b) for a, b in zip(q["h1"].to_list(), q["h2"].to_list())]))
print("\nTRUE copies: kind of house-number change:")
print(q.group_by("kind").len().sort("len", descending=True).with_columns((pl.col("len") / pl.col("len").sum()).round(4)).rows())
q.select("s1", "m", "src", "h1", "h2", "kind").write_parquet("work/true_numchange_kinds.parquet")
