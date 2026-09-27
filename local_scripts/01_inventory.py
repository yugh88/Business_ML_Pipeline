import polars as pl, unicodedata, re, json, collections, random
out=open("work/01_inventory.txt","w")
def P(*a): print(*a,file=out,flush=True)
def script_of(ch):
    if ch.isascii(): return "ASCII"
    n=unicodedata.name(ch,"UNK").split()[0]
    return n
for split in ["train","test"]:
  for s in [1,2,3]:
    df=pl.read_parquet(f"work/{split}_s{s}.parquet")
    P(f"\n######## {split} S{s} rows={df.height}")
    P("dup entity_id:", df.height-df["entity_id"].n_unique())
    P("id prefix ok:", df.filter(~pl.col("entity_id").str.contains(f"^S{s}-\\d+$")).height, "bad")
    ids=df["entity_id"].str.slice(3).cast(pl.Int64)
    P("id numeric range:", ids.min(), ids.max(), "digits len dist:", df["entity_id"].str.len_chars().value_counts().sort("count",descending=True).head(5).rows())
    P("dup full record (name,addr,country):", df.height-df.select(["business_name","business_address","country"]).n_unique())
    for c in ["business_name","business_address"]:
        col=df[c]
        P(f"-- {c}: unique={col.n_unique()} dup_rows={df.height-col.n_unique()} empty={(col=='').sum()}")
        L=col.str.len_chars()
        P("   len quantiles:",[int(L.quantile(q)) for q in [0,.01,.05,.25,.5,.75,.95,.99]], "max", L.max())
        P("   leading/trailing ws:", col.str.contains(r"^\s|\s$").sum(), " double space:", col.str.contains(r"  ").sum(),
          " nbsp/odd ws:", col.str.contains("[  -​　\t]").sum(), " newline:",col.str.contains("\n").sum())
        P("   non-ascii rows:", col.str.contains(r"[^\x00-\x7f]").sum(), " devanagari:", col.str.contains(r"[ऀ-ॿ]").sum(),
          " latin-accent:", col.str.contains(r"[À-ɏ]").sum(), " other-script:", col.str.contains(r"[^\x00-\x7fऀ-ॿÀ-ɏ]").sum())
        P("   all-upper:", col.str.contains(r"^[^a-z]*[A-Z][^a-z]*$").sum(), " all-lower:", col.str.contains(r"^[^A-Z]*[a-z][^A-Z]*$").sum(),
          " starts non-alnum:", col.str.contains(r"^[^\w]").sum(), " digits-only:", col.str.contains(r"^[\d\s\-/]+$").sum(),
          " has url/.com:", col.str.contains(r"(?i)\.(com|net|org|in|fr|co)\b|www\.|https?:").sum(), " has email@:", col.str.contains("@").sum(),
          " has quote:", col.str.contains('"').sum())
        vc=col.value_counts().sort("count",descending=True).head(12)
        P("   top values:", [(v[:60],n) for v,n in vc.rows()])
        # leading junk prefixes
        pre=col.str.extract(r"^([^\w\s]+)\s",1).drop_nulls().value_counts().sort("count",descending=True).head(15)
        P("   leading symbol prefixes:", pre.rows())
        suf=col.str.extract(r"\s([^\w\s]+)$",1).drop_nulls().value_counts().sort("count",descending=True).head(15)
        P("   trailing symbol suffixes:", suf.rows())
        # char script distribution on a sample
        samp=col.sample(min(200000,df.height),seed=0).to_list()
        cc=collections.Counter()
        for x in samp:
            for ch in x:
                if not ch.isascii(): cc[script_of(ch)]+=1
        P("   non-ascii char scripts (200k sample):", cc.most_common(12))
        nc=collections.Counter(ch for x in samp for ch in x if not ch.isalnum() and not ch.isspace())
        P("   punctuation chars (sample):", nc.most_common(25))
    P("-- country:", df["country"].value_counts().sort("count",descending=True).rows())
    for ctry in df["country"].unique().to_list():
        d=df.filter(pl.col("country")==ctry)
        P(f"   {ctry}: n={d.height} name_empty={(d['business_name']=='').sum()} addr_empty={(d['business_address']=='').sum()} both_empty={((d['business_name']=='')&(d['business_address']=='')).sum()} name_devanagari={d['business_name'].str.contains(r'[ऀ-ॿ]').sum()} addr_devanagari={d['business_address'].str.contains(r'[ऀ-ॿ]').sum()} name_accent={d['business_name'].str.contains(r'[À-ɏ]').sum()}")
    P("dup name+addr:", df.height-df.select(["business_name","business_address"]).n_unique())
    P("dup name+addr (casefold,ws-collapsed):", df.height-df.select(pl.col("business_name").str.to_lowercase().str.replace_all(r"\s+"," ").str.strip_chars(), pl.col("business_address").str.to_lowercase().str.replace_all(r"\s+"," ").str.strip_chars()).n_unique())
# cross-split id overlap
for s in [1,2,3]:
    a=set(pl.read_parquet(f"work/train_s{s}.parquet",columns=["entity_id"])["entity_id"].to_list())
    b=set(pl.read_parquet(f"work/test_s{s}.parquet",columns=["entity_id"])["entity_id"].to_list())
    P(f"S{s} train/test id overlap:", len(a&b))
# cross-split exact record overlap
for s in [1,2,3]:
    a=pl.read_parquet(f"work/train_s{s}.parquet").select(["business_name","business_address"]).unique()
    b=pl.read_parquet(f"work/test_s{s}.parquet").select(["business_name","business_address"]).unique()
    P(f"S{s} train/test exact name+addr overlap:", a.join(b,on=["business_name","business_address"]).height, " name-only overlap:", a.select("business_name").unique().join(b.select("business_name").unique(),on="business_name").height)
