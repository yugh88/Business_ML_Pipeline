import polars as pl, sys, collections, pickle
sys.path.insert(0,"scripts")
from norm import *
D=pickle.load(open("work/indic_dict.pkl","rb"))
s1=pl.read_parquet("work/train_s1.parquet").rename({"entity_id":"s1","business_address":"a1"}).select("s1","a1","country")
pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")]).rename({"entity_id":"m","business_address":"a2"}).select("m","a2")
last1=s1.with_columns(pl.col("a1").str.split(",").list.last().str.strip_chars().str.to_lowercase().alias("l1"))
top=last1.group_by("country","l1").len().sort("len",descending=True)
states={c:set(top.filter((pl.col("country")==c)&(pl.col("len")>=300))["l1"].to_list()) for c in ["US","India"]}
print({c:len(v) for c,v in states.items()}, sorted(states["India"])[:40])
df=last1.filter(pl.col("l1").is_in(list(states["US"]|states["India"]))).join(pairs,on="s1").join(m,on="m").filter(pl.col("a2")!="")
cnt=collections.defaultdict(collections.Counter)
for c,l1,a2 in zip(df["country"].to_list(),df["l1"].to_list(),df["a2"].to_list()):
    for comp in a2.split(","):
        comp=comp.strip(); k=D["addr"].get(comp, comp)
        k=alnum(anyascii(k))
        if k: cnt[(c,k)][l1]+=1
S={}
for (c,k),ctr in cnt.items():
    y,v=ctr.most_common(1)[0]
    if v>=200 and v/sum(ctr.values())>0.9: S.setdefault(c,{})[k]=y
print({c:len(v) for c,v in S.items()})
print("US sample", [(k,v) for k,v in S["US"].items() if k!=v][:15])
print("India sample", [(k,v) for k,v in S["India"].items() if k!=v][:25])
pickle.dump(S,open("work/state_map.pkl","wb"))
