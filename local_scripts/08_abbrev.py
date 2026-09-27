import polars as pl, sys, collections, re, pickle
sys.path.insert(0,"scripts")
from norm import *
s1=pl.read_parquet("work/train_s1.parquet").sample(400000,seed=5).rename({"entity_id":"s1","business_address":"a1"}).select("s1","a1","country")
pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")]).rename({"entity_id":"m","business_address":"a2"}).select("m","a2")
df=s1.join(pairs,on="s1").join(m,on="m").filter(pl.col("a2")!="")
cnt=collections.Counter(); tot=collections.Counter(); comp=collections.Counter()
for c,a1,a2 in zip(df["country"].to_list(),df["a1"].to_list(),df["a2"].to_list()):
    c1=[alnum(x) for x in a1.split(",")]; c2=[alnum(x) for x in anyascii(a2).split(",")]
    for x in c1:
        for t in x.split(): tot[(c,t)]+=1
    # component-level (state/city) mapping for last component
    comp[(c,c2[-1],c1[-1])]+=1
    for x in c1:
        tx=x.split()
        for y in c2:
            ty=y.split()
            if len(tx)==len(ty) and len(tx)>=2 and sum(a==b for a,b in zip(tx,ty))==len(tx)-1:
                for a,b in zip(tx,ty):
                    if a!=b and a[0]==b[0] and len(b)<len(a) and not a.isdigit(): cnt[(c,a,b)]+=1
ab=[(k,v) for k,v in cnt.most_common(400) if v>=30]
print("learned abbreviations (full -> abbrev):", len(ab))
for (c,a,b),v in ab[:70]: print(f"{c[:2]} {a}->{b}:{v}", end="  ")
print()
# keep mapping abbrev->full only if abbreviation isn't itself a frequent full token in S1
M={}
for (c,a,b),v in ab:
    if tot[(c,b)] < 0.2*v: M.setdefault(c,{})[b]=a
print({c:len(v) for c,v in M.items()})
st=collections.defaultdict(collections.Counter)
for (c,x,y),v in comp.items(): st[(c,x)][y]+=v
S={}
for (c,x),ctr in st.items():
    y,v=ctr.most_common(1)[0]
    if v>=20 and v/sum(ctr.values())>0.6 and x!=y: S.setdefault(c,{})[x]=y
print("state/last-component maps:", {c:len(v) for c,v in S.items()}, list(S.get("US",{}).items())[:10], list(S.get("India",{}).items())[:10])
pickle.dump({"abbr":M,"state":S},open("work/addr_maps.pkl","wb"))
