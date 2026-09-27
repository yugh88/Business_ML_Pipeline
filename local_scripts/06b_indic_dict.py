import polars as pl, sys, collections, re, pickle
sys.path.insert(0,"scripts")
from norm import *
# Full-train dictionary for names AND address tokens (native -> latin), learned from aligned true pairs.
s1=pl.read_parquet("work/train_s1.parquet").rename({"entity_id":"s1","business_name":"n1","business_address":"a1"})
pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")]).rename({"entity_id":"m","business_name":"n2","business_address":"a2"}).drop("country")
df=s1.join(pairs,on="s1").join(m,on="m").filter(pl.col("n2").str.contains(r"[ऀ-෿]")|pl.col("a2").str.contains(r"[ऀ-෿]"))
def toks_native(s): return [t for t in re.split(r"[\s\-,.()\[\]&/]+", base_clean(s)) if t]
cnt=collections.defaultdict(collections.Counter)
for n1,n2 in zip(df["n1"].to_list(),df["n2"].to_list()):
    a=name_key(n1,romanize_indic=False).split(); b=toks_native(n2)
    if len(a)==len(b) and has_indic(n2):
        for x,y in zip(b,a):
            if has_indic(x): cnt[x][y]+=1
# address: native component vs S1 components -- align last component (state) and any native component by best char match after anyascii
from rapidfuzz import process, fuzz
acnt=collections.defaultdict(collections.Counter)
for a1,a2 in zip(df["a1"].to_list(),df["a2"].to_list()):
    if not has_indic(a2): continue
    c1=[c.strip() for c in a1.split(",")]
    for c in a2.split(","):
        c=c.strip()
        if has_indic(c):
            best=process.extractOne(alnum(anyascii(c)),[alnum(x) for x in c1],scorer=fuzz.ratio)
            if best and best[1]>=50: acnt[c][best[0]]+=1
D={k:v.most_common(1)[0][0] for k,v in cnt.items() if v.most_common(1)[0][1]>=2 and v.most_common(1)[0][1]/sum(v.values())>=0.5}
A={k:v.most_common(1)[0][0] for k,v in acnt.items() if v.most_common(1)[0][1]>=3 and v.most_common(1)[0][1]/sum(v.values())>=0.5}
print("name dict",len(D),"addr-component dict",len(A))
print("addr dict sample", list(A.items())[:12])
pickle.dump({"name":D,"addr":A},open("work/indic_dict.pkl","wb"))
# coverage on test
for split in ["train","test"]:
  for s in [2,3]:
    x=pl.read_parquet(f"work/{split}_s{s}.parquet",columns=["business_name","business_address"])
    nt=collections.Counter(t for n in x.filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))["business_name"].to_list() for t in toks_native(n) if has_indic(t))
    ac=collections.Counter(c.strip() for a in x.filter(pl.col("business_address").str.contains(r"[ऀ-෿]"))["business_address"].to_list() for c in a.split(",") if has_indic(c))
    cov=sum(v for k,v in nt.items() if k in D)/sum(nt.values()); acov=sum(v for k,v in ac.items() if k in A)/sum(ac.values())
    print(split,s,"native name tokens: distinct",len(nt),"token-coverage",round(cov,4)," native addr comps distinct",len(ac),"coverage",round(acov,4))
    if split=="test": print("   uncovered test name tokens top:", [(k,v) for k,v in nt.most_common() if k not in D][:15])
