import polars as pl, sys, collections, random, re, pickle
sys.path.insert(0,"scripts")
from norm import *
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
s1=pl.read_parquet("work/train_s1.parquet").filter(pl.col("country")=="India").rename({"entity_id":"s1","business_name":"n1"}).select("s1","n1")
pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")]).rename({"entity_id":"m","business_name":"n2"}).select("m","n2")
df=s1.join(pairs,on="s1").join(m,on="m").filter(pl.col("n2").str.contains(r"[ऀ-෿]"))
print("indic-name pairs", df.height)
ids=df["s1"].unique().shuffle(seed=0); cut=int(len(ids)*0.8)
tr=df.filter(pl.col("s1").is_in(ids[:cut].implicit()) if False else pl.col("s1").is_in(ids[:cut].to_list())); te=df.filter(~pl.col("s1").is_in(ids[:cut].to_list()))
def toks_native(s): return [t for t in re.split(r"[\s\-,.()\[\]&/]+", base_clean(s)) if t]
cnt=collections.defaultdict(collections.Counter)
for n1,n2 in zip(tr["n1"].to_list(),tr["n2"].to_list()):
    a=name_key(n1,romanize_indic=False).split(); b=toks_native(n2)
    if len(a)==len(b):
        for x,y in zip(b,a):
            if has_indic(x): cnt[x][y]+=1
D={k:v.most_common(1)[0][0] for k,v in cnt.items() if v.most_common(1)[0][1]>=2 and v.most_common(1)[0][1]/sum(v.values())>=0.6}
print("dict size", len(D), "native tokens seen", len(cnt))
st=collections.Counter()
for n1,n2 in zip(te["n1"].to_list(),te["n2"].to_list()):
    t1=set(name_key(n1,romanize_indic=False).split())
    for x in toks_native(n2):
        if not has_indic(x): continue
        st["tok"]+=1
        r=alnum(anyascii(x))
        st["anyascii_exact"]+= r in t1
        st["anyascii_jw>=.85"]+= max([JaroWinkler.similarity(r,y) for y in t1] or [0])>=0.85
        if x in D:
            st["in_dict"]+=1; st["dict_exact"]+= D[x] in t1
    k1=name_key(n1); k_any=name_key(n2)
    k_dict=" ".join(D.get(x, alnum(anyascii(x))) for x in toks_native(n2))
    st["pairs"]+=1
    st["pair_tsr_anyascii"]+=fuzz.token_set_ratio(k1,k_any); st["pair_tsr_dict"]+=fuzz.token_set_ratio(k1,alnum(k_dict))
    st["pair_eq_dict"]+= sorted(k1.split())==sorted(alnum(k_dict).split())
print({k:(v if k in("tok","pairs") else round(v/(st['tok'] if not k.startswith('pair') else st['pairs']),4)) for k,v in st.items()})
# examples of anyascii mistakes
ex=0
for n1,n2 in zip(te["n1"].to_list()[:400],te["n2"].to_list()[:400]):
    if ex<12 and name_key(n1)!=name_key(n2): print(n1[:40],"|",name_key(n2)[:50]); ex+=1
pickle.dump(D,open("work/indic_dict_trainsplit.pkl","wb"))
