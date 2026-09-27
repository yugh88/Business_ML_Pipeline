import polars as pl, sys, time
from multiprocessing import Pool
sys.path.insert(0,"scripts")
from norm2 import name_norm, name_core, addr_norm, is_brand_token
import re
JUNK=re.compile(r"^(?:>>|\*\*\*|--|\.\.\.|<<)")
def work(rows):
    out=[]
    for n,a,c in rows:
        nn,dom,ind=name_norm(n); core=name_core(nn)
        an,nums,st=addr_norm(a,c)
        brand=any(is_brand_token(t) for t in nn.split())
        out.append((nn,core,dom,ind,brand,bool(JUNK.match(n)),an,nums,st))
    return out
if __name__=="__main__":
    for split in ["train","test"]:
        for s in [1,2,3]:
            t=time.time()
            df=pl.read_parquet(f"work/{split}_s{s}.parquet")
            rows=list(zip(df["business_name"].to_list(),df["business_address"].to_list(),df["country"].to_list()))
            chunks=[rows[i:i+50000] for i in range(0,len(rows),50000)]
            with Pool(10) as p: res=[r for ch in p.map(work,chunks) for r in ch]
            cols=list(zip(*res))
            out=df.select("entity_id","country").with_columns(
                pl.Series("n",cols[0]),pl.Series("ncore",cols[1]),pl.Series("f_domain",cols[2]),pl.Series("f_indic",cols[3]),
                pl.Series("f_brand",cols[4]),pl.Series("f_junk",cols[5]),pl.Series("a",cols[6]),pl.Series("nums",cols[7]),pl.Series("state",cols[8]))
            out.write_parquet(f"work/{split}_s{s}_norm.parquet")
            print(split,s,out.height,f"{time.time()-t:.0f}s",flush=True)
