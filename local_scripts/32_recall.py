"""Recall@k per view and for unions, on eval S1 sample.
usage: 32_recall.py view1,view2,... [ks] [exact_keys_to_union]   (files work/cand_eval/{view}_{country}.parquet)"""
import duckdb, sys, json
views = sys.argv[1].split(","); ks = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "10,20,50,100").split(",")]
extra = [e for e in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if e]
con = duckdb.connect(); con.execute("SET memory_limit='4GB'; SET threads=4")
con.execute("create temp table ev as select * from 'work/eval_s1_ids.parquet'")
NEV = con.execute("select count(*) from ev").fetchone()[0]
con.execute("create temp table tp as select k.* from 'work/true_pair_keys.parquet' k semi join ev on ev.entity_id=k.s1")
piv = ", ".join(f"min(rank) filter (where vw='{v}') as r_{i}" for i, v in enumerate(views))
u = " union all ".join(f"select s1, m, rank, '{v}' vw from 'work/cand_eval/{v}_*.parquet'" for v in views)
con.execute(f"create temp table c as select s1, m, {piv} from ({u}) group by s1, m")
con.execute("create temp table h as select tp.*, " + ", ".join(f"coalesce(c.r_{i}, 9999) r_{i}" for i in range(len(views))) +
            " from tp left join c using (s1, m)")
segs = [("true", "all"), ("country='US'", "US"), ("country='India'", "India"), ("src='S2'", "S2"), ("src='S3'", "S3"),
        ("nmatch=1", "n1"), ("nmatch>=5", "n5p"), ("name_tsr<50", "weakname"), ("a2_empty", "addr_empty"), ("dom2", "domain"),
        ("brand2", "brand"), ("indic2", "indic"), ("addr_tsr<70", "weakaddr")]
res = {}
sets = [[i] for i in range(len(views))] + ([list(range(len(views)))] if len(views) > 1 else [])
for k in ks:
    for vs in sets:
        hit = " or ".join(f"r_{i}<={k}" for i in vs) + ("".join(f" or {e}" for e in extra) if len(vs) > 1 else "")
        r = con.execute("select " + ", ".join(f"avg(({hit})::int) filter (where {e}) as {n}" for e, n in segs) + " from h").df().iloc[0].to_dict()
        cond = " or ".join(f"r_{i}<={k}" for i in vs)
        nc = con.execute(f"select count(*) from c where {cond}").fetchone()[0]
        name = "+".join(views[i] for i in vs) + (f"+keys[{','.join(extra)}]" if len(vs) > 1 and extra else "")
        r["cands_per_s1"] = nc / NEV
        res[f"{name}@{k}"] = r
        print(f"{name:52s} k={k:3d} c/s1={r['cands_per_s1']:6.1f} " + " ".join(f"{n}={r[n]:.4f}" for _, n in segs), flush=True)
json.dump(res, open(f"analysis/out/32_recall_{'_'.join(views)}{'_keys' if extra else ''}.json", "w"), indent=1, default=float)
