"""Exact-key blocking rules: recall on ALL true pairs + exact candidate counts from key-frequency products (no pair materialization)."""
import duckdb, json
con = duckdb.connect(); con.execute("SET memory_limit='5GB'; SET threads=4; SET preserve_insertion_order=false; SET temp_directory='work/duck_tmp'")
# extended name core: strip honorifics / trailing com / id / phone tokens (candidate norm fix found in true-pair audit)
CORE2 = r"""trim(regexp_replace(regexp_replace(regexp_replace(ncore,
   '^((shri|sri|smt|mr|mrs|ms|dr|m s|messrs) )+',''), ' (id [0-9]+|[0-9]{6,}|com|c0m|in|net|org)( |$).*$',''), 'com$',''))"""
KEYS = {
  "core": "ncore",
  "core_sorted": "array_to_string(list_sort(string_split(ncore,' ')),' ')",
  "core2_sorted": f"array_to_string(list_sort(string_split({CORE2},' ')),' ')",
  "ns6": f"case when length(replace({CORE2},' ',''))>=4 then substr(replace({CORE2},' ',''),1,6) end",
  "ns8": f"case when length(replace({CORE2},' ',''))>=5 then substr(replace({CORE2},' ',''),1,8) end",
  "addr": "nullif(a,'')",
  "street": r"nullif(regexp_extract(a,'([0-9]+ [a-z]{3,})',1),'')",
  "street_state": r"case when regexp_extract(a,'([0-9]+ [a-z]{3,})',1)<>'' and state<>'' then regexp_extract(a,'([0-9]+ [a-z]{3,})',1)||'|'||state end",
  "num1_state": r"case when nums<>'' and state<>'' then split_part(nums,' ',1)||'|'||state end",
  "tok1_state": "case when state<>'' then split_part(ncore,' ',1)||'|'||state end",
}
sel = ", ".join(f"{v} as {k}" for k, v in KEYS.items())
con.execute(f"create temp table s1k as select entity_id, country, {sel} from 'work/train_s1_norm.parquet'")
con.execute(f"""create temp table pk as select entity_id, country, {sel} from
  (select * from 'work/train_s2_norm.parquet' union all select * from 'work/train_s3_norm.parquet')""")
con.execute("""create temp table pr as select p.s1, p.m, p.src, s.country, pc.n nmatch from 'work/train_pairs.parquet' p
   join s1k s on s.entity_id=p.s1 join 'work/train_per_s1.parquet' pc on pc.s1=p.s1""")
eqs = ", ".join(f"coalesce(a.{k}=b.{k},false) as {k}" for k in KEYS)
con.execute(f"create temp table pe as select pr.*, t.name_tsr, t.addr_tsr, t.a2_empty, t.dom2, t.indic2, t.brand2, {eqs} from pr join s1k a on a.entity_id=pr.s1 join pk b on b.entity_id=pr.m join 'work/true_pair_sims.parquet' t on t.m=pr.m")
con.execute("copy pe to 'work/true_pair_keys.parquet'")
ALLP = con.execute("select sum(a.n*b.n) from (select country,count(*) n from s1k group by 1) a join (select country,count(*) n from pk group by 1) b using(country)").fetchone()[0]
out = {}
for k in KEYS:
    cand = con.execute(f"""select sum(a.n*b.n), sum(a.n*b.n) filter (where a.country='US'), sum(a.n*b.n) filter (where a.country='India'),
        max(b.n) from (select country,{k} kk,count(*) n from s1k where {k} is not null group by all) a
        join (select country,{k} kk,count(*) n from pk where {k} is not null group by all) b using(country,kk)""").fetchone()
    rec = con.execute(f"""select avg({k}::int), avg({k}::int) filter (where country='US'), avg({k}::int) filter (where country='India'),
        avg({k}::int) filter (where src='S2'), avg({k}::int) filter (where src='S3'),
        avg({k}::int) filter (where name_tsr<50), avg({k}::int) filter (where a2_empty), avg({k}::int) filter (where dom2), avg({k}::int) filter (where brand2)
        from pe""").fetchone()
    out[k] = dict(cands=cand[0], cands_US=cand[1], cands_India=cand[2], max_block=cand[3], per_s1=cand[0]/2206821, RR=1-cand[0]/ALLP,
                  recall=rec[0], rec_US=rec[1], rec_India=rec[2], rec_S2=rec[3], rec_S3=rec[4], rec_weakname=rec[5], rec_addr_empty=rec[6], rec_domain=rec[7], rec_brand=rec[8])
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in out[k].items()}, flush=True)
json.dump(out, open("analysis/out/30_block_keys.json", "w"), indent=1, default=float)
