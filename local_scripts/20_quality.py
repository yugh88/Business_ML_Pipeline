"""Phase 2 gap-fill: normalized duplication/ambiguity + missingness per split/source/country. DuckDB, one file at a time."""
import duckdb, json
con = duckdb.connect(); con.execute("SET memory_limit='4GB'; SET threads=4")
res = {}
for split in ["train", "test"]:
    for s in [1, 2, 3]:
        f = f"work/{split}_s{s}_norm.parquet"
        q = f"""
        with t as (select country, n, ncore, a, nums, state, f_domain, f_indic, f_brand, f_junk,
                          count(*) over (partition by country, ncore) c_core,
                          count(*) over (partition by country, ncore, a) c_core_a,
                          count(*) over (partition by country, a) c_a
                   from '{f}')
        select country, count(*) n_rows,
          avg((a='')::int) addr_empty, avg((nums='')::int) no_numbers, avg((state='')::int) no_state,
          avg(f_domain::int) is_domain, avg(f_indic::int) indic_name, avg(f_brand::int) brand, avg(f_junk::int) junk,
          avg((ncore='')::int) core_empty,
          count(distinct ncore) distinct_core,
          avg((c_core>1)::int) core_shared, avg((c_core>10)::int) core_shared_gt10,
          avg((c_core_a>1)::int) core_addr_shared,
          avg(case when a<>'' then (c_a>1)::int end) addr_shared,
          avg(length(n)) name_len, avg(len(string_split(n,' '))) name_toks, avg(len(string_split(a,' '))) addr_toks
        from t group by country order by country"""
        df = con.execute(q).df()
        res[f"{split}_s{s}"] = df.to_dict(orient="records")
        print(split, s); print(df.round(4).to_string(index=False), flush=True)
json.dump(res, open("analysis/out/20_quality.json", "w"), indent=1, default=float)
