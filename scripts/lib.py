"""Shared helpers: config, paths, S3 checkpoint sync, atomic parquet writes, memory guards, logging."""
import os, sys, time, json, subprocess, threading, resource, hashlib
import yaml, psutil
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.environ.get("ER_CONFIG", os.path.join(HERE, "..", "config.yaml"))))
WORK = os.environ.get("ER_WORKDIR", CFG["workdir"])
S3 = os.environ.get("ER_S3", CFG["s3_prefix"]).rstrip("/")
USE_S3 = S3.startswith("s3://") and "CHANGE-ME" not in S3


def P(*parts):
    p = os.path.join(WORK, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(P("logs", "pipeline.log"), "a") as f:
        f.write(msg + "\n")


def done(stage):
    return os.path.exists(P("checkpoints", f"{stage}.done"))


def mark_done(stage, info=None):
    with open(P("checkpoints", f"{stage}.done"), "w") as f:
        json.dump(info or {}, f)
    s3_push(f"checkpoints/{stage}.done")


def write_parquet_atomic(df, rel):
    """Write df to WORK/rel via tmp+rename; returns path."""
    path = P(rel)
    tmp = path + ".tmp"
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def s3_push(rel, recursive=False):
    if not USE_S3:
        return
    src = P(rel) if not recursive else os.path.join(WORK, rel)
    cmd = ["aws", "s3", "sync" if recursive else "cp", src, f"{S3}/{rel}", "--only-show-errors"]
    subprocess.run(cmd, check=True)


def s3_pull(rel, recursive=False):
    if not USE_S3:
        return
    dst = os.path.join(WORK, rel)
    os.makedirs(dst if recursive else os.path.dirname(dst), exist_ok=True)
    cmd = ["aws", "s3", "sync" if recursive else "cp", f"{S3}/{rel}", dst, "--only-show-errors"]
    subprocess.run(cmd, check=False)


def limit_worker_memory(gb):
    """Linux: cap address space so a runaway worker raises MemoryError instead of driving the host into swap."""
    if sys.platform.startswith("linux") and gb:
        b = int(gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))


class MemGuard:
    """Abort the whole process tree if system available memory drops under `min_frac` of total."""
    def __init__(self, min_frac=0.2):
        self.min_frac = min_frac
        self.peak_rss = 0.0
        self.min_avail = 1e18
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        me = psutil.Process(os.getpid())
        while True:
            vm = psutil.virtual_memory()
            try:
                rss = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
            except psutil.Error:
                rss = 0
            self.peak_rss = max(self.peak_rss, rss)
            self.min_avail = min(self.min_avail, vm.available)
            if vm.available < self.min_frac * vm.total:
                log(f"MEMGUARD ABORT avail={vm.available/1e9:.1f}GB rss_tree={rss/1e9:.1f}GB")
                for c in me.children(recursive=True):
                    try: c.kill()
                    except psutil.Error: pass
                os._exit(99)
            time.sleep(0.5)

    def report(self):
        return f"peak_rss_tree={self.peak_rss/1e9:.1f}GB min_avail={self.min_avail/1e9:.1f}GB"


def s1_bucket(col, n):
    return (pl.col(col).hash(11) % n).alias("bucket")


def s1_fold(col, n=10):
    return (pl.col(col).hash(99) % n).alias("fold")


def orphan_frac():
    return float(CFG["model"].get("orphan_frac") or 0.0)


_DROP = None


def _drop_list():
    """V5: explicit dropped-S1 list (hard look-alike orphans + random fill), shipped as orphan/dropped_s1.parquet."""
    global _DROP
    if _DROP is None:
        f = P("orphan/dropped_s1.parquet")
        _DROP = pl.read_parquet(f)["s1"].to_list() if (CFG["model"].get("orphan_list") and os.path.exists(f)) else []
    return _DROP


def keep_s1_expr(col="s1"):
    """polars: True for S1 kept after the orphan simulation."""
    fr = orphan_frac()
    if fr <= 0:
        return pl.lit(True)
    if CFG["model"].get("orphan_list"):
        return ~pl.col(col).is_in(_drop_list())
    return (pl.col(col).hash(777) % 10000) >= int(fr * 10000)


def keep_s1_sql(col="s1"):
    """DuckDB twin of keep_s1_expr (DuckDB hash differs from polars hash -> use a stored list instead)."""
    fr = orphan_frac()
    if fr <= 0:
        return "true"
    return f"{col} in (select s1 from '{P('orphan/kept_s1.parquet')}')"


def write_kept_s1():
    """Materialize the kept-S1 list with the POLARS hash so DuckDB and polars agree."""
    fr = orphan_frac()
    if fr <= 0 or os.path.exists(P("orphan/kept_s1.parquet")):
        return
    s1 = pl.read_parquet(P("input", "train_s1.parquet"), columns=["entity_id"]).rename({"entity_id": "s1"})
    write_parquet_atomic(s1.filter(keep_s1_expr("s1")), "orphan/kept_s1.parquet")
    s3_push("orphan", recursive=True)
