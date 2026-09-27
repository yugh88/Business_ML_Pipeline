"""Tune the V8 decision on fold 7 (same selection rule as V5; reduced tau/variant grid to keep it < 1 h locally)."""
import sys, os
sys.path.insert(0, "aws/scripts")
os.environ.setdefault("TUNE_VARIANTS", "p2x2,p2x2b")
os.environ.setdefault("ORPHAN_LIST", "work/v5_dropped_s1.parquet")
os.environ.setdefault("TUNE_SKIP_TEST", "1")
import tune_decision as T
T.TAUS = [float(x) for x in os.environ.get("TUNE_TAUS", "0.0001,0.02,0.05").split(",")]
T.VARIANTS_USED = os.environ["TUNE_VARIANTS"].split(",")
_orig = T.configs
T.configs = lambda: (c for c in _orig() if c["policy"] == "G_expected_f")     # G won every previous tuning (V3-V5)
sys.argv = ["tune_decision.py", sys.argv[1], "work", sys.argv[2]]
T.main()
