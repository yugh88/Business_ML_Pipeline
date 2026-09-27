"""236 refined: the TRUE-copy sharing baseline depends on how many other accepted copies the S1 has (set size k). For each class and
k-bucket, s_true = sharing rate of the same class's high-confidence French pairs (p3 >= 0.99) in S1 with the same k; s_false from
train (identical 0.036 etc.). Then per-class optimal France threshold and the value of a class-specific policy."""
import sys
exec(open("scripts/236_fr_threshold.py").read().split("rows = []")[0])
x = x.with_columns(pl.col("k").clip(1, 6).alias("kb"))
hi = x.filter(pl.col("p3") >= 0.99).group_by("cls", "kb").agg(pl.col("shared").mean().alias("s_true_k"), pl.len().alias("n_hi"))
x = x.join(hi, on=["cls", "kb"], how="left")
print("TRUE-copy sharing baseline by set size (identical@same, p3>=0.99):", hi.filter(pl.col("cls") == "identical@same").sort("kb").select("kb", pl.col("s_true_k").round(3), "n_hi").rows())
best_policy, total = {}, 0.0
for cls, (sf, _) in BASE.items():
    res = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        z = x.filter((pl.col("cls") == cls) & (pl.col("p3") < t) & pl.col("s_true_k").is_not_null() & (pl.col("s_true_k") - sf >= 0.05))   # informative strata only
        if z.height == 0: res.append((t, 0, 0.0, 0.0)); continue
        g = z.group_by("kb").agg(pl.col("shared").mean().alias("sh"), pl.col("s_true_k").first().alias("st"), pl.len().alias("n"),
                                 pl.col("gain_fp").sum().alias("G"), pl.col("loss_tp").sum().alias("L"))
        g = g.with_columns(((pl.col("sh") - sf) / (pl.col("st") - sf).clip(1e-3)).clip(0, 1).alias("pt"))
        val = float(g.select(((1 - pl.col("pt")) * pl.col("G") - pl.col("pt") * pl.col("L")).sum()).item())
        ptrue = float((g["pt"] * g["n"]).sum() / g["n"].sum())
        res.append((t, z.height, ptrue, val))
    bt = max(res, key=lambda r: r[3])
    best_policy[cls] = bt[0] if bt[3] > 0 else None
    total += max(0.0, bt[3])
    print(f"{cls:26s}: " + " | ".join(f"t<{r[0]}: n={r[1]} true~{r[2]:.2f} val/1k {r[3] / NS:+.2f}" for r in res) + f"  -> best t {best_policy[cls]}")
print(f"CLASS-SPECIFIC FRANCE POLICY {best_policy}: estimated France F {total / (NS * 1000):+.5f}  LB {total / 1732544:+.5f}")
import json; json.dump(best_policy, open("analysis/out/237_fr_policy.json", "w"))
