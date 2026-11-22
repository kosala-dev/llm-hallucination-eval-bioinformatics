"""Compute accuracy, hallucination rate, abstention, paired A-vs-B comparison, error analysis and a chart.
Reads every scores_dev.csv / scores_test.csv (the stability file is excluded)."""
import glob, json, math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [f for f in glob.glob("scores_*.csv") if "stability" not in f]
if not files:
    raise SystemExit("No scores_*.csv found. Run run_eval.py first.")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
for c in ["score", "abstain", "halluc"]:
    if df[c].isna().any():
        raise SystemExit(f"Column '{c}' still has empty cells. Finish manual scoring first.")
df[["score", "abstain", "halluc"]] = df[["score", "abstain", "halluc"]].astype(int)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def summarize(d):
    n, k = len(d), int(d.halluc.sum())
    lo, hi = wilson(k, n)
    nontrap, trap = d[d.type != "trap"], d[d.type == "trap"]
    return {"n": n,
            "accuracy": round(float(d.score.mean()) / 2, 3),
            "halluc": f"{k}/{n}", "halluc_rate": round(k / n, 3),
            "halluc_95ci": [round(lo, 3), round(hi, 3)],
            "abstain_rate": round(float(d.abstain.mean()), 3),
            "over_abstain": f"{int(nontrap.abstain.sum())}/{len(nontrap)}",
            "trap_score": round(float(trap.score.mean()) / 2, 3) if len(trap) else None}


summary = {}
for split in ["dev", "test", "all"]:
    sub = df if split == "all" else df[df.split == split]
    for cfg in sorted(sub.config.unique()):
        summary[f"{split}|{cfg}"] = summarize(sub[sub.config == cfg])

print("\n=== Results ===")
print(pd.DataFrame(summary).T.to_string())

# Paired comparison A vs B on the held-out test split
t = df[df.split == "test"].pivot(index="id", columns="config", values="halluc")
if {"A", "B"} <= set(t.columns):
    b = int(((t.A == 1) & (t.B == 0)).sum())
    c = int(((t.A == 0) & (t.B == 1)).sum())
    p = sign_test(b, c)
    summary["paired_test"] = {"A_only_halluc": b, "B_only_halluc": c, "sign_test_p": round(p, 3)}
    print(f"\nTest split, paired hallucinations: A-only={b}, B-only={c}, exact sign-test p={p:.3f}")
    print("(with ~20 questions, expect a large p-value unless the effect is big; say so in the README)")

# Error analysis
wrong = df[(df.score < 2) | (df.halluc == 1)].copy()
wrong.to_csv("error_analysis.csv", index=False)
print(f"\n=== Error analysis: {len(wrong)} problem answers ===")
if len(wrong) < 10:
    print("WARNING: fewer than 10 wrong answers. Pool dev and test, or add questions.")
if wrong.error_category.isna().any() or (wrong.error_category.astype(str).str.strip() == "").any():
    print("WARNING: some wrong answers have no error_category. Tag them in your scores files.")
print(wrong.groupby(["config", "error_category"]).size().to_string())

# Chart (held-out test split)
rows = [(cfg, summary[f"test|{cfg}"]) for cfg in sorted(df[df.split == "test"].config.unique())]
if rows:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(rows))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [r[1]["accuracy"] for r in rows], w, label="Accuracy")
    ax.bar([i + w / 2 for i in x], [r[1]["halluc_rate"] for r in rows], w, label="Hallucination rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Config {r[0]}" for r in rows])
    ax.set_ylim(0, 1)
    ax.set_title("Llama 3.1 8B, held-out test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results_chart.png", dpi=150)

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved results_summary.json, results_chart.png, error_analysis.csv")
