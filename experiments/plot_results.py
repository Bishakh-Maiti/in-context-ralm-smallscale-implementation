import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

results_path = "results/results.csv"

df = pd.read_csv(results_path)

print("Loaded results:")
print(df)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# FILTER RALM RESULTS
# ============================================================

ralm = df[
    df["system"] == "In-Context RALM"
].copy()

baseline = df[
    df["system"] == "Baseline"
].iloc[0]


# ============================================================
# PLOT 1 — ANSWER QUALITY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    ralm["top_k"],
    ralm["exact_match"] * 100,
    marker="o",
    label="Exact Match"
)

plt.plot(
    ralm["top_k"],
    ralm["f1"] * 100,
    marker="o",
    label="F1"
)

# Baseline reference lines
plt.axhline(
    baseline["exact_match"] * 100,
    linestyle="--",
    label="Baseline Exact Match"
)

plt.axhline(
    baseline["f1"] * 100,
    linestyle=":",
    label="Baseline F1"
)

plt.xlabel("Top-k Retrieved Documents")
plt.ylabel("Score (%)")

plt.title(
    "Answer Quality vs. Number of Retrieved Documents"
)

plt.xticks(
    ralm["top_k"]
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/answer_quality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2 — RETRIEVAL RECALL
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    ralm["top_k"],
    ralm["retrieval_recall"] * 100,
    marker="o"
)

plt.xlabel("Top-k Retrieved Documents")

plt.ylabel("Retrieval Recall@k (%)")

plt.title(
    "Retrieval Recall vs. Number of Retrieved Documents"
)

plt.xticks(
    ralm["top_k"]
)

plt.ylim(0, 100)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/retrieval_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\nPlots generated successfully!")

print(
    "results/answer_quality.png"
)

print(
    "results/retrieval_recall.png"
)