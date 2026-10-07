# ============================================================
# analyze_results.py
#
# Complete paired statistical analysis for the
# context allocation experiment.
#
# Comparisons:
#   Uniform vs Weighted
#   Uniform vs Top-heavy
#   Weighted vs Top-heavy
#
# For k = 3, 5, 7
#
# Tests:
#   - Wilcoxon signed-rank test for F1
#   - Bootstrap 95% CI for paired F1 difference
#   - Cohen's dz
#   - Exact McNemar test for Exact Match
#   - Holm correction across all comparisons
#
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from scipy.stats import binomtest


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "context_allocation_detailed_1000.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "statistical_tests.csv"
)

N_BOOTSTRAP = 5000

RANDOM_SEED = 42


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(p_values)

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(m)

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (
            p_values[index]
            * (m - rank)
        )

        running_max = max(
            running_max,
            value
        )

        adjusted[index] = min(
            running_max,
            1.0
        )

    return adjusted


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_ci(
    differences,
    n_bootstrap=5000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    differences = np.asarray(
        differences,
        dtype=float
    )

    n = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    return lower, upper


# ============================================================
# COHEN'S DZ
# ============================================================

def cohens_dz(
    differences
):

    differences = np.asarray(
        differences,
        dtype=float
    )

    sd = np.std(
        differences,
        ddof=1
    )

    if sd == 0:
        return np.nan

    return (
        np.mean(differences)
        / sd
    )


# ============================================================
# EXACT MCNEMAR
# ============================================================

def exact_mcnemar(
    first,
    second
):

    first = np.asarray(
        first,
        dtype=int
    )

    second = np.asarray(
        second,
        dtype=int
    )

    # first = 1, second = 0
    b = np.sum(
        (first == 1)
        & (second == 0)
    )

    # first = 0, second = 1
    c = np.sum(
        (first == 0)
        & (second == 1)
    )

    discordant = b + c

    if discordant == 0:

        p_value = 1.0

    else:

        result = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        )

        p_value = result.pvalue

    return (
        b,
        c,
        p_value
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DETAILED RESULTS")
print("=" * 70)

if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nRows loaded: {len(df):,}"
)

print(
    f"Unique questions: "
    f"{df['question_id'].nunique():,}"
)


# ============================================================
# VALIDATE DATA
# ============================================================

expected_rows = (
    1000
    * 4
    * 3
)

if len(df) != expected_rows:

    raise ValueError(
        f"\nExpected {expected_rows:,} rows "
        f"but found {len(df):,}."
    )


expected_allocations = {
    "uniform",
    "weighted",
    "top_heavy"
}

actual_allocations = set(
    df["allocation"].unique()
)

if actual_allocations != expected_allocations:

    raise ValueError(
        "\nUnexpected allocation strategies:\n"
        f"{actual_allocations}"
    )


print(
    "\nData validation passed."
)


# ============================================================
# COMPARISONS
# ============================================================

comparisons = [
    ("uniform", "weighted"),
    ("uniform", "top_heavy"),
    ("weighted", "top_heavy")
]


k_values = [
    3,
    5,
    7
]


# ============================================================
# STORAGE
# ============================================================

f1_results = []
em_results = []


# ============================================================
# F1 ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PAIRED F1 ANALYSIS")
print("=" * 70)


for k in k_values:

    k_df = df[
        df["top_k"] == k
    ].copy()


    # --------------------------------------------------------
    # Pivot F1
    # --------------------------------------------------------

    f1_pivot = k_df.pivot(
        index=[
            "sample_index",
            "question_id"
        ],
        columns="allocation",
        values="f1"
    )


    for first, second in comparisons:

        first_scores = (
            f1_pivot[first]
            .to_numpy()
        )

        second_scores = (
            f1_pivot[second]
            .to_numpy()
        )


        # ----------------------------------------------------
        # Paired differences
        #
        # Positive = second allocation is better
        # ----------------------------------------------------

        differences = (
            second_scores
            - first_scores
        )


        mean_difference = np.mean(
            differences
        )


        # ----------------------------------------------------
        # Wilcoxon
        # ----------------------------------------------------

        if np.allclose(
            differences,
            0
        ):

            wilcoxon_p = 1.0
            wilcoxon_stat = 0.0

        else:

            result = wilcoxon(
                second_scores,
                first_scores,
                alternative="two-sided",
                zero_method="wilcox"
            )

            wilcoxon_p = result.pvalue
            wilcoxon_stat = result.statistic


        # ----------------------------------------------------
        # Bootstrap CI
        # ----------------------------------------------------

        ci_low, ci_high = bootstrap_ci(
            differences,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_SEED + k
        )


        # ----------------------------------------------------
        # Cohen's dz
        # ----------------------------------------------------

        dz = cohens_dz(
            differences
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        f1_results.append(
            {
                "metric":
                    "F1",

                "top_k":
                    k,

                "comparison":
                    f"{first}_vs_{second}",

                "first_allocation":
                    first,

                "second_allocation":
                    second,

                "n":
                    len(differences),

                "first_mean":
                    np.mean(first_scores),

                "second_mean":
                    np.mean(second_scores),

                "mean_difference_second_minus_first":
                    mean_difference,

                "bootstrap_ci_low":
                    ci_low,

                "bootstrap_ci_high":
                    ci_high,

                "wilcoxon_statistic":
                    wilcoxon_stat,

                "wilcoxon_p":
                    wilcoxon_p,

                "cohens_dz":
                    dz
            }
        )


        print(
            f"\nk={k} | "
            f"{first} -> {second}"
        )

        print(
            f"  {first} F1 : "
            f"{np.mean(first_scores):.4f}"
        )

        print(
            f"  {second} F1: "
            f"{np.mean(second_scores):.4f}"
        )

        print(
            f"  Difference: "
            f"{mean_difference:+.4f}"
        )

        print(
            f"  95% CI: "
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )

        print(
            f"  Wilcoxon p: "
            f"{wilcoxon_p:.4e}"
        )

        print(
            f"  Cohen's dz: "
            f"{dz:.4f}"
        )


# ============================================================
# HOLM CORRECTION FOR F1
# ============================================================

f1_results_df = pd.DataFrame(
    f1_results
)

f1_results_df[
    "holm_adjusted_p"
] = holm_correction(
    f1_results_df[
        "wilcoxon_p"
    ].values
)


# ============================================================
# EM ANALYSIS
# ============================================================

print()
print("=" * 70)
print("EXACT MATCH / McNEMAR ANALYSIS")
print("=" * 70)


for k in k_values:

    k_df = df[
        df["top_k"] == k
    ].copy()


    em_pivot = k_df.pivot(
        index=[
            "sample_index",
            "question_id"
        ],
        columns="allocation",
        values="exact_match"
    )


    for first, second in comparisons:

        first_em = (
            em_pivot[first]
            .to_numpy()
        )

        second_em = (
            em_pivot[second]
            .to_numpy()
        )


        # ----------------------------------------------------
        # McNemar
        # ----------------------------------------------------

        b, c, p_value = exact_mcnemar(
            first_em,
            second_em
        )


        first_rate = np.mean(
            first_em
        )

        second_rate = np.mean(
            second_em
        )

        difference = (
            second_rate
            - first_rate
        )


        # ----------------------------------------------------
        # Bootstrap CI for EM difference
        # ----------------------------------------------------

        em_differences = (
            second_em
            - first_em
        )

        ci_low, ci_high = bootstrap_ci(
            em_differences,
            n_bootstrap=N_BOOTSTRAP,
            seed=100 + k
        )


        em_results.append(
            {
                "metric":
                    "Exact_Match",

                "top_k":
                    k,

                "comparison":
                    f"{first}_vs_{second}",

                "first_allocation":
                    first,

                "second_allocation":
                    second,

                "n":
                    len(first_em),

                "first_mean":
                    first_rate,

                "second_mean":
                    second_rate,

                "mean_difference_second_minus_first":
                    difference,

                "bootstrap_ci_low":
                    ci_low,

                "bootstrap_ci_high":
                    ci_high,

                "discordant_first_1_second_0":
                    b,

                "discordant_first_0_second_1":
                    c,

                "mcnemar_p":
                    p_value
            }
        )


        print(
            f"\nk={k} | "
            f"{first} -> {second}"
        )

        print(
            f"  {first} EM : "
            f"{first_rate:.4f}"
        )

        print(
            f"  {second} EM: "
            f"{second_rate:.4f}"
        )

        print(
            f"  Difference: "
            f"{difference:+.4f}"
        )

        print(
            f"  Discordant: "
            f"{b} vs {c}"
        )

        print(
            f"  McNemar p: "
            f"{p_value:.4e}"
        )


# ============================================================
# HOLM CORRECTION FOR EM
# ============================================================

em_results_df = pd.DataFrame(
    em_results
)

em_results_df[
    "holm_adjusted_p"
] = holm_correction(
    em_results_df[
        "mcnemar_p"
    ].values
)


# ============================================================
# COMBINE RESULTS
# ============================================================

final_results = pd.concat(
    [
        f1_results_df,
        em_results_df
    ],
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

final_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print("=" * 70)
print("FINAL STATISTICAL RESULTS")
print("=" * 70)

pd.set_option(
    "display.max_columns",
    None
)

print(
    final_results.to_string(
        index=False
    )
)


# ============================================================
# HIGHLIGHT WEIGHTED VS TOP-HEAVY
# ============================================================

print()
print("=" * 70)
print("WEIGHTED VS TOP-HEAVY")
print("=" * 70)


weighted_top = final_results[
    final_results[
        "comparison"
    ] == "weighted_vs_top_heavy"
]


print(
    weighted_top.to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print(
    "\nTotal F1 comparisons: "
    f"{len(f1_results_df)}"
)

print(
    "Total EM comparisons: "
    f"{len(em_results_df)}"
)

print(
    "\nHolm correction:"
)

print(
    "  F1: 9 comparisons"
)

print(
    "  EM: 9 comparisons"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)