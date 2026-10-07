# ============================================================
# mechanism_analysis.py
#
# Mechanistic / Characteristic Analysis
# for the Context Allocation Experiment
#
# NO LM GENERATION IS PERFORMED.
#
# ============================================================

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORT
# ============================================================

from src.retriever import BM25Retriever


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EVAL_SAMPLES = 1000
TOP_K = 7

MODEL_NAME = "google/flan-t5-small"
TOTAL_CONTEXT_BUDGET = 350

DETAILED_RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "context_allocation_detailed_1000.csv"
)

QUESTION_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "mechanism_question_level.csv"
)

RANK_SUMMARY_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "mechanism_rank_summary.csv"
)

ALLOCATION_SUMMARY_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "mechanism_allocation_summary.csv"
)

RANK_F1_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "mechanism_rank_f1_summary.csv"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# ALLOCATION STRATEGIES
# ============================================================

def get_uniform_budgets(k, total_budget):

    base = total_budget // k
    remainder = total_budget % k

    budgets = [base] * k

    for i in range(remainder):
        budgets[i] += 1

    assert sum(budgets) == total_budget

    return budgets


def get_weighted_budgets(k, total_budget):

    allocations = {
        1: [350],
        3: [150, 110, 90],
        5: [100, 80, 65, 55, 50],
        7: [100, 70, 50, 40, 35, 30, 25]
    }

    budgets = allocations[k]

    assert sum(budgets) == total_budget

    return budgets


def get_top_heavy_budgets(k, total_budget):

    allocations = {
        1: [350],
        3: [190, 90, 70],
        5: [150, 80, 50, 40, 30],
        7: [140, 70, 45, 35, 25, 20, 15]
    }

    budgets = allocations[k]

    assert sum(budgets) == total_budget

    return budgets


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    return " ".join(
        text.split()
    )


# ============================================================
# FIND ANSWER CHARACTER SPAN
# ============================================================

def find_answer_span(context, answers):

    if not answers:
        return None, None, None

    # Try exact matching first
    for answer in answers:

        if not answer:
            continue

        start = context.find(answer)

        if start != -1:

            return (
                start,
                start + len(answer),
                answer
            )

    # Fallback to normalized matching
    normalized_context = normalize_text(
        context
    )

    for answer in answers:

        normalized_answer = normalize_text(
            answer
        )

        if not normalized_answer:
            continue

        position = normalized_context.find(
            normalized_answer
        )

        if position != -1:

            return (
                position,
                position + len(normalized_answer),
                answer
            )

    return None, None, None


# ============================================================
# FIND ANSWER TOKEN SPAN
# ============================================================

def get_answer_token_span(
    tokenizer,
    context,
    char_start,
    char_end
):

    encoded = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )

    offsets = encoded["offset_mapping"]

    answer_token_indices = []

    for token_index, (
        token_start,
        token_end
    ) in enumerate(offsets):

        if token_end <= token_start:
            continue

        if (
            token_start < char_end
            and token_end > char_start
        ):

            answer_token_indices.append(
                token_index
            )

    if not answer_token_indices:

        return (
            None,
            None,
            0,
            len(offsets)
        )

    return (
        min(answer_token_indices),
        max(answer_token_indices),
        len(answer_token_indices),
        len(offsets)
    )


# ============================================================
# CHECK WHETHER ANSWER SURVIVES
# ============================================================

def answer_survives_budget(
    first_answer_token,
    last_answer_token,
    budget
):

    if (
        first_answer_token is None
        or last_answer_token is None
    ):
        return False

    return last_answer_token < budget


# ============================================================
# LOAD EXISTING EXPERIMENT RESULTS
# ============================================================

print()
print("=" * 70)
print("LOADING EXISTING EXPERIMENT RESULTS")
print("=" * 70)

if not os.path.exists(
    DETAILED_RESULTS_FILE
):

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{DETAILED_RESULTS_FILE}\n\n"
        "Make sure the 1000-sample "
        "context allocation experiment "
        "has already been run."
    )


detailed_df = pd.read_csv(
    DETAILED_RESULTS_FILE
)

print(
    f"\nLoaded {len(detailed_df):,} rows."
)


# ============================================================
# BASIC VALIDATION
# ============================================================

expected_rows = (
    NUM_EVAL_SAMPLES
    * 4
    * 3
)

if len(detailed_df) != expected_rows:

    print(
        "\nWARNING:"
        f" Expected {expected_rows:,} rows "
        f"but found {len(detailed_df):,}."
    )


# ============================================================
# LOAD SQuAD
# ============================================================

print()
print("=" * 70)
print("LOADING SQuAD")
print("=" * 70)

dataset = load_dataset(
    "rajpurkar/squad"
)

validation = dataset["validation"]

eval_count = min(
    NUM_EVAL_SAMPLES,
    len(validation)
)

eval_data = validation.select(
    range(eval_count)
)

print(
    f"\nEvaluation questions: {eval_count:,}"
)


# ============================================================
# BUILD FULL RETRIEVAL CORPUS
# ============================================================

print()
print("=" * 70)
print("BUILDING FULL BM25 CORPUS")
print("=" * 70)

corpus = list(
    dict.fromkeys(
        validation["context"]
    )
)

print(
    f"\nUnique corpus documents: "
    f"{len(corpus):,}"
)

retriever = BM25Retriever(
    corpus
)

print(
    "BM25 retriever ready."
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    f"\nLoaded: {MODEL_NAME}"
)


# ============================================================
# ALLOCATION CONFIGURATION
# ============================================================

allocation_budgets = {
    "uniform": get_uniform_budgets(
        TOP_K,
        TOTAL_CONTEXT_BUDGET
    ),

    "weighted": get_weighted_budgets(
        TOP_K,
        TOTAL_CONTEXT_BUDGET
    ),

    "top_heavy": get_top_heavy_budgets(
        TOP_K,
        TOTAL_CONTEXT_BUDGET
    )
}

print()
print("k=7 allocation budgets:")

for name, budgets in allocation_budgets.items():

    print(
        f"  {name:10s}: {budgets}"
    )


# ============================================================
# MECHANISM ANALYSIS
# ============================================================

print()
print("=" * 70)
print("RUNNING MECHANISM ANALYSIS")
print("=" * 70)

question_results = []


for sample_index, sample in enumerate(
    tqdm(
        eval_data,
        desc="Analyzing questions"
    )
):

    question_id = sample["id"]
    question = sample["question"]
    context = sample["context"]

    answers = sample["answers"]["text"]


    # --------------------------------------------------------
    # RETRIEVE TOP-7
    # --------------------------------------------------------

    retrieved_docs = retriever.retrieve(
        question,
        k=TOP_K
    )


    # --------------------------------------------------------
    # FIND GOLD CONTEXT RANK
    # --------------------------------------------------------

    normalized_gold = normalize_text(
        context
    )

    gold_rank = None

    for rank, document in enumerate(
        retrieved_docs,
        start=1
    ):

        if (
            normalize_text(document)
            == normalized_gold
        ):

            gold_rank = rank
            break


    # --------------------------------------------------------
    # FIND ANSWER CHARACTER POSITION
    # --------------------------------------------------------

    (
        answer_char_start,
        answer_char_end,
        answer_text
    ) = find_answer_span(
        context,
        answers
    )


    # --------------------------------------------------------
    # FIND ANSWER TOKEN POSITION
    # --------------------------------------------------------

    if (
        answer_char_start is not None
        and answer_char_end is not None
    ):

        (
            first_answer_token,
            last_answer_token,
            answer_token_count,
            total_context_tokens
        ) = get_answer_token_span(
            tokenizer,
            context,
            answer_char_start,
            answer_char_end
        )

    else:

        first_answer_token = None
        last_answer_token = None
        answer_token_count = 0

        total_context_tokens = len(
            tokenizer(
                context,
                add_special_tokens=False
            )["input_ids"]
        )


    # --------------------------------------------------------
    # NORMALIZED ANSWER POSITION
    # --------------------------------------------------------

    if (
        first_answer_token is not None
        and total_context_tokens > 0
    ):

        answer_position_ratio = (
            first_answer_token
            / total_context_tokens
        )

    else:

        answer_position_ratio = np.nan


    # --------------------------------------------------------
    # ANSWER SURVIVAL
    # --------------------------------------------------------

    survival = {}

    for allocation, budgets in (
        allocation_budgets.items()
    ):

        if gold_rank is not None:

            gold_document_budget = (
                budgets[gold_rank - 1]
            )

            survives = answer_survives_budget(
                first_answer_token,
                last_answer_token,
                gold_document_budget
            )

        else:

            gold_document_budget = np.nan
            survives = False

        survival[allocation] = survives


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    question_results.append(
        {
            "sample_index": sample_index,

            "question_id": question_id,

            "gold_context_rank": (
                gold_rank
                if gold_rank is not None
                else 0
            ),

            "retrieved_top7": int(
                gold_rank is not None
            ),

            "answer_text": answer_text,

            "answer_char_start":
                answer_char_start,

            "answer_char_end":
                answer_char_end,

            "answer_first_token":
                first_answer_token,

            "answer_last_token":
                last_answer_token,

            "answer_token_count":
                answer_token_count,

            "total_context_tokens":
                total_context_tokens,

            "answer_position_ratio":
                answer_position_ratio,

            "uniform_answer_survives":
                int(
                    survival["uniform"]
                ),

            "weighted_answer_survives":
                int(
                    survival["weighted"]
                ),

            "top_heavy_answer_survives":
                int(
                    survival["top_heavy"]
                )
        }
    )


mechanism_df = pd.DataFrame(
    question_results
)


# ============================================================
# MERGE WITH EXISTING F1 RESULTS
# ============================================================

print()
print("=" * 70)
print("MERGING WITH QA RESULTS")
print("=" * 70)

f1_df = detailed_df[
    detailed_df["top_k"] == TOP_K
][
    [
        "sample_index",
        "question_id",
        "allocation",
        "f1",
        "exact_match"
    ]
].copy()


f1_pivot = f1_df.pivot(
    index=[
        "sample_index",
        "question_id"
    ],
    columns="allocation",
    values=[
        "f1",
        "exact_match"
    ]
).reset_index()


# Flatten MultiIndex columns
flattened_columns = []

for column in f1_pivot.columns:

    if isinstance(column, tuple):

        parts = [
            str(x)
            for x in column
            if str(x) not in ("", "None")
        ]

        flattened_columns.append(
            "_".join(parts)
        )

    else:

        flattened_columns.append(
            str(column)
        )


f1_pivot.columns = flattened_columns


mechanism_df = mechanism_df.merge(
    f1_pivot,
    on=[
        "sample_index",
        "question_id"
    ],
    how="left"
)


# ============================================================
# SAVE QUESTION-LEVEL RESULTS
# ============================================================

mechanism_df.to_csv(
    QUESTION_OUTPUT,
    index=False
)

print(
    f"\nSaved:\n{QUESTION_OUTPUT}"
)


# ============================================================
# GOLD CONTEXT RANK SUMMARY
# ============================================================

print()
print("=" * 70)
print("GOLD CONTEXT RANK ANALYSIS")
print("=" * 70)

rank_summary = (
    mechanism_df
    .groupby("gold_context_rank")
    .agg(
        questions=(
            "question_id",
            "count"
        ),

        retrieval_rate=(
            "retrieved_top7",
            "mean"
        ),

        mean_answer_position=(
            "answer_position_ratio",
            "mean"
        ),

        mean_answer_token=(
            "answer_first_token",
            "mean"
        ),

        uniform_survival=(
            "uniform_answer_survives",
            "mean"
        ),

        weighted_survival=(
            "weighted_answer_survives",
            "mean"
        ),

        top_heavy_survival=(
            "top_heavy_answer_survives",
            "mean"
        ),

        uniform_f1=(
            "f1_uniform",
            "mean"
        ),

        weighted_f1=(
            "f1_weighted",
            "mean"
        ),

        top_heavy_f1=(
            "f1_top_heavy",
            "mean"
        )
    )
    .reset_index()
)

rank_summary.to_csv(
    RANK_SUMMARY_OUTPUT,
    index=False
)

print(
    rank_summary.to_string(
        index=False
    )
)


# ============================================================
# ALLOCATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("ANSWER SURVIVAL BY ALLOCATION")
print("=" * 70)

allocation_summary = []

for allocation in [
    "uniform",
    "weighted",
    "top_heavy"
]:

    column = (
        f"{allocation}_answer_survives"
    )

    survival_rate = (
        mechanism_df[column]
        .mean()
    )

    allocation_summary.append(
        {
            "allocation": allocation,

            "top_k": TOP_K,

            "total_context_budget":
                TOTAL_CONTEXT_BUDGET,

            "answer_survival_rate":
                survival_rate,

            "answer_survival_percent":
                survival_rate * 100
        }
    )


allocation_summary_df = pd.DataFrame(
    allocation_summary
)

allocation_summary_df.to_csv(
    ALLOCATION_SUMMARY_OUTPUT,
    index=False
)

print(
    allocation_summary_df.to_string(
        index=False
    )
)


# ============================================================
# F1 BY GOLD CONTEXT RANK
# ============================================================

print()
print("=" * 70)
print("F1 BY GOLD CONTEXT RANK")
print("=" * 70)

rank_f1_summary = (
    mechanism_df[
        mechanism_df["gold_context_rank"] > 0
    ]
    .groupby(
        "gold_context_rank"
    )
    .agg(
        questions=(
            "question_id",
            "count"
        ),

        uniform_f1=(
            "f1_uniform",
            "mean"
        ),

        weighted_f1=(
            "f1_weighted",
            "mean"
        ),

        top_heavy_f1=(
            "f1_top_heavy",
            "mean"
        ),

        uniform_survival=(
            "uniform_answer_survives",
            "mean"
        ),

        weighted_survival=(
            "weighted_answer_survives",
            "mean"
        ),

        top_heavy_survival=(
            "top_heavy_answer_survives",
            "mean"
        )
    )
    .reset_index()
)

rank_f1_summary.to_csv(
    RANK_F1_OUTPUT,
    index=False
)

print(
    rank_f1_summary.to_string(
        index=False
    )
)


# ============================================================
# RETRIEVAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("RETRIEVAL STATISTICS")
print("=" * 70)

retrieved_count = int(
    mechanism_df["retrieved_top7"].sum()
)

retrieval_rate = (
    retrieved_count
    / len(mechanism_df)
)

print(
    f"\nGold context retrieved in top-7:"
)

print(
    f"  {retrieved_count:,} / "
    f"{len(mechanism_df):,}"
)

print(
    f"  Recall@7 = "
    f"{retrieval_rate:.4f}"
)


# ============================================================
# ANSWER POSITION STATISTICS
# ============================================================

print()
print("=" * 70)
print("ANSWER POSITION STATISTICS")
print("=" * 70)

valid_positions = mechanism_df[
    mechanism_df[
        "answer_first_token"
    ].notna()
]

print(
    f"\nQuestions with identified answer span: "
    f"{len(valid_positions):,}"
)

print(
    f"Mean first answer token: "
    f"{valid_positions['answer_first_token'].mean():.2f}"
)

print(
    f"Median first answer token: "
    f"{valid_positions['answer_first_token'].median():.2f}"
)

print(
    f"Mean normalized answer position: "
    f"{valid_positions['answer_position_ratio'].mean():.4f}"
)


# ============================================================
# FIGURE 1: GOLD RANK DISTRIBUTION
# ============================================================

rank_counts = (
    mechanism_df[
        mechanism_df["gold_context_rank"] > 0
    ]["gold_context_rank"]
    .value_counts()
    .sort_index()
)

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    rank_counts.index,
    rank_counts.values
)

plt.xlabel(
    "Gold Context Rank in BM25 Top-7"
)

plt.ylabel(
    "Number of Questions"
)

plt.title(
    "Distribution of Gold Context Rank"
)

plt.xticks(
    range(1, TOP_K + 1)
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

rank_figure = os.path.join(
    FIGURE_DIR,
    "gold_context_rank_distribution.png"
)

plt.savefig(
    rank_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2: ANSWER SURVIVAL
# ============================================================

plot_allocations = [
    "uniform",
    "weighted",
    "top_heavy"
]

plot_values = [
    allocation_summary_df[
        allocation_summary_df["allocation"]
        == allocation
    ][
        "answer_survival_percent"
    ].iloc[0]
    for allocation in plot_allocations
]

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    plot_allocations,
    plot_values
)

plt.xlabel(
    "Context Allocation Strategy"
)

plt.ylabel(
    "Answer Evidence Survival (%)"
)

plt.title(
    "Answer Evidence Survival Under Fixed 350-Token Budget"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

survival_figure = os.path.join(
    FIGURE_DIR,
    "answer_survival_by_allocation.png"
)

plt.savefig(
    survival_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3: F1 BY GOLD RANK
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for allocation in [
    "uniform",
    "weighted",
    "top_heavy"
]:

    plt.plot(
        rank_f1_summary[
            "gold_context_rank"
        ],
        rank_f1_summary[
            f"{allocation}_f1"
        ],
        marker="o",
        linewidth=2,
        label=allocation
    )

plt.xlabel(
    "Gold Context Rank in BM25 Top-7"
)

plt.ylabel(
    "Token F1"
)

plt.title(
    "QA Performance as a Function of Gold Context Rank"
)

plt.xticks(
    range(1, TOP_K + 1)
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

f1_rank_figure = os.path.join(
    FIGURE_DIR,
    "f1_by_gold_context_rank.png"
)

plt.savefig(
    f1_rank_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MECHANISM ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    f"1. {QUESTION_OUTPUT}"
)

print(
    f"2. {RANK_SUMMARY_OUTPUT}"
)

print(
    f"3. {ALLOCATION_SUMMARY_OUTPUT}"
)

print(
    f"4. {RANK_F1_OUTPUT}"
)

print(
    "\nFigures:"
)

print(
    f"5. {rank_figure}"
)

print(
    f"6. {survival_figure}"
)

print(
    f"7. {f1_rank_figure}"
)


# ============================================================
# KEY FINDINGS
# ============================================================

print()
print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print(
    f"\nTop-7 retrieval recall: "
    f"{retrieval_rate * 100:.2f}%"
)

for allocation in plot_allocations:

    survival_rate = allocation_summary_df[
        allocation_summary_df["allocation"]
        == allocation
    ][
        "answer_survival_percent"
    ].iloc[0]

    print(
        f"\n{allocation}:"
    )

    print(
        f"  Answer survival = "
        f"{survival_rate:.2f}%"
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)