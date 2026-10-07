# ============================================================
# newsqa_mechanism_analysis.py
#
# Mechanistic / Characteristic Analysis
# for the NewsQA Context Allocation Experiment
#
# IMPORTANT:
# This script DOES NOT run LM generation.
#
# It analyzes:
#   1. Gold-context rank under BM25
#   2. Answer position inside the gold context
#   3. Answer survival under each allocation
#   4. Relationship between gold-context rank and F1
#
# Uses the EXACT same:
#   - NewsQA dataset
#   - evaluation setup
#   - retrieval corpus
#   - first 1000 valid examples
#   - BM25 retriever
#   - 350-token budget
#
# Existing generation results are loaded from:
#
#   results/newsqa_context_allocation_detailed_1000.csv
#
# Outputs:
#
#   results/newsqa_mechanism_question_level.csv
#   results/newsqa_mechanism_rank_summary.csv
#   results/newsqa_mechanism_allocation_summary.csv
#   results/newsqa_mechanism_rank_f1_summary.csv
#
# Figures:
#
#   results/figures/newsqa_gold_context_rank_distribution.png
#   results/figures/newsqa_answer_survival_by_allocation.png
#   results/figures/newsqa_f1_by_gold_context_rank.png
#
# ============================================================


import os
import sys
import ast

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

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.retriever import BM25Retriever


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EVAL_SAMPLES = 1000

TOP_K = 7

TOTAL_CONTEXT_BUDGET = 350

MODEL_NAME = "google/flan-t5-small"

DATASET_NAME = "hlt-lab/newsqa-safe"


# Existing generation results
DETAILED_RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "newsqa_context_allocation_detailed_1000.csv"
)


# Output files
QUESTION_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "newsqa_mechanism_question_level.csv"
)

RANK_SUMMARY_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "newsqa_mechanism_rank_summary.csv"
)

ALLOCATION_SUMMARY_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "newsqa_mechanism_allocation_summary.csv"
)

RANK_F1_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "newsqa_mechanism_rank_f1_summary.csv"
)


# Figure directory
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
# CONTEXT ALLOCATION STRATEGIES
# ============================================================

def get_uniform_budgets(
    k,
    total_budget
):
    """
    Divide the total context budget as equally as possible.
    """

    base = total_budget // k

    remainder = total_budget % k

    budgets = [base] * k

    for i in range(remainder):
        budgets[i] += 1

    return budgets


def get_weighted_budgets(
    k,
    total_budget
):
    """
    Moderate rank-aware allocation.
    """

    allocations = {
        1: [350],

        3: [150, 110, 90],

        5: [100, 80, 65, 55, 50],

        7: [100, 70, 50, 40, 35, 30, 25],
    }

    budgets = allocations[k]

    assert sum(budgets) == total_budget

    return budgets


def get_top_heavy_budgets(
    k,
    total_budget
):
    """
    Strong rank-aware allocation.
    """

    allocations = {
        1: [350],

        3: [190, 90, 70],

        5: [150, 80, 50, 40, 30],

        7: [140, 70, 45, 35, 25, 20, 15],
    }

    budgets = allocations[k]

    assert sum(budgets) == total_budget

    return budgets


# ============================================================
# VALIDATE ALLOCATIONS
# ============================================================

ALLOCATION_BUDGETS = {
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


for allocation, budgets in ALLOCATION_BUDGETS.items():

    assert len(budgets) == TOP_K

    assert sum(budgets) == TOTAL_CONTEXT_BUDGET

    assert all(
        budget > 0
        for budget in budgets
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Basic normalization used for comparing contexts.
    """

    if text is None:
        return ""

    text = str(text).lower()

    return " ".join(
        text.split()
    )


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_newsqa_answers(
    example
):
    """
    Convert the NewsQA answer field into a clean list
    of answer strings.

    Handles common representations:

        string
        list[string]
        list[dict]
        dict
    """

    answers = example.get(
        "answers",
        None
    )

    if answers is None:

        answers = example.get(
            "answer",
            None
        )


    if answers is None:
        return []


    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(
        answers,
        str
    ):

        answer = answers.strip()

        return (
            [answer]
            if answer
            else []
        )


    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        answers,
        list
    ):

        cleaned = []

        for answer in answers:

            if isinstance(
                answer,
                str
            ):

                answer = answer.strip()

                if answer:
                    cleaned.append(
                        answer
                    )


            elif isinstance(
                answer,
                dict
            ):

                text = answer.get(
                    "text",
                    None
                )

                if text is None:

                    text = answer.get(
                        "answer",
                        None
                    )

                if text is not None:

                    text = str(
                        text
                    ).strip()

                    if text:
                        cleaned.append(
                            text
                        )


        return cleaned


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        answers,
        dict
    ):

        text_values = answers.get(
            "text",
            None
        )


        if text_values is not None:

            if isinstance(
                text_values,
                str
            ):

                text_values = [
                    text_values
                ]


            if isinstance(
                text_values,
                list
            ):

                return [
                    str(x).strip()
                    for x in text_values
                    if str(x).strip()
                ]


        answer_values = answers.get(
            "answer",
            None
        )


        if answer_values is not None:

            if isinstance(
                answer_values,
                str
            ):

                answer_values = [
                    answer_values
                ]


            if isinstance(
                answer_values,
                list
            ):

                return [
                    str(x).strip()
                    for x in answer_values
                    if str(x).strip()
                ]


    return []


# ============================================================
# FIND ANSWER SPAN
# ============================================================

def find_answer_span(
    context,
    answers
):
    """
    Find an answer string inside the original context.

    We deliberately prefer exact substring matching because
    character offsets must refer to the ORIGINAL context.

    Returns:

        char_start
        char_end
        answer_text
    """

    if not context:
        return None, None, None

    if not answers:
        return None, None, None


    # --------------------------------------------------------
    # Pass 1:
    # Exact case-sensitive match
    # --------------------------------------------------------

    for answer in answers:

        if not answer:
            continue

        answer = str(
            answer
        ).strip()

        start = context.find(
            answer
        )

        if start != -1:

            return (
                start,
                start + len(answer),
                answer
            )


    # --------------------------------------------------------
    # Pass 2:
    # Case-insensitive match
    #
    # This preserves valid character offsets in the
    # ORIGINAL context.
    # --------------------------------------------------------

    lower_context = context.lower()


    for answer in answers:

        if not answer:
            continue

        answer = str(
            answer
        ).strip()

        lower_answer = answer.lower()

        start = lower_context.find(
            lower_answer
        )

        if start != -1:

            return (
                start,
                start + len(answer),
                answer
            )


    # --------------------------------------------------------
    # No safe original-context span found.
    # --------------------------------------------------------

    return None, None, None


# ============================================================
# TOKEN ANSWER POSITION
# ============================================================

def get_answer_token_span(
    tokenizer,
    context,
    char_start,
    char_end
):
    """
    Tokenize the complete context with offsets and determine
    which tokens overlap the answer span.

    Returns:

        first_answer_token
        last_answer_token
        answer_token_count
        total_context_tokens
    """

    encoded = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )


    offsets = encoded[
        "offset_mapping"
    ]


    answer_token_indices = []


    for token_index, (
        token_start,
        token_end
    ) in enumerate(offsets):

        if token_end <= token_start:
            continue


        # Token overlaps answer span.
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
        min(
            answer_token_indices
        ),

        max(
            answer_token_indices
        ),

        len(
            answer_token_indices
        ),

        len(offsets)
    )


# ============================================================
# ANSWER SURVIVAL
# ============================================================

def answer_survives_budget(
    first_answer_token,
    last_answer_token,
    budget
):
    """
    True if the COMPLETE answer span is contained within
    the first `budget` document tokens.
    """

    if (
        first_answer_token is None
        or last_answer_token is None
    ):

        return False


    return (
        last_answer_token < budget
    )


# ============================================================
# LOAD EXISTING DETAILED RESULTS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING EXISTING NEWSQA RESULTS"
)

print(
    "=" * 70
)


if not os.path.exists(
    DETAILED_RESULTS_FILE
):

    raise FileNotFoundError(
        "\nCould not find:\n"
        f"{DETAILED_RESULTS_FILE}\n\n"
        "Run the NewsQA context allocation "
        "experiment first."
    )


detailed_df = pd.read_csv(
    DETAILED_RESULTS_FILE
)


print(
    f"\nLoaded {len(detailed_df):,} "
    "detailed result rows."
)


# ============================================================
# VALIDATE DETAILED RESULTS
# ============================================================

expected_rows = (
    NUM_EVAL_SAMPLES
    * 4
    * 3
)

if len(detailed_df) != expected_rows:

    raise ValueError(
        f"Expected {expected_rows:,} "
        f"rows but found "
        f"{len(detailed_df):,}."
    )


required_columns = {
    "sample_index",
    "question_id",
    "top_k",
    "allocation",
    "f1",
    "exact_match",
    "retrieval_recall",
}


missing_columns = (
    required_columns
    - set(detailed_df.columns)
)


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


print(
    "Detailed-results validation passed."
)


# ============================================================
# LOAD NEWSQA
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING NEWSQA DATASET"
)

print(
    "=" * 70
)


dataset = load_dataset(
    DATASET_NAME
)


# ------------------------------------------------------------
# Select the same split used by the main experiment.
# ------------------------------------------------------------

if "validation" in dataset:

    data = dataset[
        "validation"
    ]

    selected_split = "validation"


elif "dev" in dataset:

    data = dataset[
        "dev"
    ]

    selected_split = "dev"


elif "test" in dataset:

    data = dataset[
        "test"
    ]

    selected_split = "test"


else:

    data = dataset[
        "train"
    ]

    selected_split = "train"


print(
    f"\nSelected split: "
    f"{selected_split}"
)


# ============================================================
# FILTER VALID EXAMPLES
# ============================================================

valid_examples = []


for example in data:

    context = example.get(
        "context",
        None
    )

    question = example.get(
        "question",
        None
    )

    answers = extract_newsqa_answers(
        example
    )


    if context is None:
        continue

    if question is None:
        continue


    context = str(
        context
    ).strip()

    question = str(
        question
    ).strip()


    if not context:
        continue

    if not question:
        continue

    if not answers:
        continue


    example_id = example.get(
        "id",
        example.get(
            "qid",
            len(valid_examples)
        )
    )


    valid_examples.append(
        {
            "question":
                question,

            "context":
                context,

            "answers":
                answers,

            "id":
                example_id
        }
    )


print(
    f"Valid examples available: "
    f"{len(valid_examples):,}"
)


if len(valid_examples) < NUM_EVAL_SAMPLES:

    raise ValueError(
        f"Need {NUM_EVAL_SAMPLES} "
        f"valid examples but only "
        f"{len(valid_examples)} found."
    )


# EXACT same first-1000 setup as main experiment.
eval_data = valid_examples[
    :NUM_EVAL_SAMPLES
]


print(
    f"Evaluation examples used: "
    f"{len(eval_data):,}"
)


# ============================================================
# BUILD FULL RETRIEVAL CORPUS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "BUILDING FULL NEWSQA BM25 CORPUS"
)

print(
    "=" * 70
)


corpus = [
    example["context"]
    for example in valid_examples
]


print(
    f"\nCorpus documents: "
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

print(
    "=" * 70
)

print(
    "LOADING TOKENIZER"
)

print(
    "=" * 70
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    f"\nTokenizer loaded: "
    f"{MODEL_NAME}"
)


# ============================================================
# PRINT ALLOCATIONS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "CONTEXT ALLOCATIONS"
)

print(
    "=" * 70
)


for allocation, budgets in (
    ALLOCATION_BUDGETS.items()
):

    print(
        f"{allocation:12s}: "
        f"{budgets} "
        f"(sum={sum(budgets)})"
    )


# ============================================================
# ANALYZE EACH QUESTION
# ============================================================

print()

print(
    "=" * 70
)

print(
    "RUNNING NEWSQA MECHANISM ANALYSIS"
)

print(
    "=" * 70
)


question_results = []


for sample_index, sample in enumerate(
    tqdm(
        eval_data,
        desc="Analyzing NewsQA questions"
    )
):

    question_id = sample[
        "id"
    ]

    question = sample[
        "question"
    ]

    context = sample[
        "context"
    ]

    answers = sample[
        "answers"
    ]


    # ========================================================
    # RETRIEVE TOP-7
    # ========================================================

    retrieved_docs = retriever.retrieve(
        question,
        k=TOP_K
    )


    if len(retrieved_docs) != TOP_K:

        raise ValueError(
            f"Expected {TOP_K} retrieved "
            f"documents but got "
            f"{len(retrieved_docs)} "
            f"for sample {sample_index}."
        )


    # ========================================================
    # FIND GOLD CONTEXT RANK
    # ========================================================

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


    # ========================================================
    # ANSWER CHARACTER POSITION
    # ========================================================

    (
        answer_char_start,
        answer_char_end,
        answer_text
    ) = find_answer_span(
        context,
        answers
    )


    # ========================================================
    # ANSWER TOKEN POSITION
    # ========================================================

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
            )[
                "input_ids"
            ]
        )


    # ========================================================
    # ANSWER POSITION NORMALIZED
    # ========================================================

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


    # ========================================================
    # ALLOCATION SURVIVAL
    # ========================================================

    survival = {}

    gold_document_budgets = {}


    for allocation, budgets in (
        ALLOCATION_BUDGETS.items()
    ):

        if gold_rank is not None:

            gold_document_budget = (
                budgets[
                    gold_rank - 1
                ]
            )

            survives = (
                answer_survives_budget(
                    first_answer_token,
                    last_answer_token,
                    gold_document_budget
                )
            )

        else:

            gold_document_budget = np.nan

            survives = False


        gold_document_budgets[
            allocation
        ] = gold_document_budget


        survival[
            allocation
        ] = survives


    # ========================================================
    # RETRIEVAL RECALL
    # ========================================================

    retrieved_top7 = int(
        gold_rank is not None
    )


    # ========================================================
    # STORE
    # ========================================================

    question_results.append(
        {
            "sample_index":
                sample_index,

            "question_id":
                question_id,

            "gold_context_rank":
                (
                    gold_rank
                    if gold_rank is not None
                    else 0
                ),

            "retrieved_top7":
                retrieved_top7,

            "answer_text":
                answer_text,

            "answer_span_found":
                int(
                    answer_char_start
                    is not None
                ),

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

            "uniform_gold_document_budget":
                gold_document_budgets[
                    "uniform"
                ],

            "weighted_gold_document_budget":
                gold_document_budgets[
                    "weighted"
                ],

            "top_heavy_gold_document_budget":
                gold_document_budgets[
                    "top_heavy"
                ],

            "uniform_answer_survives":
                int(
                    survival[
                        "uniform"
                    ]
                ),

            "weighted_answer_survives":
                int(
                    survival[
                        "weighted"
                    ]
                ),

            "top_heavy_answer_survives":
                int(
                    survival[
                        "top_heavy"
                    ]
                )
        }
    )


mechanism_df = pd.DataFrame(
    question_results
)


# ============================================================
# VALIDATE QUESTION-LEVEL ANALYSIS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "VALIDATING MECHANISM RESULTS"
)

print(
    "=" * 70
)


if len(mechanism_df) != NUM_EVAL_SAMPLES:

    raise ValueError(
        f"Expected {NUM_EVAL_SAMPLES} "
        f"mechanism rows but found "
        f"{len(mechanism_df)}."
    )


assert (
    mechanism_df[
        "sample_index"
    ].nunique()
    == NUM_EVAL_SAMPLES
)


# ============================================================
# MERGE WITH EXISTING F1 RESULTS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "MERGING WITH NEWSQA QA RESULTS"
)

print(
    "=" * 70
)


# Only k=7 is needed for mechanism analysis.
f1_df = detailed_df[
    detailed_df["top_k"] == TOP_K
].copy()


expected_f1_rows = (
    NUM_EVAL_SAMPLES
    * 3
)


if len(f1_df) != expected_f1_rows:

    raise ValueError(
        f"Expected {expected_f1_rows} "
        f"k=7 rows but found "
        f"{len(f1_df)}."
    )


f1_df = f1_df[
    [
        "sample_index",
        "question_id",
        "allocation",
        "f1",
        "exact_match"
    ]
]


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


f1_pivot.columns = [
    "_".join(
        [
            str(x)
            for x in column
            if str(x) != ""
        ]
    ).strip("_")
    for column in f1_pivot.columns
]


mechanism_df = mechanism_df.merge(
    f1_pivot,
    on=[
        "sample_index",
        "question_id"
    ],
    how="left",
    validate="one_to_one"
)


# ============================================================
# VALIDATE MERGE
# ============================================================

for column in [
    "f1_uniform",
    "f1_weighted",
    "f1_top_heavy",
    "exact_match_uniform",
    "exact_match_weighted",
    "exact_match_top_heavy"
]:

    if mechanism_df[
        column
    ].isna().any():

        raise ValueError(
            f"Missing values after merge "
            f"in column: {column}"
        )


print(
    "QA-result merge passed."
)


# ============================================================
# SAVE QUESTION-LEVEL RESULTS
# ============================================================

mechanism_df.to_csv(
    QUESTION_OUTPUT,
    index=False
)


print(
    f"\nSaved:\n"
    f"{QUESTION_OUTPUT}"
)


# ============================================================
# GOLD CONTEXT RANK SUMMARY
# ============================================================

print()

print(
    "=" * 70
)

print(
    "GOLD CONTEXT RANK ANALYSIS"
)

print(
    "=" * 70
)


rank_summary = (
    mechanism_df
    .groupby(
        "gold_context_rank"
    )
    .agg(

        questions=(
            "question_id",
            "count"
        ),

        retrieval_rate=(
            "retrieved_top7",
            "mean"
        ),

        answer_span_found_rate=(
            "answer_span_found",
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

print(
    "=" * 70
)

print(
    "ANSWER SURVIVAL BY ALLOCATION"
)

print(
    "=" * 70
)


allocation_summary = []


for allocation in [
    "uniform",
    "weighted",
    "top_heavy"
]:

    column = (
        f"{allocation}"
        "_answer_survives"
    )


    survival_rate = (
        mechanism_df[column]
        .mean()
    )


    # Conditional on the gold context actually
    # appearing in top-7.
    retrieved_subset = mechanism_df[
        mechanism_df[
            "retrieved_top7"
        ] == 1
    ]


    if len(retrieved_subset) > 0:

        conditional_survival = (
            retrieved_subset[
                column
            ].mean()
        )

    else:

        conditional_survival = np.nan


    allocation_summary.append(
        {
            "allocation":
                allocation,

            "top_k":
                TOP_K,

            "total_context_budget":
                TOTAL_CONTEXT_BUDGET,

            "answer_survival_rate":
                survival_rate,

            "answer_survival_percent":
                survival_rate * 100,

            "conditional_on_retrieved_survival":
                conditional_survival,

            "conditional_survival_percent":
                conditional_survival * 100
                if not np.isnan(
                    conditional_survival
                )
                else np.nan
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

print(
    "=" * 70
)

print(
    "F1 BY GOLD CONTEXT RANK"
)

print(
    "=" * 70
)


rank_f1_summary = (
    mechanism_df[
        mechanism_df[
            "gold_context_rank"
        ] > 0
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

print(
    "=" * 70
)

print(
    "RETRIEVAL STATISTICS"
)

print(
    "=" * 70
)


retrieved_count = int(
    mechanism_df[
        "retrieved_top7"
    ].sum()
)


retrieval_rate = (
    retrieved_count
    / len(mechanism_df)
)


print(
    f"\nGold context retrieved "
    f"in top-7:"
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
# ANSWER SPAN STATISTICS
# ============================================================

print()

print(
    "=" * 70
)

print(
    "ANSWER SPAN STATISTICS"
)

print(
    "=" * 70
)


valid_spans = mechanism_df[
    mechanism_df[
        "answer_span_found"
    ] == 1
]


span_rate = (
    len(valid_spans)
    / len(mechanism_df)
)


print(
    f"\nQuestions with answer span "
    f"located in original context:"
)


print(
    f"  {len(valid_spans):,} / "
    f"{len(mechanism_df):,}"
)


print(
    f"  Span identification rate = "
    f"{span_rate:.4f}"
)


if len(valid_spans) > 0:

    print(
        f"\nMean first answer token: "
        f"{valid_spans['answer_first_token'].mean():.2f}"
    )


    print(
        f"Median first answer token: "
        f"{valid_spans['answer_first_token'].median():.2f}"
    )


    print(
        f"Mean normalized answer position: "
        f"{valid_spans['answer_position_ratio'].mean():.4f}"
    )


    print(
        f"Median normalized answer position: "
        f"{valid_spans['answer_position_ratio'].median():.4f}"
    )


# ============================================================
# FIGURE 1
# GOLD CONTEXT RANK DISTRIBUTION
# ============================================================

print()

print(
    "=" * 70
)

print(
    "GENERATING FIGURES"
)

print(
    "=" * 70
)


rank_counts = (
    mechanism_df[
        mechanism_df[
            "gold_context_rank"
        ] > 0
    ][
        "gold_context_rank"
    ]
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
    "NewsQA: Distribution of Gold Context Rank"
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
    "newsqa_gold_context_rank_distribution.png"
)


plt.savefig(
    rank_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2
# ANSWER SURVIVAL
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plot_allocations = [
    "uniform",
    "weighted",
    "top_heavy"
]


plot_values = [
    allocation_summary_df[
        allocation_summary_df[
            "allocation"
        ] == allocation
    ][
        "answer_survival_percent"
    ].iloc[0]

    for allocation in plot_allocations
]


plt.bar(
    plot_allocations,
    plot_values
)


plt.xlabel(
    "Context Allocation Strategy"
)


plt.ylabel(
    "Answer-Containing Evidence Survived (%)"
)


plt.title(
    "NewsQA: Answer Evidence Survival "
    "Under Fixed 350-Token Budget"
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
    "newsqa_answer_survival_by_allocation.png"
)


plt.savefig(
    survival_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
# F1 BY GOLD CONTEXT RANK
#
# IMPORTANT:
# Only plot ranks with enough observations.
#
# This avoids visually over-interpreting extremely small
# groups such as rank 6/7 if the data are sparse.
# ============================================================

MIN_RANK_SAMPLES_FOR_PLOT = 20


plot_rank_f1 = rank_f1_summary[
    rank_f1_summary[
        "questions"
    ] >= MIN_RANK_SAMPLES_FOR_PLOT
].copy()


plt.figure(
    figsize=(10, 6)
)


for allocation in [
    "uniform",
    "weighted",
    "top_heavy"
]:

    plt.plot(
        plot_rank_f1[
            "gold_context_rank"
        ],

        plot_rank_f1[
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
    "NewsQA: QA Performance as a Function "
    "of Gold Context Rank"
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
    "newsqa_f1_by_gold_context_rank.png"
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

print(
    "=" * 70
)

print(
    "NEWSQA MECHANISM ANALYSIS COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nGenerated files:"
)


print(
    f"\n1. {QUESTION_OUTPUT}"
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

print(
    "=" * 70
)

print(
    "KEY FINDINGS"
)

print(
    "=" * 70
)


print(
    f"\nTop-7 retrieval recall:"
    f" {retrieval_rate * 100:.2f}%"
)


print(
    f"Answer span identification:"
    f" {span_rate * 100:.2f}%"
)


print(
    "\nAnswer survival:"
)


for allocation in [
    "uniform",
    "weighted",
    "top_heavy"
]:

    row = allocation_summary_df[
        allocation_summary_df[
            "allocation"
        ] == allocation
    ].iloc[0]


    print(
        f"  {allocation:10s}: "
        f"{row['answer_survival_percent']:.2f}% "
        f"overall, "
        f"{row['conditional_survival_percent']:.2f}% "
        f"conditional on retrieval"
    )


print(
    "\nDone."
)