"""
NewsQA Context Allocation Experiment
====================================

Research question:
Given a fixed document-context budget, how should retrieved
context be allocated across ranked documents?

Dataset:
    NewsQA (hlt-lab/newsqa-safe)

Setup:
    - BM25 retrieval
    - Frozen FLAN-T5-small
    - Total document budget = 350 tokens
    - k = 1, 3, 5, 7
    - Uniform allocation
    - Weighted allocation
    - Top-heavy allocation

Metrics:
    - Exact Match
    - Token F1
    - Retrieval Recall

Important:
    NewsQA can contain multiple valid answer strings.
    EM/F1 are therefore computed against every valid answer,
    and the best score is used.
"""

import os
import sys
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.retriever import BM25Retriever
from src.ralm import InContextRALM
from src.evaluate import (
    exact_match,
    token_f1,
    retrieval_recall,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "hlt-lab/newsqa-safe"

N_EVAL = 1000

TOP_K_VALUES = [1, 3, 5, 7]

TOTAL_BUDGET = 350

OUTPUT_DETAILED = (
    "results/newsqa_context_allocation_detailed_1000.csv"
)

OUTPUT_SUMMARY = (
    "results/newsqa_context_allocation_summary.csv"
)

RANDOM_SEED = 42

# ============================================================
# ALLOCATION STRATEGIES
# ============================================================

ALLOCATIONS = {
    "uniform": {
        1: [350],
        3: [117, 117, 116],
        5: [70, 70, 70, 70, 70],
        7: [50, 50, 50, 50, 50, 50, 50],
    },

    "weighted": {
        1: [350],
        3: [150, 110, 90],
        5: [100, 80, 65, 55, 50],
        7: [100, 70, 50, 40, 35, 30, 25],
    },

    "top_heavy": {
        1: [350],
        3: [190, 90, 70],
        5: [150, 80, 50, 40, 30],
        7: [140, 70, 45, 35, 25, 20, 15],
    },
}


# ============================================================
# VALIDATE ALLOCATIONS
# ============================================================

for strategy, strategy_allocations in ALLOCATIONS.items():

    for k, budgets in strategy_allocations.items():

        assert len(budgets) == k, (
            f"{strategy}, k={k}: "
            f"expected {k} budgets, got {len(budgets)}"
        )

        assert sum(budgets) == TOTAL_BUDGET, (
            f"{strategy}, k={k}: "
            f"budgets sum to {sum(budgets)}, "
            f"expected {TOTAL_BUDGET}"
        )

        assert all(
            budget > 0 for budget in budgets
        ), f"{strategy}, k={k}: non-positive budget found"


# ============================================================
# HELPER: EXTRACT TEXT
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return str(value)


# ============================================================
# HELPER: EXTRACT ANSWERS
# ============================================================

def extract_gold_answers(example):
    """
    Convert the NewsQA answer field into a clean list of
    valid answer strings.

    NewsQA may represent answers in different structures,
    so this function handles common cases.
    """

    answers = example.get("answers", None)

    if answers is None:
        answers = example.get("answer", None)

    if answers is None:
        return []

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(answers, str):

        answer = answers.strip()

        return [answer] if answer else []

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(answers, list):

        cleaned = []

        for answer in answers:

            if isinstance(answer, str):

                answer = answer.strip()

                if answer:
                    cleaned.append(answer)

            elif isinstance(answer, dict):

                # Common possibility:
                # {"text": "..."} or {"answer": "..."}

                text = answer.get("text")

                if text is None:
                    text = answer.get("answer")

                if text is not None:

                    text = str(text).strip()

                    if text:
                        cleaned.append(text)

        return cleaned

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(answers, dict):

        # HuggingFace-style structure:
        # {"text": [...], "answer_start": [...]}

        text_values = answers.get("text")

        if text_values is not None:

            if isinstance(text_values, str):

                text_values = [text_values]

            if isinstance(text_values, list):

                return [
                    str(x).strip()
                    for x in text_values
                    if str(x).strip()
                ]

        # Alternative structure

        answer_values = answers.get("answer")

        if answer_values is not None:

            if isinstance(answer_values, str):

                return [answer_values.strip()]

            if isinstance(answer_values, list):

                return [
                    str(x).strip()
                    for x in answer_values
                    if str(x).strip()
                ]

    return []


# ============================================================
# HELPER: BEST EM/F1
# ============================================================

def score_against_multiple_answers(
    prediction,
    gold_answers
):
    """
    Compute EM and F1 against all valid answers and
    keep the best score.
    """

    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    gold_answers = [
        str(answer).strip()
        for answer in gold_answers
        if str(answer).strip()
    ]

    if not gold_answers:

        return 0, 0.0

    em = max(
        exact_match(
            prediction,
            answer
        )
        for answer in gold_answers
    )

    f1 = max(
        token_f1(
            prediction,
            answer
        )
        for answer in gold_answers
    )

    return em, f1


# ============================================================
# LOAD NEWSQA
# ============================================================

print()
print("=" * 70)
print("LOADING NEWSQA")
print("=" * 70)

dataset = load_dataset(DATASET_NAME)

print()
print("Available splits:")

for split_name in dataset.keys():

    print(
        f"  {split_name}: "
        f"{len(dataset[split_name])} examples"
    )


# ============================================================
# SELECT SPLIT
# ============================================================

if "validation" in dataset:

    split_name = "validation"

elif "dev" in dataset:

    split_name = "dev"

elif "test" in dataset:

    split_name = "test"

else:

    split_name = "train"

print()
print(f"Using split: {split_name}")

data = dataset[split_name]


# ============================================================
# INSPECT SCHEMA
# ============================================================

print()
print("Dataset columns:")
print(data.column_names)

print()
print("First example:")
print(data[0])


# ============================================================
# FILTER VALID EXAMPLES
# ============================================================

valid_examples = []

for example in data:

    context = example.get("context")

    question = example.get("question")

    gold_answers = extract_gold_answers(example)

    if context is None:
        continue

    if question is None:
        continue

    context = safe_text(context).strip()

    question = safe_text(question).strip()

    if not context:
        continue

    if not question:
        continue

    if not gold_answers:
        continue

    valid_examples.append(
        {
            "question": question,
            "context": context,
            "answers": gold_answers,
            "id": example.get(
                "id",
                example.get(
                    "qid",
                    len(valid_examples)
                )
            ),
        }
    )


print()
print(
    f"Valid examples available: "
    f"{len(valid_examples)}"
)


# ============================================================
# LIMIT EVALUATION SET
# ============================================================

random.seed(RANDOM_SEED)

eval_examples = valid_examples[:N_EVAL]

print(
    f"Evaluation examples: "
    f"{len(eval_examples)}"
)


assert len(eval_examples) == N_EVAL, (
    f"Expected {N_EVAL} valid examples, "
    f"but only found {len(eval_examples)}"
)


# ============================================================
# RETRIEVAL CORPUS
# ============================================================

# Use the complete selected split as retrieval corpus.

retrieval_corpus = [
    example["context"]
    for example in valid_examples
]

print()
print(
    f"Retrieval corpus size: "
    f"{len(retrieval_corpus)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = InContextRALM()

print("Model loaded.")


# ============================================================
# RETRIEVAL CACHE
# ============================================================

retrieval_cache = {}

# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_results = []

print()
print("=" * 70)
print("STARTING NEWSQA EXPERIMENT")
print("=" * 70)


for top_k in TOP_K_VALUES:

    print()
    print("=" * 70)
    print(f"TOP-K = {top_k}")
    print("=" * 70)

    # --------------------------------------------------------
    # BM25 RETRIEVAL
    # --------------------------------------------------------

    print()
    print("Retrieving documents...")

    retriever = BM25Retriever(
        retrieval_corpus
    )

    retrieved_documents = []

    for example in tqdm(
        eval_examples,
        desc=f"BM25 retrieval k={top_k}"
    ):

        question = example["question"]

        documents = retriever.retrieve(
            question,
            k=top_k
        )

        retrieved_documents.append(
            documents
        )

    retrieval_cache[top_k] = retrieved_documents

    # --------------------------------------------------------
    # CHECK RETRIEVAL RECALL
    # --------------------------------------------------------

    recall_values = []

    for idx, example in enumerate(
        eval_examples
    ):

        gold_context = example["context"]

        documents = retrieved_documents[idx]

        recall = retrieval_recall(
            documents,
            gold_context
        )

        recall_values.append(recall)

    retrieval_recall_value = float(
        np.mean(recall_values)
    )

    print(
        f"Retrieval Recall@{top_k}: "
        f"{retrieval_recall_value:.4f}"
    )

    # --------------------------------------------------------
    # ALLOCATION STRATEGIES
    # --------------------------------------------------------

    for allocation_name in [
        "uniform",
        "weighted",
        "top_heavy"
    ]:

        budgets = ALLOCATIONS[
            allocation_name
        ][top_k]

        print()
        print(
            f"Allocation: "
            f"{allocation_name}"
        )

        print(
            f"Budgets: "
            f"{budgets}"
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        for idx, example in enumerate(
            tqdm(
                eval_examples,
                desc=(
                    f"{allocation_name} "
                    f"k={top_k}"
                )
            )
        ):

            question = example["question"]

            gold_context = example["context"]

            gold_answers = example["answers"]

            documents = retrieved_documents[idx]

            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            prediction = model.generate_ralm(
                question=question,
                documents=documents,
                doc_token_budgets=budgets
            )

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            em, f1 = score_against_multiple_answers(
                prediction,
                gold_answers
            )

            # ------------------------------------------------
            # Retrieval recall
            # ------------------------------------------------

            recall = retrieval_recall(
                documents,
                gold_context
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            all_results.append(
                {
                    "sample_index": idx,

                    "question_id": example["id"],

                    "top_k": top_k,

                    "allocation": allocation_name,

                    "document_budgets": json.dumps(
                        budgets
                    ),

                    "exact_match": em,

                    "f1": f1,

                    "retrieval_recall": recall,
                }
            )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# VERIFY RESULT COUNT
# ============================================================

expected_rows = (
    N_EVAL
    * len(TOP_K_VALUES)
    * 3
)

print()
print(
    f"Generated rows: "
    f"{len(results_df)}"
)

print(
    f"Expected rows: "
    f"{expected_rows}"
)

assert len(results_df) == expected_rows


# ============================================================
# VERIFY RETRIEVAL RECALL IS IDENTICAL
# ACROSS ALLOCATION STRATEGIES
# ============================================================

print()
print("=" * 70)
print("VERIFYING RETRIEVAL CONSISTENCY")
print("=" * 70)

recall_check = (
    results_df
    .groupby(
        ["top_k", "allocation"]
    )["retrieval_recall"]
    .mean()
    .reset_index()
)

print()
print(recall_check.to_string(index=False))


for top_k in TOP_K_VALUES:

    values = (
        recall_check[
            recall_check["top_k"] == top_k
        ]["retrieval_recall"]
        .values
    )

    assert np.allclose(
        values,
        values[0]
    ), (
        f"Retrieval recall changed across "
        f"allocations for k={top_k}"
    )


# ============================================================
# SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "top_k",
            "allocation",
            "document_budgets",
        ]
    )
    .agg(
        exact_match=(
            "exact_match",
            "mean"
        ),

        f1=(
            "f1",
            "mean"
        ),

        retrieval_recall=(
            "retrieval_recall",
            "mean"
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_DETAILED,
    index=False
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("NEWSQA EXPERIMENT COMPLETE")
print("=" * 70)

print()

print(
    summary_df.to_string(
        index=False
    )
)

print()

print(
    f"Detailed results saved to:\n"
    f"  {OUTPUT_DETAILED}"
)

print(
    f"Summary saved to:\n"
    f"  {OUTPUT_SUMMARY}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)