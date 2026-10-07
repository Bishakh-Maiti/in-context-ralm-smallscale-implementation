# experiments/context_budget_experiment.py

import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.retriever import BM25Retriever
from src.ralm import InContextRALM
from src.evaluate import exact_match, token_f1, retrieval_recall


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EVAL_SAMPLES = 1000

TOP_K_VALUES = [1, 3, 5, 7]

TOTAL_CONTEXT_BUDGET = 350

MODEL_NAME = "google/flan-t5-small"

# Aggregate results
AGGREGATE_OUTPUT = (
    "results/context_allocation_ablation_1000.csv"
)

# Per-question results
DETAILED_OUTPUT = (
    "results/context_allocation_detailed_1000.csv"
)


# ============================================================
# CONTEXT ALLOCATION STRATEGIES
# ============================================================

def get_uniform_budgets(k, total_budget):
    """
    Divide the total context budget as equally as possible.
    """

    base = total_budget // k
    remainder = total_budget % k

    budgets = [base] * k

    for i in range(remainder):
        budgets[i] += 1

    return budgets


def get_weighted_budgets(k, total_budget):
    """
    Moderate rank-aware allocation.

    Higher-ranked BM25 documents receive more tokens.
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


def get_top_heavy_budgets(k, total_budget):
    """
    Strong rank-aware allocation.

    Higher-ranked documents receive substantially
    more context.
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


def get_allocation_strategies(k):

    return {
        "uniform": get_uniform_budgets(
            k,
            TOTAL_CONTEXT_BUDGET
        ),

        "weighted": get_weighted_budgets(
            k,
            TOTAL_CONTEXT_BUDGET
        ),

        "top_heavy": get_top_heavy_budgets(
            k,
            TOTAL_CONTEXT_BUDGET
        ),
    }


# ============================================================
# VALIDATE ALLOCATIONS
# ============================================================

def validate_allocations():

    print("\n" + "=" * 70)
    print("CONTEXT ALLOCATION VALIDATION")
    print("=" * 70)

    for k in TOP_K_VALUES:

        strategies = get_allocation_strategies(k)

        for strategy_name, budgets in strategies.items():

            assert len(budgets) == k

            assert sum(budgets) == TOTAL_CONTEXT_BUDGET

            assert all(
                budget > 0
                for budget in budgets
            )

            print(
                f"k={k:<2} "
                f"{strategy_name:<10} "
                f"{budgets} "
                f"sum={sum(budgets)}"
            )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RALM CONTEXT ALLOCATION EXPERIMENT")
    print("=" * 70)

    print(
        f"Evaluation samples    : {NUM_EVAL_SAMPLES}"
    )

    print(
        f"Top-k values          : {TOP_K_VALUES}"
    )

    print(
        f"Total context budget : {TOTAL_CONTEXT_BUDGET}"
    )

    print(
        f"Model                 : {MODEL_NAME}"
    )

    print(
        "Retrieval corpus      : FULL SQuAD validation contexts"
    )

    print("=" * 70)

    # ========================================================
    # VALIDATE ALLOCATIONS
    # ========================================================

    validate_allocations()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading SQuAD...")

    dataset = load_dataset(
        "rajpurkar/squad"
    )

    validation_data = dataset["validation"]

    print(
        f"Full validation set: "
        f"{len(validation_data)} questions"
    )

    # ========================================================
    # BUILD FULL RETRIEVAL CORPUS
    # ========================================================

    print("\nBuilding FULL retrieval corpus...")

    corpus = list(
        dict.fromkeys(
            sample["context"]
            for sample in validation_data
        )
    )

    print(
        f"Unique retrieval documents: "
        f"{len(corpus)}"
    )

    # ========================================================
    # SELECT EVALUATION DATA
    # ========================================================

    print(
        f"\nSelecting first "
        f"{NUM_EVAL_SAMPLES} evaluation samples..."
    )

    eval_data = validation_data.select(
        range(NUM_EVAL_SAMPLES)
    )

    print(
        f"Evaluation samples: "
        f"{len(eval_data)}"
    )

    # ========================================================
    # INITIALIZE RETRIEVER
    # ========================================================

    print("\nInitializing BM25 retriever...")

    retriever = BM25Retriever(corpus)

    print("BM25 ready.")

    # ========================================================
    # INITIALIZE MODEL
    # ========================================================

    print("\nLoading FLAN-T5...")

    model = InContextRALM(
        model_name=MODEL_NAME
    )

    print("Model ready.")

    # ========================================================
    # PRECOMPUTE RETRIEVAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PRECOMPUTING RETRIEVAL")
    print("=" * 70)

    retrieval_cache = {}

    retrieval_recall_cache = {}

    for k in TOP_K_VALUES:

        print(
            f"\nComputing retrieval for top-{k}..."
        )

        retrieval_cache[k] = []

        retrieval_recall_cache[k] = []

        for sample in tqdm(
            eval_data,
            desc=f"Retrieval top-{k}"
        ):

            question = sample["question"]

            gold_context = sample["context"]

            retrieved_docs = retriever.retrieve(
                question,
                k=k
            )

            retrieval_cache[k].append(
                retrieved_docs
            )

            recall = retrieval_recall(
                retrieved_docs,
                gold_context
            )

            retrieval_recall_cache[k].append(
                recall
            )

    # ========================================================
    # CALCULATE RETRIEVAL RECALL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RETRIEVAL SANITY CHECK")
    print("=" * 70)

    average_recall = {}

    for k in TOP_K_VALUES:

        recall = (
            sum(
                retrieval_recall_cache[k]
            )
            /
            len(
                retrieval_recall_cache[k]
            )
        )

        average_recall[k] = recall

        print(
            f"Top-{k}: "
            f"Retrieval Recall = {recall:.4f}"
        )

    print("=" * 70)

    # ========================================================
    # RUN GENERATION EXPERIMENT
    # ========================================================

    aggregate_results = []

    detailed_results = []

    print("\n")
    print("=" * 70)
    print("RUNNING CONTEXT ALLOCATION EXPERIMENT")
    print("=" * 70)

    for k in TOP_K_VALUES:

        strategies = get_allocation_strategies(k)

        for strategy_name, doc_budgets in strategies.items():

            print("\n")
            print("-" * 70)

            print(
                f"k={k} | "
                f"Allocation={strategy_name} | "
                f"Budgets={doc_budgets}"
            )

            print("-" * 70)

            exact_match_scores = []

            f1_scores = []

            # ------------------------------------------------
            # GENERATE ANSWERS
            # ------------------------------------------------

            for sample_idx, sample in enumerate(
                tqdm(
                    eval_data,
                    desc=f"k={k} | {strategy_name}"
                )
            ):

                question = sample["question"]

                gold_answers = sample["answers"]["text"]

                retrieved_docs = retrieval_cache[k][
                    sample_idx
                ]

                # --------------------------------------------
                # GENERATE
                # --------------------------------------------

                prediction = model.generate_ralm(
                    question,
                    retrieved_docs,
                    doc_token_budgets=doc_budgets
                )

                # --------------------------------------------
                # EVALUATE
                # --------------------------------------------

                em = max(
                    exact_match(
                        prediction,
                        gold_answer
                    )
                    for gold_answer in gold_answers
                )

                f1 = max(
                    token_f1(
                        prediction,
                        gold_answer
                    )
                    for gold_answer in gold_answers
                )

                exact_match_scores.append(em)

                f1_scores.append(f1)

                # --------------------------------------------
                # SAVE PER-QUESTION RESULT
                # --------------------------------------------

                detailed_results.append(
                    {
                        "sample_index": sample_idx,

                        "question_id":
                            sample["id"],

                        "top_k": k,

                        "allocation":
                            strategy_name,

                        "document_budgets":
                            str(doc_budgets),

                        "exact_match": em,

                        "f1": f1,

                        "retrieval_recall":
                            retrieval_recall_cache[k][
                                sample_idx
                            ],
                    }
                )

            # ------------------------------------------------
            # AGGREGATE
            # ------------------------------------------------

            avg_em = (
                sum(exact_match_scores)
                /
                len(exact_match_scores)
            )

            avg_f1 = (
                sum(f1_scores)
                /
                len(f1_scores)
            )

            avg_recall = average_recall[k]

            # ------------------------------------------------
            # SAVE AGGREGATE RESULT
            # ------------------------------------------------

            aggregate_results.append(
                {
                    "system":
                        "In-Context RALM",

                    "top_k":
                        k,

                    "allocation":
                        strategy_name,

                    "total_context_budget":
                        TOTAL_CONTEXT_BUDGET,

                    "document_budgets":
                        str(doc_budgets),

                    "exact_match":
                        avg_em,

                    "f1":
                        avg_f1,

                    "retrieval_recall":
                        avg_recall,
                }
            )

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print("\nRESULT")

            print(
                f"Allocation       : "
                f"{strategy_name}"
            )

            print(
                f"Top-k             : "
                f"{k}"
            )

            print(
                f"Document budgets  : "
                f"{doc_budgets}"
            )

            print(
                f"Exact Match       : "
                f"{avg_em:.4f}"
            )

            print(
                f"F1                : "
                f"{avg_f1:.4f}"
            )

            print(
                f"Retrieval Recall  : "
                f"{avg_recall:.4f}"
            )

    # ========================================================
    # CONVERT TO DATAFRAMES
    # ========================================================

    aggregate_df = pd.DataFrame(
        aggregate_results
    )

    detailed_df = pd.DataFrame(
        detailed_results
    )

    # ========================================================
    # FINAL RETRIEVAL CONSISTENCY CHECK
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RETRIEVAL CONSISTENCY CHECK")
    print("=" * 70)

    for k in TOP_K_VALUES:

        recalls = aggregate_df[
            aggregate_df["top_k"] == k
        ]["retrieval_recall"].unique()

        print(
            f"k={k}: "
            f"recall values across allocations = "
            f"{recalls}"
        )

        assert len(recalls) == 1, (
            f"Retrieval recall differs across "
            f"allocation strategies for k={k}"
        )

    print(
        "\nPASS: Retrieval recall is identical "
        "across allocation strategies."
    )

    print("=" * 70)

    # ========================================================
    # CHECK DETAILED DATASET SIZE
    # ========================================================

    expected_rows = (
        NUM_EVAL_SAMPLES
        * len(TOP_K_VALUES)
        * 3
    )

    actual_rows = len(detailed_df)

    print("\n")
    print("=" * 70)
    print("DETAILED RESULT VALIDATION")
    print("=" * 70)

    print(
        f"Expected detailed rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual detailed rows  : "
        f"{actual_rows}"
    )

    assert actual_rows == expected_rows

    print(
        "PASS: All per-question results were saved."
    )

    print("=" * 70)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    aggregate_df.to_csv(
        AGGREGATE_OUTPUT,
        index=False
    )

    detailed_df.to_csv(
        DETAILED_OUTPUT,
        index=False
    )

    # ========================================================
    # DISPLAY FINAL AGGREGATE RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL AGGREGATE RESULTS")
    print("=" * 70)

    display_df = aggregate_df.copy()

    display_df["exact_match"] = (
        display_df["exact_match"] * 100
    ).round(2)

    display_df["f1"] = (
        display_df["f1"] * 100
    ).round(2)

    display_df["retrieval_recall"] = (
        display_df["retrieval_recall"] * 100
    ).round(2)

    print(
        display_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Aggregate results:\n"
        f"{AGGREGATE_OUTPUT}"
    )

    print(
        f"\nDetailed per-question results:\n"
        f"{DETAILED_OUTPUT}"
    )

    print("=" * 70)

    print(
        "\nExperiment completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()