import os
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.retriever import BM25Retriever
from src.ralm import InContextRALM

from src.evaluate import (
    exact_match,
    token_f1,
    retrieval_recall
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EVAL_SAMPLES = 200

TOP_K_VALUES = [
    1,
    3,
    5
]


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING SQuAD DATASET")
print("=" * 70)

dataset = load_dataset(
    "rajpurkar/squad"
)

train_data = dataset["train"]
eval_data = dataset["validation"]

print(
    f"Training examples   : {len(train_data):,}"
)

print(
    f"Validation examples : {len(eval_data):,}"
)


# ============================================================
# BUILD RETRIEVAL CORPUS
# ============================================================

print("\n" + "=" * 70)
print("PREPARING RETRIEVAL CORPUS")
print("=" * 70)

documents = list(
    dict.fromkeys(
        eval_data["context"]
    )
)

print(
    f"Unique corpus documents: {len(documents):,}"
)


# ============================================================
# BUILD BM25 RETRIEVER
# ============================================================

print("\n" + "=" * 70)
print("BUILDING BM25 RETRIEVER")
print("=" * 70)

retriever = BM25Retriever(
    documents
)


# ============================================================
# LOAD LANGUAGE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING LANGUAGE MODEL")
print("=" * 70)

model = InContextRALM(
    model_name="google/flan-t5-small"
)


# ============================================================
# SELECT EVALUATION SAMPLES
# ============================================================

eval_samples = eval_data.select(
    range(
        min(
            NUM_EVAL_SAMPLES,
            len(eval_data)
        )
    )
)

print(
    f"\nUsing {len(eval_samples)} validation examples "
    f"for evaluation."
)


# ============================================================
# BASELINE EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("BASELINE EXPERIMENT")
print("=" * 70)

baseline_em_scores = []
baseline_f1_scores = []


for sample in tqdm(
    eval_samples,
    desc="Baseline"
):

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = sample["question"]

    # --------------------------------------------------------
    # GOLD ANSWERS
    # --------------------------------------------------------

    answers = sample["answers"]["text"]

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    prediction = model.generate_baseline(
        question
    )

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    best_em = max(
        exact_match(
            prediction,
            answer
        )
        for answer in answers
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    best_f1 = max(
        token_f1(
            prediction,
            answer
        )
        for answer in answers
    )

    baseline_em_scores.append(
        best_em
    )

    baseline_f1_scores.append(
        best_f1
    )


# ============================================================
# BASELINE AGGREGATE SCORES
# ============================================================

baseline_em_score = (
    sum(baseline_em_scores)
    / len(baseline_em_scores)
)

baseline_f1_score = (
    sum(baseline_f1_scores)
    / len(baseline_f1_scores)
)


# ============================================================
# RALM EXPERIMENT
# ============================================================

results = []


for k in TOP_K_VALUES:

    print("\n" + "=" * 70)
    print(
        f"IN-CONTEXT RALM — TOP-{k}"
    )
    print("=" * 70)

    em_scores = []
    f1_scores = []
    retrieval_scores = []


    # ========================================================
    # EVALUATE EACH SAMPLE
    # ========================================================

    for sample in tqdm(
        eval_samples,
        desc=f"RALM top-{k}"
    ):

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        question = sample["question"]

        # ----------------------------------------------------
        # GOLD ANSWERS
        # ----------------------------------------------------

        answers = sample["answers"]["text"]

        # ----------------------------------------------------
        # GOLD SQuAD CONTEXT
        #
        # This is the actual context associated with the
        # question in the SQuAD validation set.
        # ----------------------------------------------------

        gold_context = sample["context"]


        # ====================================================
        # RETRIEVAL
        # ====================================================

        retrieved_docs = retriever.retrieve(
            question,
            k=k
        )


        # ====================================================
        # RETRIEVAL EVALUATION
        #
        # Checks whether the actual gold SQuAD context
        # was retrieved in the top-k documents.
        # ====================================================

        retrieval_score = retrieval_recall(
            retrieved_docs,
            gold_context
        )


        # ====================================================
        # RALM GENERATION
        # ====================================================

        prediction = model.generate_ralm(
            question,
            retrieved_docs
        )


        # ====================================================
        # ANSWER EVALUATION
        # ====================================================

        best_em = max(
            exact_match(
                prediction,
                answer
            )
            for answer in answers
        )

        best_f1 = max(
            token_f1(
                prediction,
                answer
            )
            for answer in answers
        )


        # ====================================================
        # STORE SAMPLE RESULTS
        # ====================================================

        em_scores.append(
            best_em
        )

        f1_scores.append(
            best_f1
        )

        retrieval_scores.append(
            retrieval_score
        )


    # ========================================================
    # AGGREGATE RALM RESULTS
    # ========================================================

    results.append(
        {
            "system": "In-Context RALM",

            "top_k": k,

            "exact_match": (
                sum(em_scores)
                / len(em_scores)
            ),

            "f1": (
                sum(f1_scores)
                / len(f1_scores)
            ),

            "retrieval_recall": (
                sum(retrieval_scores)
                / len(retrieval_scores)
            )
        }
    )


# ============================================================
# ADD BASELINE TO RESULTS
# ============================================================

results.insert(
    0,
    {
        "system": "Baseline",

        "top_k": 0,

        "exact_match": baseline_em_score,

        "f1": baseline_f1_score,

        "retrieval_recall": None
    }
)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/results.csv",
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")

print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(
    "results/results.csv"
)

print("\nExperiment completed successfully.")