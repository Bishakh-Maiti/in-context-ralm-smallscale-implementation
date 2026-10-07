"""
Rank-Aware vs Rank-Shuffled Control Experiment
================================================

Research question
-----------------
Is the performance gain from top-heavy context allocation caused
specifically by assigning larger token budgets to higher-ranked
retrieved documents?

Control condition
-----------------
Keep the exact same top-heavy budget vectors, but randomly shuffle
those budgets across the retrieved BM25 ranks.

Everything else is held constant:
    - same evaluation questions
    - same BM25 retrieval
    - same retrieved documents
    - same 350-token total context budget
    - same frozen FLAN-T5-small
    - same evaluation metrics

Datasets
--------
    1. SQuAD validation
    2. NewsQA-safe

We run k = 3, 5, 7.

k = 1 is intentionally excluded because [350] has no meaningful
permutation.

Outputs
-------
results/rank_shuffled_control_detailed.csv
results/rank_shuffled_control_summary.csv
results/rank_shuffled_control_stats.csv

IMPORTANT
---------
The top-heavy baseline is NOT regenerated here. It is loaded from
the verified final context-allocation experiment results.

Expected baseline files:

    SQuAD:
        results/context_allocation_detailed_1000.csv

    NewsQA:
        results/newsqa_context_allocation_detailed_1000.csv

The NewsQA path above is intentional. Do NOT use the old
results/archives/ path.
"""

import os
import sys
import json
import random

import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import wilcoxon


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

N_EVAL = 1000

TOP_K_VALUES = [3, 5, 7]

TOTAL_BUDGET = 350

MODEL_NAME = "google/flan-t5-small"

SHUFFLE_SEED = 42

BOOTSTRAP_SAMPLES = 5000


# ============================================================
# TOP-HEAVY ALLOCATION
# ============================================================

TOP_HEAVY = {
    3: [190, 90, 70],
    5: [150, 80, 50, 40, 30],
    7: [140, 70, 45, 35, 25, 20, 15],
}


# Validate allocation configuration immediately.
for k, budgets in TOP_HEAVY.items():

    assert len(budgets) == k, (
        f"k={k}: expected {k} budgets, got {len(budgets)}"
    )

    assert sum(budgets) == TOTAL_BUDGET, (
        f"k={k}: budgets sum to {sum(budgets)}, "
        f"expected {TOTAL_BUDGET}"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DETAILED = (
    "results/rank_shuffled_control_detailed.csv"
)

OUTPUT_SUMMARY = (
    "results/rank_shuffled_control_summary.csv"
)

OUTPUT_STATS = (
    "results/rank_shuffled_control_stats.csv"
)


# ============================================================
# DATASET-SPECIFIC BASELINE PATHS
# ============================================================

BASELINE_PATHS = {

    "squad":
        "results/context_allocation_detailed_1000.csv",

    "newsqa":
        "results/newsqa_context_allocation_detailed_1000.csv",
}


# ============================================================
# HELPER: SHUFFLE BUDGETS
# ============================================================

def shuffled_budgets(k, sample_index):
    """
    Create a deterministic random permutation of the top-heavy
    allocation for a particular question.

    The permutation changes between questions while remaining
    completely reproducible.

    Example:

        Original:
            [140, 70, 45, 35, 25, 20, 15]

        Shuffled:
            [25, 140, 15, 70, 35, 20, 45]

    The total budget remains exactly 350.
    """

    budgets = TOP_HEAVY[k].copy()

    rng = random.Random(
        SHUFFLE_SEED
        + (sample_index * 1009)
        + k
    )

    rng.shuffle(budgets)

    assert len(budgets) == k
    assert sum(budgets) == TOTAL_BUDGET

    return budgets


# ============================================================
# HELPER: NEWSQA ANSWER EXTRACTION
# ============================================================

def extract_newsqa_answers(example):
    """
    Convert the different possible NewsQA answer formats into:

        list[str]
    """

    answers = example.get(
        "answers",
        None,
    )

    if answers is None:
        answers = example.get(
            "answer",
            None,
        )

    if answers is None:
        return []


    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(answers, str):

        answers = answers.strip()

        if answers:
            return [answers]

        return []


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

                text = answer.get(
                    "text",
                    None,
                )

                if text is None:
                    text = answer.get(
                        "answer",
                        None,
                    )

                if text is not None:

                    text = str(text).strip()

                    if text:
                        cleaned.append(text)

        return cleaned


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(answers, dict):

        text_values = answers.get(
            "text",
            None,
        )

        if text_values is not None:

            if isinstance(
                text_values,
                str,
            ):
                text_values = [text_values]

            if isinstance(
                text_values,
                list,
            ):

                return [
                    str(x).strip()
                    for x in text_values
                    if str(x).strip()
                ]


        answer_values = answers.get(
            "answer",
            None,
        )

        if answer_values is not None:

            if isinstance(
                answer_values,
                str,
            ):
                answer_values = [
                    answer_values
                ]

            if isinstance(
                answer_values,
                list,
            ):

                return [
                    str(x).strip()
                    for x in answer_values
                    if str(x).strip()
                ]


    return []


# ============================================================
# HELPER: SCORE MULTIPLE GOLD ANSWERS
# ============================================================

def score_multiple_answers(
    prediction,
    gold_answers,
):
    """
    When multiple gold answers exist, use the best EM and F1.
    """

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
            answer,
        )
        for answer in gold_answers
    )


    f1 = max(
        token_f1(
            prediction,
            answer,
        )
        for answer in gold_answers
    )


    return em, f1


# ============================================================
# HELPER: PAIRED BOOTSTRAP CI
# ============================================================

def bootstrap_ci(
    differences,
    n_boot=BOOTSTRAP_SAMPLES,
    seed=42,
):
    """
    Paired bootstrap 95% confidence interval for the
    mean difference.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        n_boot,
        dtype=float,
    )


    for i in range(n_boot):

        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_means[i] = np.mean(
            sample
        )


    return (
        float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
    )


# ============================================================
# HELPER: HOLM CORRECTION
# ============================================================

def holm_correction(
    p_values,
):
    """
    Holm-Bonferroni correction.

    Returns corrected p-values in the original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    order = np.argsort(
        p_values
    )

    sorted_p = p_values[
        order
    ]

    adjusted = np.empty(
        m,
        dtype=float,
    )


    running_max = 0.0


    for i, p in enumerate(
        sorted_p
    ):

        corrected = (
            m - i
        ) * p

        running_max = max(
            running_max,
            corrected,
        )

        adjusted[i] = min(
            running_max,
            1.0,
        )


    result = np.empty(
        m,
        dtype=float,
    )

    result[order] = adjusted

    return result


# ============================================================
# LOAD SQUAD
# ============================================================

def load_squad():

    print(
        "\nLoading SQuAD..."
    )

    dataset = load_dataset(
        "rajpurkar/squad"
    )

    validation = dataset[
        "validation"
    ]


    # Full validation contexts form
    # the retrieval corpus.
    corpus = list(
        dict.fromkeys(
            sample["context"]
            for sample in validation
        )
    )


    if len(validation) < N_EVAL:

        raise ValueError(
            f"SQuAD validation contains "
            f"only {len(validation)} examples; "
            f"{N_EVAL} required."
        )


    eval_data = validation.select(
        range(N_EVAL)
    )


    print(
        f"SQuAD corpus contexts : "
        f"{len(corpus)}"
    )

    print(
        f"SQuAD evaluation       : "
        f"{len(eval_data)}"
    )


    return (
        corpus,
        eval_data,
    )


# ============================================================
# LOAD NEWSQA
# ============================================================

def load_newsqa():

    print(
        "\nLoading NewsQA..."
    )

    dataset = load_dataset(
        "hlt-lab/newsqa-safe"
    )


    # Prefer a held-out split if the dataset
    # provides one.
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
        f"NewsQA selected split : "
        f"{selected_split}"
    )


    valid_examples = []


    for example in data:

        context = example.get(
            "context"
        )

        question = example.get(
            "question"
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
                len(valid_examples),
            ),
        )


        valid_examples.append(
            {
                "question": question,
                "context": context,
                "answers": answers,
                "id": example_id,
            }
        )


    if len(valid_examples) < N_EVAL:

        raise ValueError(
            f"NewsQA contains only "
            f"{len(valid_examples)} valid examples; "
            f"{N_EVAL} required."
        )


    # Same methodology as the original
    # NewsQA context-allocation experiment.
    eval_data = valid_examples[
        :N_EVAL
    ]


    # Full available split forms the retrieval corpus.
    corpus = [
        example["context"]
        for example in valid_examples
    ]


    print(
        f"NewsQA valid examples : "
        f"{len(valid_examples)}"
    )

    print(
        f"NewsQA corpus contexts: "
        f"{len(corpus)}"
    )

    print(
        f"NewsQA evaluation     : "
        f"{len(eval_data)}"
    )


    return (
        corpus,
        eval_data,
    )


# ============================================================
# LOAD AND VALIDATE TOP-HEAVY BASELINE
# ============================================================

def load_top_heavy_baseline(
    dataset_name,
):
    """
    Load the VERIFIED top-heavy results from the corresponding
    context-allocation experiment.

    This function deliberately does NOT silently fall back to
    archived files.

    That prevents accidentally comparing the shuffled control
    against an incompatible experiment.
    """

    if dataset_name not in BASELINE_PATHS:

        raise ValueError(
            f"Unknown dataset: "
            f"{dataset_name}"
        )


    path = BASELINE_PATHS[
        dataset_name
    ]


    print(
        "\nLoading top-heavy baseline:"
    )

    print(
        f"    {path}"
    )


    if not os.path.exists(path):

        raise FileNotFoundError(
            "\nRequired top-heavy baseline "
            "was not found.\n\n"
            f"Dataset : {dataset_name}\n"
            f"Path    : {path}\n\n"
            "Run the corresponding context "
            "allocation experiment first, "
            "then rerun this control experiment."
        )


    df = pd.read_csv(
        path
    )


    required_columns = {
        "sample_index",
        "top_k",
        "allocation",
        "exact_match",
        "f1",
    }


    missing = (
        required_columns
        - set(df.columns)
    )


    if missing:

        raise ValueError(
            f"{dataset_name} baseline is "
            f"missing columns: "
            f"{sorted(missing)}"
        )


    df = df[
        df["allocation"]
        == "top_heavy"
    ].copy()


    df = df[
        df["top_k"].isin(
            TOP_K_VALUES
        )
    ].copy()


    expected_rows = (
        N_EVAL
        * len(TOP_K_VALUES)
    )


    if len(df) != expected_rows:

        raise ValueError(
            f"{dataset_name} baseline "
            f"contains {len(df)} top-heavy rows; "
            f"expected {expected_rows}."
        )


    # --------------------------------------------------------
    # Validate every k has exactly 1000 questions.
    # --------------------------------------------------------

    counts = (
        df.groupby(
            "top_k"
        )["sample_index"]
        .nunique()
    )


    for k in TOP_K_VALUES:

        count = int(
            counts.get(
                k,
                0,
            )
        )

        if count != N_EVAL:

            raise ValueError(
                f"{dataset_name} baseline "
                f"k={k}: expected "
                f"{N_EVAL} unique questions, "
                f"found {count}."
            )


    # --------------------------------------------------------
    # Validate the expected allocation.
    # --------------------------------------------------------

    if "document_budgets" in df.columns:

        for k in TOP_K_VALUES:

            subset = df[
                df["top_k"] == k
            ]

            expected_budget = json.dumps(
                TOP_HEAVY[k]
            )

            # JSON representation may differ only
            # in whitespace, so compare parsed values.
            for raw_budget in subset[
                "document_budgets"
            ]:

                try:

                    parsed = json.loads(
                        raw_budget
                    )

                except Exception as exc:

                    raise ValueError(
                        f"{dataset_name} k={k}: "
                        f"invalid document_budgets "
                        f"value: {raw_budget}"
                    ) from exc


                if parsed != TOP_HEAVY[k]:

                    raise ValueError(
                        f"{dataset_name} k={k}: "
                        f"baseline allocation is "
                        f"{parsed}, expected "
                        f"{TOP_HEAVY[k]}."
                    )


    baseline = df[
        [
            "sample_index",
            "top_k",
            "exact_match",
            "f1",
        ]
    ].rename(
        columns={
            "exact_match":
                "top_heavy_em",

            "f1":
                "top_heavy_f1",
        }
    )


    # --------------------------------------------------------
    # Print baseline values.
    # --------------------------------------------------------

    print(
        "\nVerified top-heavy baseline:"
    )


    for k in TOP_K_VALUES:

        subset = baseline[
            baseline["top_k"] == k
        ]

        print(
            f"    k={k}: "
            f"EM={subset['top_heavy_em'].mean():.6f}, "
            f"F1={subset['top_heavy_f1'].mean():.6f}"
        )


    return baseline


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(
    dataset_name,
    corpus,
    eval_data,
    model,
):
    """
    Run BM25 retrieval and the rank-shuffled control
    for one dataset.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"RANK-SHUFFLED CONTROL: "
        f"{dataset_name.upper()}"
    )

    print(
        "=" * 80
    )


    print(
        f"Evaluation questions : "
        f"{len(eval_data)}"
    )

    print(
        f"Retrieval corpus     : "
        f"{len(corpus)}"
    )

    print(
        f"k values             : "
        f"{TOP_K_VALUES}"
    )

    print(
        f"Total budget         : "
        f"{TOTAL_BUDGET}"
    )

    print(
        f"Shuffle seed         : "
        f"{SHUFFLE_SEED}"
    )


    # --------------------------------------------------------
    # Build BM25 once.
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    retriever = BM25Retriever(
        corpus
    )

    print(
        "BM25 index built."
    )


    all_results = []


    # ========================================================
    # EACH k
    # ========================================================

    for k in TOP_K_VALUES:

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"RETRIEVAL k={k}"
        )

        print(
            "-" * 80
        )


        # ----------------------------------------------------
        # Retrieve once.
        # ----------------------------------------------------

        retrieved_documents = []

        recall_values = []


        for example in tqdm(
            eval_data,
            desc=f"BM25 retrieval k={k}",
        ):

            question = (
                example["question"]
            )


            documents = (
                retriever.retrieve(
                    question,
                    k=k,
                )
            )


            retrieved_documents.append(
                documents
            )


            recall_values.append(
                retrieval_recall(
                    documents,
                    example["context"],
                )
            )


        avg_recall = float(
            np.mean(
                recall_values
            )
        )


        print(
            f"\nRetrieval Recall@{k}: "
            f"{avg_recall:.4f}"
        )


        # ----------------------------------------------------
        # Generate shuffled condition.
        # ----------------------------------------------------

        for idx, example in enumerate(
            tqdm(
                eval_data,
                desc=f"Shuffled k={k}",
            )
        ):

            documents = (
                retrieved_documents[idx]
            )


            budgets = shuffled_budgets(
                k,
                idx,
            )


            # ------------------------------------------------
            # Sanity checks before generation.
            # ------------------------------------------------

            assert len(
                documents
            ) == k


            assert len(
                budgets
            ) == k


            assert sum(
                budgets
            ) == TOTAL_BUDGET


            # ------------------------------------------------
            # Generate.
            # ------------------------------------------------

            prediction = (
                model.generate_ralm(
                    question=example[
                        "question"
                    ],

                    documents=documents,

                    doc_token_budgets=budgets,
                )
            )


            # ------------------------------------------------
            # Gold answers.
            # ------------------------------------------------

            if dataset_name == "newsqa":

                gold_answers = (
                    example["answers"]
                )

            else:

                gold_answers = (
                    example["answers"]["text"]
                )


            # ------------------------------------------------
            # Metrics.
            # ------------------------------------------------

            em, f1 = (
                score_multiple_answers(
                    prediction,
                    gold_answers,
                )
            )


            recall = (
                retrieval_recall(
                    documents,
                    example["context"],
                )
            )


            # ------------------------------------------------
            # Store.
            # ------------------------------------------------

            all_results.append(
                {
                    "dataset":
                        dataset_name,

                    "sample_index":
                        idx,

                    "question_id":
                        example["id"],

                    "top_k":
                        k,

                    "allocation":
                        "top_heavy_shuffled",

                    "document_budgets":
                        json.dumps(
                            budgets
                        ),

                    "exact_match":
                        em,

                    "f1":
                        f1,

                    "retrieval_recall":
                        recall,
                }
            )


    return pd.DataFrame(
        all_results
    )


# ============================================================
# PAIRED STATISTICAL COMPARISON
# ============================================================

def compare_with_top_heavy(
    shuffled_df,
    dataset_name,
):
    """
    Compare top-heavy vs rank-shuffled using paired
    question-level evaluation.

    Top-heavy:
        larger budgets assigned to higher BM25 ranks.

    Shuffled:
        same budgets randomly assigned to ranks.
    """

    baseline = (
        load_top_heavy_baseline(
            dataset_name
        )
    )


    current = shuffled_df[
        shuffled_df["dataset"]
        == dataset_name
    ].copy()


    # --------------------------------------------------------
    # Merge by exact question and k.
    # --------------------------------------------------------

    merged = baseline.merge(
        current[
            [
                "sample_index",
                "top_k",
                "exact_match",
                "f1",
            ]
        ],

        on=[
            "sample_index",
            "top_k",
        ],

        how="inner",

        validate="one_to_one",
    )


    expected_rows = (
        N_EVAL
        * len(TOP_K_VALUES)
    )


    if len(merged) != expected_rows:

        raise ValueError(
            f"{dataset_name}: paired merge "
            f"produced {len(merged)} rows; "
            f"expected {expected_rows}."
        )


    rows = []


    # --------------------------------------------------------
    # Per-k paired comparisons.
    # --------------------------------------------------------

    for k in TOP_K_VALUES:

        subset = merged[
            merged["top_k"] == k
        ].copy()


        # ----------------------------------------------------
        # F1 differences.
        # ----------------------------------------------------

        top_heavy_f1 = (
            subset[
                "top_heavy_f1"
            ].to_numpy(
                dtype=float
            )
        )

        shuffled_f1 = (
            subset[
                "f1"
            ].to_numpy(
                dtype=float
            )
        )


        f1_diff = (
            top_heavy_f1
            - shuffled_f1
        )


        # ----------------------------------------------------
        # EM differences.
        # ----------------------------------------------------

        top_heavy_em = (
            subset[
                "top_heavy_em"
            ].to_numpy(
                dtype=float
            )
        )

        shuffled_em = (
            subset[
                "exact_match"
            ].to_numpy(
                dtype=float
            )
        )


        em_diff = (
            top_heavy_em
            - shuffled_em
        )


        # ----------------------------------------------------
        # Wilcoxon F1.
        # ----------------------------------------------------

        try:

            (
                f1_stat,
                f1_p,
            ) = wilcoxon(
                top_heavy_f1,
                shuffled_f1,
                zero_method="wilcox",
                alternative="two-sided",
            )

        except ValueError:

            f1_stat = np.nan
            f1_p = np.nan


        # ----------------------------------------------------
        # Wilcoxon EM.
        # ----------------------------------------------------

        try:

            (
                em_stat,
                em_p,
            ) = wilcoxon(
                top_heavy_em,
                shuffled_em,
                zero_method="wilcox",
                alternative="two-sided",
            )

        except ValueError:

            em_stat = np.nan
            em_p = np.nan


        # ----------------------------------------------------
        # Bootstrap confidence intervals.
        # ----------------------------------------------------

        f1_ci = bootstrap_ci(
            f1_diff,
            seed=(
                SHUFFLE_SEED
                + k
            ),
        )


        em_ci = bootstrap_ci(
            em_diff,
            seed=(
                SHUFFLE_SEED
                + 100
                + k
            ),
        )


        # ----------------------------------------------------
        # Cohen's dz.
        # ----------------------------------------------------

        f1_std = np.std(
            f1_diff,
            ddof=1,
        )

        em_std = np.std(
            em_diff,
            ddof=1,
        )


        if f1_std > 0:

            f1_dz = (
                np.mean(f1_diff)
                / f1_std
            )

        else:

            f1_dz = np.nan


        if em_std > 0:

            em_dz = (
                np.mean(em_diff)
                / em_std
            )

        else:

            em_dz = np.nan


        # ----------------------------------------------------
        # Save row.
        # ----------------------------------------------------

        rows.append(
            {
                "dataset":
                    dataset_name,

                "top_k":
                    k,

                "top_heavy_f1":
                    float(
                        np.mean(
                            top_heavy_f1
                        )
                    ),

                "shuffled_f1":
                    float(
                        np.mean(
                            shuffled_f1
                        )
                    ),

                "top_heavy_minus_shuffled_f1":
                    float(
                        np.mean(
                            f1_diff
                        )
                    ),

                "f1_ci_low":
                    f1_ci[0],

                "f1_ci_high":
                    f1_ci[1],

                "f1_wilcoxon_stat":
                    f1_stat,

                "f1_wilcoxon_p":
                    f1_p,

                "f1_cohens_dz":
                    f1_dz,

                "top_heavy_em":
                    float(
                        np.mean(
                            top_heavy_em
                        )
                    ),

                "shuffled_em":
                    float(
                        np.mean(
                            shuffled_em
                        )
                    ),

                "top_heavy_minus_shuffled_em":
                    float(
                        np.mean(
                            em_diff
                        )
                    ),

                "em_ci_low":
                    em_ci[0],

                "em_ci_high":
                    em_ci[1],

                "em_wilcoxon_stat":
                    em_stat,

                "em_wilcoxon_p":
                    em_p,

                "em_cohens_dz":
                    em_dz,

                "paired_questions":
                    len(subset),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RANK-AWARE vs RANK-SHUFFLED CONTROL"
    )

    print(
        "=" * 80
    )


    print(
        "\nResearch question:"
    )

    print(
        "Does assigning larger token budgets to higher-ranked "
        "BM25 documents specifically cause the top-heavy gain?"
    )


    print(
        "\nExperimental controls:"
    )

    print(
        "    Total context budget : 350 tokens"
    )

    print(
        "    Model                : "
        f"{MODEL_NAME}"
    )

    print(
        "    Questions/dataset    : "
        f"{N_EVAL}"
    )

    print(
        "    k values             : "
        f"{TOP_K_VALUES}"
    )

    print(
        "    Shuffle seed         : "
        f"{SHUFFLE_SEED}"
    )

    print(
        "\nk=1 is intentionally excluded because "
        "[350] cannot be meaningfully shuffled."
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Loading FLAN-T5-small..."
    )

    print(
        "-" * 80
    )


    model = InContextRALM(
        model_name=MODEL_NAME
    )


    print(
        "Model ready."
    )


    # ========================================================
    # SQUAD
    # ========================================================

    (
        squad_corpus,
        squad_eval,
    ) = load_squad()


    squad_results = run_dataset(
        dataset_name="squad",

        corpus=squad_corpus,

        eval_data=squad_eval,

        model=model,
    )


    # ========================================================
    # NEWSQA
    # ========================================================

    (
        newsqa_corpus,
        newsqa_eval,
    ) = load_newsqa()


    newsqa_results = run_dataset(
        dataset_name="newsqa",

        corpus=newsqa_corpus,

        eval_data=newsqa_eval,

        model=model,
    )


    # ========================================================
    # COMBINE CONTROL RESULTS
    # ========================================================

    shuffled_results = pd.concat(
        [
            squad_results,
            newsqa_results,
        ],
        ignore_index=True,
    )


    expected_rows = (
        2
        * N_EVAL
        * len(TOP_K_VALUES)
    )


    assert len(
        shuffled_results
    ) == expected_rows, (
        f"Expected {expected_rows} "
        f"rows, got "
        f"{len(shuffled_results)}"
    )


    # ========================================================
    # VALIDATE CONTROL OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VALIDATING CONTROL RESULTS"
    )

    print(
        "=" * 80
    )


    # Every dataset/k combination should
    # contain exactly 1000 questions.

    counts = (
        shuffled_results
        .groupby(
            [
                "dataset",
                "top_k",
            ]
        )["sample_index"]
        .nunique()
    )


    for dataset_name in [
        "squad",
        "newsqa",
    ]:

        for k in TOP_K_VALUES:

            count = int(
                counts.get(
                    (
                        dataset_name,
                        k,
                    ),
                    0,
                )
            )


            assert count == N_EVAL, (
                f"{dataset_name} k={k}: "
                f"expected {N_EVAL} "
                f"questions, got {count}"
            )


    # Validate every shuffled budget vector.
    for _, row in (
        shuffled_results.iterrows()
    ):

        k = int(
            row["top_k"]
        )

        budgets = json.loads(
            row[
                "document_budgets"
            ]
        )


        assert len(
            budgets
        ) == k


        assert sum(
            budgets
        ) == TOTAL_BUDGET


        assert sorted(
            budgets
        ) == sorted(
            TOP_HEAVY[k]
        )


    print(
        "All validation checks passed."
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        shuffled_results
        .groupby(
            [
                "dataset",
                "top_k",
                "document_budgets",
            ],
            as_index=False,
        )
        .agg(
            exact_match=(
                "exact_match",
                "mean",
            ),

            f1=(
                "f1",
                "mean",
            ),

            retrieval_recall=(
                "retrieval_recall",
                "mean",
            ),
        )
    )


    # ========================================================
    # STATISTICAL COMPARISON
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "COMPARING TOP-HEAVY vs SHUFFLED"
    )

    print(
        "=" * 80
    )


    squad_stats = (
        compare_with_top_heavy(
            shuffled_results,
            "squad",
        )
    )


    newsqa_stats = (
        compare_with_top_heavy(
            shuffled_results,
            "newsqa",
        )
    )


    stats = pd.concat(
        [
            squad_stats,
            newsqa_stats,
        ],
        ignore_index=True,
    )


    # ========================================================
    # HOLM CORRECTION
    # ========================================================

    print(
        "\nApplying Holm correction..."
    )


    # Six F1 comparisons:
    #
    # SQuAD:  k=3,5,7
    # NewsQA: k=3,5,7
    #
    # Correct all six together.

    stats[
        "f1_holm_p"
    ] = holm_correction(
        stats[
            "f1_wilcoxon_p"
        ].fillna(1.0)
        .to_numpy()
    )


    # Six EM comparisons.

    stats[
        "em_holm_p"
    ] = holm_correction(
        stats[
            "em_wilcoxon_p"
        ].fillna(1.0)
        .to_numpy()
    )


    # Add significance indicators.

    stats[
        "f1_significant_holm"
    ] = (
        stats["f1_holm_p"]
        < 0.05
    )


    stats[
        "em_significant_holm"
    ] = (
        stats["em_holm_p"]
        < 0.05
    )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )


    shuffled_results.to_csv(
        OUTPUT_DETAILED,
        index=False,
    )


    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )


    stats.to_csv(
        OUTPUT_STATS,
        index=False,
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL TOP-HEAVY vs SHUFFLED RESULTS"
    )

    print(
        "=" * 80
    )


    display_columns = [
        "dataset",
        "top_k",

        "top_heavy_f1",
        "shuffled_f1",
        "top_heavy_minus_shuffled_f1",

        "f1_ci_low",
        "f1_ci_high",

        "f1_holm_p",

        "top_heavy_em",
        "shuffled_em",
        "top_heavy_minus_shuffled_em",

        "em_ci_low",
        "em_ci_high",

        "em_holm_p",
    ]


    print(
        stats[
            display_columns
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # FILES
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 80
    )


    print(
        f"\n{OUTPUT_DETAILED}"
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_STATS
    )


    # ========================================================
    # FINAL SANITY CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL SANITY CHECK"
    )

    print(
        "=" * 80
    )


    for dataset_name in [
        "squad",
        "newsqa",
    ]:

        subset = stats[
            stats["dataset"]
            == dataset_name
        ]


        print(
            f"\n{dataset_name.upper()}"
        )


        for _, row in (
            subset.iterrows()
        ):

            print(
                f"  k={int(row['top_k'])}: "
                f"F1 "
                f"{row['top_heavy_f1']:.4f} "
                f"vs "
                f"{row['shuffled_f1']:.4f}, "
                f"Δ="
                f"{row['top_heavy_minus_shuffled_f1']:+.4f}, "
                f"Holm p="
                f"{row['f1_holm_p']:.3e}"
            )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONTROL EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()