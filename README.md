In-Context RALM — Small-Scale Implementation

A small-scale implementation of the core methodology from In-Context Retrieval-Augmented Language Models (Ram et al., TACL 2023).

Note: This is a methodology reproduction / small-scale implementation, not an exact reproduction of the original paper's experimental setup.

Original Paper and Code

Paper: Ram, O., Levine, Y., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., & Shoham, Y. (2023). In-Context Retrieval-Augmented Language Models. Transactions of the Association for Computational Linguistics, 11, 1316–1331.

Paper: https://aclanthology.org/2023.tacl-1.75/

MIT Press: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00605/118118/In-Context-Retrieval-Augmented-Language-Models

Original authors' code: https://github.com/AI21Labs/in-context-ralm

The original authors' repository contains code for their WikiText-103 and QA experiments and is currently archived/read-only.

Overview

Retrieval-Augmented Language Modeling (RALM) provides a language model with relevant external documents during generation.

The key idea of In-Context RALM is to keep the language model architecture unchanged and provide retrieved grounding documents directly in its input context.

Question
   |
   v
BM25 Retriever
   |
   v
Top-k Documents
   |
   v
Context + Question
   |
   v
FLAN-T5-small
   |
   v
Answer

A baseline is also evaluated where FLAN-T5-small receives only the question.

Objective

The objective is to investigate whether adding retrieved textual context to a pretrained language model improves question-answering performance, and to study how the number of retrieved documents affects retrieval and answer quality.

Methodology

Baseline

Question -> FLAN-T5-small -> Answer

Retrieval

BM25 retrieves the top 1, 3, or 5 contexts for each question.

In-Context RALM

Retrieved Documents + Question
              |
              v
       FLAN-T5-small
              |
              v
            Answer

The language model is not fine-tuned for this experiment.

Dataset

The implementation uses SQuAD (Stanford Question Answering Dataset) through Hugging Face:

rajpurkar/squad

The experiment uses SQuAD validation questions and contexts. The final evaluation uses the first 200 validation examples.

Experimental Setup

Component

Configuration

Dataset

SQuAD

Retrieval

BM25

Language model

google/flan-t5-small

Evaluation samples

200

Top-k

1, 3, 5

Baseline

FLAN-T5-small without retrieval

RALM

BM25 contexts + FLAN-T5-small

Answer metrics

Exact Match, Token F1

Retrieval metric

Retrieval Recall@k

Evaluation Metrics

Exact Match

The normalized generated answer must exactly match one of the gold answers.

Token F1

Measures token-level overlap between the generated and gold answers using precision, recall, and their harmonic mean.

Retrieval Recall@k

Measures the proportion of questions for which the gold SQuAD context appears within the top-k retrieved BM25 results.

Final Results

Final evaluation on 200 SQuAD validation examples:

System

Top-k

Exact Match

F1

Retrieval Recall@k

Baseline

—

2.0%

5.90%

—

In-Context RALM

1

35.0%

41.26%

45.5%

In-Context RALM

3

46.0%

52.40%

75.0%

In-Context RALM

5

49.0%

54.06%

84.5%

Complete results are stored in:

results/results.csv

Findings

Increasing retrieval depth improved retrieval coverage:

Recall@1 = 45.5%
Recall@3 = 75.0%
Recall@5 = 84.5%

Answer quality also improved:

Baseline F1 = 5.90%

RALM k=1 F1 = 41.26%
RALM k=3 F1 = 52.40%
RALM k=5 F1 = 54.06%

Within this experimental setup, k=5 produced the highest Exact Match and F1.

These results demonstrate the core idea of retrieval augmentation: providing relevant external textual context can improve question-answering performance of a pretrained language model.

Limitations

This project is a small-scale methodology implementation, not an exact reproduction.

The original paper evaluates language-modeling settings and diverse corpora; this project focuses on SQuAD question answering.

The implementation uses google/flan-t5-small rather than the larger models evaluated in the paper.

Only 200 validation examples are evaluated.

Retrieval uses a straightforward BM25 implementation rather than reproducing all retrieval/reranking mechanisms from the paper.

The language model is not fine-tuned on SQuAD.

Therefore, these results should be interpreted as results from this implementation rather than as a reproduction of the paper's reported numbers.

Project Structure

in-context-ralm-smallscale-implementation/
|
├── experiments/
│   └── run_experiment.py
|
├── results/
│   └── results.csv
|
├── src/
│   ├── evaluate.py
│   ├── ralm.py
│   └── retriever.py
|
├── README.md
└── requirements.txt

Installation

git clone https://github.com/Bishakh-Maiti/in-context-ralm-smallscale-implementation.git
cd in-context-ralm-smallscale-implementation
pip install -r requirements.txt

Running the Experiment

From the project root:

python experiments/run_experiment.py

The script loads SQuAD, builds the BM25 corpus, loads FLAN-T5-small, evaluates the baseline and RALM configurations for k = 1, 3, 5, and saves the results to results/results.csv.

Implementation Details

src/retriever.py — BM25 retrieval

src/ralm.py — baseline and In-Context RALM generation

src/evaluate.py — Exact Match, Token F1, and Retrieval Recall@k

experiments/run_experiment.py — complete experiment pipeline

Relation to the Original Paper

The implementation follows the central concept of the original work: keep the pretrained language model unchanged and incorporate retrieved grounding documents through its input context.

The original paper describes In-Context RALM as a simple RALM framework that prepends retrieved grounding documents to the language-model input without additional language-model training.

This project adapts that idea to a smaller question-answering experiment.

Citation

@article{ram-etal-2023-context,
    title = "In-Context Retrieval-Augmented Language Models",
    author = "Ram, Ori and Levine, Yoav and Dalmedigos, Itay and Muhlgay, Dor and Shashua, Amnon and Leyton-Brown, Kevin and Shoham, Yoav",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "11",
    year = "2023",
    pages = "1316--1331",
    doi = "10.1162/tacl_a_00605"
}

Acknowledgements

This project is based on the methodology introduced in Ram et al., In-Context Retrieval-Augmented Language Models, TACL 2023.

Original research code: AI21 Labs — in-context-ralm