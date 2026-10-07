## 1. Clone the Repository

```bash
git clone https://github.com/Bishakh-Maiti/in-context-ralm.git
cd in-context-ralm

2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate

Linux / macOS
python -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

The required packages are:
datasets
rank-bm25
transformers
torch
sentencepiece
accelerate
pandas
numpy
tqdm
matplotlib

The first run will automatically download the required Hugging Face datasets and
google/flan-t5-small model.
4. Reproduce the Main SQuAD Experiment
Run:
python experiments/run_squad_experiment.py

This reproduces the SQuAD context-allocation experiment using:
- Dataset: rajpurkar/squad
- 1,000 evaluation questions
- Full validation contexts as the retrieval corpus
- BM25 retrieval
- google/flan-t5-small
- Frozen language model
- Total document context budget: 350 tokens
- k = 1, 3, 5, 7
- Uniform, weighted, and top-heavy allocations
Results are saved to:
results/main/squad_results.csv
results/main/squad_summary.csv

5. Reproduce the Main NewsQA Experiment
Run:
python experiments/run_newsqa_experiment.py

This uses:
- Dataset: hlt-lab/newsqa-safe
- 1,000 evaluation questions
- BM25 retrieval
- google/flan-t5-small
- Frozen language model
- Total document context budget: 350 tokens
- k = 1, 3, 5, 7
- Uniform, weighted, and top-heavy allocations
Results are saved to:
results/main/newsqa_results.csv
results/main/newsqa_summary.csv

6. Reproduce the SQuAD Mechanism Analysis
Run:
python experiments/run_squad_mechanism_analysis.py

This analyzes:
- Gold-context retrieval rank
- Answer-span location
- Answer-span survival under each allocation
- F1 by gold-context rank
- Retrieval Recall@7
Results are saved to:
results/mechanism/squad_mechanism_summary.csv
results/mechanism/squad_rank_analysis.csv
results/mechanism/squad_rank_f1_summary.csv

Figures are saved to:
results/figures/gold_context_rank_distribution.png
results/figures/answer_survival_by_allocation.png
results/figures/f1_by_gold_context_rank.png

7. Reproduce the NewsQA Mechanism Analysis
Run:
python experiments/run_newsqa_mechanism_analysis.py

Results are saved to:
results/mechanism/newsqa_mechanism_summary.csv
results/mechanism/newsqa_rank_analysis.csv
results/mechanism/newsqa_rank_f1_summary.csv

Figures are saved to:
results/figures/newsqa_gold_context_rank_distribution.png
results/figures/newsqa_answer_survival_by_allocation.png
results/figures/newsqa_f1_by_gold_context_rank.png

8. Reproduce the Rank-Shuffled Control
Run:
python experiments/run_rank_shuffled_control.py

This runs the rank-shuffled control for:
SQuAD
NewsQA

k = 3, 5, 7

The top-heavy token budgets are randomly shuffled across retrieved-document
ranks while keeping the total 350-token budget unchanged.
The random seed is fixed at:
42

Results are saved to:
results/controls/rank_shuffled_control_detailed.csv
results/controls/rank_shuffled_control_summary.csv
results/controls/rank_shuffled_control_stats.csv

9. Complete Reproduction
Run the experiments in this order:
python experiments/run_squad_experiment.py
python experiments/run_newsqa_experiment.py
python experiments/run_squad_mechanism_analysis.py
python experiments/run_newsqa_mechanism_analysis.py
python experiments/run_rank_shuffled_control.py

All final outputs will be generated under:
results/

10. Model Configuration
The experiments use:
Model: google/flan-t5-small
Input maximum length: 512 tokens
Maximum generated tokens: 64
Beam size: 2

The model automatically uses CUDA when available and otherwise runs on CPU.
11. Context Allocation
All experiments use a fixed total document budget of:
350 tokens

Uniform
k=1: [350]
k=3: [117, 117, 116]
k=5: [70, 70, 70, 70, 70]
k=7: [50, 50, 50, 50, 50, 50, 50]

Weighted
k=1: [350]
k=3: [150, 110, 90]
k=5: [100, 80, 65, 55, 50]
k=7: [100, 70, 50, 40, 35, 30, 25]

Top-heavy
k=1: [350]
k=3: [190, 90, 70]
k=5: [150, 80, 50, 40, 30]
k=7: [140, 70, 45, 35, 25, 20, 15]

12. Reproducibility Notes
The experiments use:
- Fixed evaluation size of 1,000 questions
- Fixed total context budget of 350 tokens
- Fixed allocation strategies
- BM25 retrieval
- Frozen google/flan-t5-small
- Identical retrieved documents across allocation strategies
- Fixed random seed 42 for the rank-shuffled control
No model fine-tuning is performed.
The first execution may take longer because the datasets, tokenizer, and model
must be downloaded and cached locally.