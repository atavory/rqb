# Tabular Bandits via Residual Quantization

Code for "Tabular Bandits via Residual Quantization" (NeurIPS 2026 submission).

## Setup

```bash
pip install -r requirements.txt
```

## Datasets

All 13 datasets are publicly available from OpenML/UCI. Download:
```bash
python3 scripts/download_datasets.py --datasets airlines_delay bng_elevators bng_letter \
    covertype beer_reviews hepmass higgs kddcup99 miniboone poker_hand skin_segmentation \
    susy year_prediction
```

## Reproducing Key Results

### Main bandit experiments (Tables 2-3, Figures 2-5)

Run all 6 methods (TS, SGD-LinTS, LinTS + their RQ variants) on a dataset:
```bash
python3 scripts/experiments/run_real_nts_experiment.py \
    --dataset covertype --feature-mode raw \
    --nbits 4 --auto-dcut --n-rounds 530000 --n-seeds 30
```

### Synthetic crossover experiment (Appendix)

Find the horizon T where deeper RQ beats shallow:
```bash
python3 scripts/synthetic_crossover.py \
    --T 1000000 --seeds 10 --b 16 --D 10 --max-depth 5
```

### Significance tests

```bash
python3 scripts/significance_tests.py --alpha 0.05
```

### Depth diagnostic (quantization error)

```bash
python3 scripts/diagnostic_quant_error.py --datasets covertype higgs susy
```

## Project Structure

```
modules/
  bandits/          # TS-RQ, SGD-LinTS-RQ, LinTS-RQ with shadow promotion
  contrastive/      # RQ codebook training via k-means
  encoders/         # TabNet, SCARF baselines
  data/             # Dataset loading and preprocessing
  features.py       # Feature extraction and normalization
  embeddings.py     # RQ encoding and reconstruction
  significance.py   # Statistical tests (Freedman, Bonferroni)

scripts/
  experiments/      # Main experiment scripts
  download_datasets.py
  aggregate_results.py
  synthetic_crossover.py
  significance_tests.py
  diagnostic_quant_error.py

conf/               # Hydra config files
```

## Core Modules

- `modules/bandits/hierarchical_ts.py` -- RQ Bandit with shadow promotion (Algorithm 1)
- `modules/bandits/nig_stats.py` -- Per-cell sufficient statistics
- `modules/embeddings.py` -- RQ codebook training and greedy encoding
- `modules/significance.py` -- Freedman-based promotion test

## Deploying to GitHub

To push this code to a GitHub repository:

```bash
# 1. Create a repo on https://github.com/new (public, no README)

# 2. From the code_release directory:
cd /tmp && mkdir -p rqb && cd rqb && git init
cp -r /path/to/code_release/* .
git add -A && git commit -m "Initial code release"
git branch -M main
git remote add origin https://github.com/USERNAME/rqb.git
git push -u origin main

# 3. For anonymous submission, go to https://anonymous.4open.science/
#    Paste the repo URL to get an anonymized link for the paper.
```

## License

MIT
