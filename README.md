# Residual Quantization for Contextual Bandits

Code for the paper "Residual Quantization as Structured Uncertainty" (NeurIPS 2026 submission).

## Setup

```bash
pip install -r requirements.txt
```

## Datasets

Download datasets from OpenML:
```bash
python3 scripts/download_datasets.py --datasets adult jannis volkert covertype letter helena
```

## Reproducing Key Results

### Bandit experiments (Table 1-4)

Run the bandit experiment on a dataset with Optuna HP optimization:
```bash
python3 scripts/experiments/run_real_nts_experiment.py \
    --dataset adult --feature-mode raw \
    --nbits 2 --d-cut 1 --n-rounds 100000 --n-seeds 10
```

### K-sweep (Table 2)

```bash
for K in 4 8 16 32 64 128 256; do
    python3 scripts/experiments/run_real_nts_experiment.py \
        --dataset adult --feature-mode raw \
        --km-clusters $K --n-rounds 100000 --n-seeds 10
done
```

### Contrastive learning (Table 7)

```bash
python3 scripts/experiments/run_all_experiments.py \
    --datasets jannis volkert covertype --n-seeds 10
```

### Synthetic crossover experiment

Find the horizon T where deeper RQ beats shallow:
```bash
python3 scripts/synthetic_crossover.py \
    --T 1000000 --seeds 10 --b 4 --D 10 --max-depth 3
```

### Significance tests

```bash
python3 scripts/significance_tests.py --alpha 0.05
```

### Depth diagnostic (quantization error)

```bash
python3 scripts/diagnostic_quant_error.py --datasets adult covertype jannis
```

## Project Structure

```
modules/
  bandits/          # LinTS, HierTS, cold-start sharing
  contrastive/      # m-RQ contrastive learning
  encoders/         # SCARF, ResNet, TabTransformer baselines
  data/             # Dataset loading and preprocessing
  features.py       # Feature extraction, selection, normalization
  embeddings.py     # RQ encoding and reconstruction
  significance.py   # Statistical tests

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

- `modules/bandits/hierarchical_ts.py` — HierTS (O(1) Thompson Sampling on RQ addresses)
- `modules/bandits/nig_stats.py` — Normal-Inverse-Gamma conjugate sufficient statistics
- `modules/embeddings.py` — RQ codebook training and encoding
- `modules/features.py` — Feature extraction with XGBoost importance selection
