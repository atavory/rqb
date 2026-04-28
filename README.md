# Tabular Bandits via Residual Quantization — Algorithm Implementations

This directory contains the algorithm implementations for the paper
"Tabular Bandits via Residual Quantization" (NeurIPS 2026).

## Core Algorithms

### Baselines
| File | Class | Description | Cost/step |
|------|-------|-------------|-----------|
| `nig_stats.py` | `NIGStats` | Context-free Thompson Sampling (NIG conjugate prior) | O(K) |
| `sgd_lints.py` | `SGDLinTS` | Diagonal-precision linear TS (Adagrad-style) | O(dK) |
| `lints.py` | `LinTSBaseline` | Full-covariance linear TS (Sherman-Morrison updates) | O(d²K) |

### RQ Methods (with shadow promotion)
| File | Class | Description | Cost/step |
|------|-------|-------------|-----------|
| `counter_rq.py` | `CounterDRQm` | Scalar counters per (level, centroid, arm). No features. | O(ℓK) |
| `sgd_lints_rq.py` | `SGDLinDRQm` | Diagonal LinTS on residual features per level. | O(ℓdK) |
| `lints_rq.py` | `LinTSDRQm` | Full-covariance LinTS on residual features per level. | O(ℓd²K) |

### Codebook
| File | Function | Description |
|------|----------|-------------|
| `codebook.py` | `train_rq_codebook` | Train FAISS ResidualQuantizer on unlabeled features |
| `codebook.py` | `encode` | Encode features → per-level centroid indices |
| `codebook.py` | `compute_residual_features` | Compute per-level residual vectors |

## Key Design Decisions

- **Shadow promotion**: All RQ methods start at initial depth `min_level` and
  adaptively increase depth via a shadow level that trains passively. Promotion
  is gated by a Bonferroni-corrected paired t-test on prediction error.
- **η-damping**: Scores are summed across levels with `η^(i-1)` weighting
  (default η=0.5), analogous to learning rate in gradient boosting.
- **Pseudo-residual targets**: Level i predicts the residual after levels 1..i-1,
  clipped to [0,1] via σ(z) = clip(z + 0.5, 0, 1).
- **Inductive codebook**: Codebook is trained on a separate holdout of unlabeled
  contexts and frozen before the bandit starts.

## Demo

Run the end-to-end demo on MiniBooNE (downloads automatically, cached):

```bash
pip install numpy scipy faiss-cpu
python demo_miniboone.py
```

Options:
```bash
python demo_miniboone.py --n-runs 5          # more seeds
python demo_miniboone.py --max-rounds 10000  # quick test
python demo_miniboone.py --d-cut 2 --nbits 4 # shallower codebook
```

This downloads MiniBooNE from UCI (~80MB), trains an RQ codebook on a 5K
holdout, and runs all 6 methods, printing regret rates, wall-clock timing,
and shadow promotion events.

## Dependencies

```
pip install numpy scipy faiss-cpu
```

No torch required. The demo uses a lightweight numpy NIG implementation
for context-free Thompson Sampling.
