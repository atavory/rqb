#!/usr/bin/env python3


"""Multi-classifier + RQ feature augmentation experiment.

Tests whether RQ trunk codes and trunk reconstructions improve
classification accuracy across multiple classifiers on standard tabular
benchmarks.

Classifiers: XGBoost, LightGBM, RandomForest, MLP, LogReg, KNN,
             FT-Transformer (rtdl), ResNet (rtdl).

For each dataset and RQ configuration (K, d_cut):
1. Load data (already standardized by load_dataset)
2. Train FAISS RQ on training features
3. Extract trunk codes (categorical) and trunk reconstructions (continuous)
4. Run each classifier on:
   (a) Raw features only (baseline)
   (b) Raw + trunk codes
   (c) Raw + trunk reconstructions
   (d) Raw + trunk codes + trunk reconstructions
5. Report accuracy with paired significance tests across seeds

Usage:
    python3 scripts/experiments/run_xgboost_rq.py -- \\
        --datasets adult bank_marketing covertype --seeds 10 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import faiss  # pyre-ignore[21]
import lightgbm as lgb  # pyre-ignore[21]
import numpy as np
import rtdl  # pyre-ignore[21]
import torch
import xgboost as xgb  # pyre-ignore[21]
from modules.data import load_dataset
from modules.features import (
    collect_raw_features,
    normalize_for_rq,
)
from modules.significance import (
    compute_pairwise_significance,
    log_significance,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Recommended (K, d_cut) per dataset from the fast d_cut diagnostic.
# Each entry: (nbits, d_cut) where K = 2^nbits.
# ---------------------------------------------------------------------------
RECOMMENDED_CONFIGS: dict[str, list[tuple[int, int]]] = {
    # (nbits, d_cut) pairs: first is d>1 config, second is d=1 baseline
    "adult": [(3, 3), (6, 1)],
    "bank_marketing": [(3, 3), (6, 1)],
    "covertype": [(3, 5), (6, 1)],
    "higgs": [(3, 4), (6, 1)],
    "helena": [(2, 3), (6, 1)],
    "jannis": [(3, 2), (6, 1)],
    "volkert": [(3, 2), (6, 1)],
    "aloi": [(2, 3), (6, 1)],
    "letter": [(3, 2), (6, 1)],
    "dionis": [(3, 2), (6, 1)],
}

ALL_CLASSIFIER_NAMES: list[str] = [
    "XGBoost",
    "LightGBM",
    "RandomForest",
    "MLP",
    "LogReg",
    "KNN",
    "FTTransformer",
    "ResNet",
]


# ---------------------------------------------------------------------------
# rtdl wrappers — sklearn-compatible .fit() / .predict() interface
# ---------------------------------------------------------------------------
class _RTDLClassifier:
    """Base sklearn-compatible wrapper for rtdl models."""

    def __init__(
        self,
        n_epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        patience: int = 16,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.model: torch.nn.Module | None = None
        self.classes_: np.ndarray | None = None

    def _build_model(self, n_features: int, n_classes: int) -> torch.nn.Module:
        raise NotImplementedError

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RTDLClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        classes = self.classes_
        assert classes is not None
        label_map = {int(c): i for i, c in enumerate(classes)}
        y_mapped = np.array([label_map[int(c)] for c in y])

        # 90/10 train/val split
        n_val = max(1, int(0.1 * len(X)))
        perm = np.random.permutation(len(X))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        dev = self.device
        X_tr = torch.tensor(X[train_idx], dtype=torch.float32, device=dev)
        y_tr = torch.tensor(y_mapped[train_idx], dtype=torch.long, device=dev)
        X_va = torch.tensor(X[val_idx], dtype=torch.float32, device=dev)
        y_va = torch.tensor(y_mapped[val_idx], dtype=torch.long, device=dev)

        self.model = self._build_model(X.shape[1], n_classes).to(dev)
        model = self.model
        assert model is not None
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_loss = float("inf")
        wait = 0
        best_state: dict[str, torch.Tensor] | None = None

        for _ in range(self.n_epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self._forward(bx), by)
                loss.backward()
                optimizer.step()

            # Batched validation loss
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for i in range(0, len(X_va), self.batch_size):
                    bx = X_va[i : i + self.batch_size]
                    by = y_va[i : i + self.batch_size]
                    vl = torch.nn.functional.cross_entropy(self._forward(bx), by)
                    val_loss_sum += vl.item() * len(bx)
            val_loss = val_loss_sum / len(X_va)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        model = self.model
        classes = self.classes_
        assert model is not None and classes is not None
        model.eval()
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                bx = torch.tensor(
                    X[i : i + self.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                preds.append(self._forward(bx).argmax(dim=1).cpu())
        indices = torch.cat(preds).numpy()
        return classes[indices]


class FTTransformerClassifier(_RTDLClassifier):
    """FT-Transformer for tabular classification via rtdl."""

    def _build_model(self, n_features: int, n_classes: int) -> torch.nn.Module:
        return rtdl.FTTransformer.make_default(
            n_num_features=n_features,
            cat_cardinalities=None,
            d_out=n_classes,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(x, None)  # pyre-ignore[29]


class TabResNetClassifier(_RTDLClassifier):
    """ResNet for tabular classification via rtdl."""

    def _build_model(self, n_features: int, n_classes: int) -> torch.nn.Module:
        return rtdl.ResNet.make_baseline(
            d_in=n_features,
            n_blocks=4,
            d_main=256,
            d_hidden=512,
            dropout_first=0.25,
            dropout_second=0.0,
            d_out=n_classes,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(x)  # pyre-ignore[29]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def extract_features(
    dataset: torch.utils.data.Dataset,  # pyre-ignore[2]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw features and labels from a dataset as numpy arrays.

    Delegates to the shared collect_raw_features utility.
    """
    return collect_raw_features(dataset)


def fit_rq_and_augment(
    train_features: np.ndarray,
    test_features: np.ndarray,
    nbits: int,
    d_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit RQ on train features and return trunk codes, reconstructions, residuals."""
    # Standardize before RQ
    train_scaled, test_scaled, _ = normalize_for_rq(train_features, test_features)
    assert test_scaled is not None

    dim = train_scaled.shape[1]
    rq = faiss.ResidualQuantizer(dim, d_cut, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.train(train_scaled)

    # Trunk codes
    train_codes_raw = rq.compute_codes(train_scaled)
    test_codes_raw = rq.compute_codes(test_scaled)
    train_codes = train_codes_raw[:, :d_cut]
    test_codes = test_codes_raw[:, :d_cut]

    # Trunk reconstructions (in scaled space)
    train_recon = rq.decode(train_codes_raw)
    test_recon = rq.decode(test_codes_raw)

    # Quantization residuals: what the trunk misses
    train_residual = (train_scaled - train_recon).astype(np.float32)
    test_residual = (test_scaled - test_recon).astype(np.float32)

    return (
        train_codes,
        test_codes,
        train_recon,
        test_recon,
        train_residual,
        test_residual,
    )


def compute_target_encoding(
    train_code_ids: np.ndarray,
    train_labels: np.ndarray,
    test_code_ids: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """K-fold target encoding: for each trunk code, compute smoothed class probs.

    Uses leave-fold-out on training data to prevent target leakage.
    For test data, uses the full training set.

    Returns (train_tenc, test_tenc) each of shape (N, n_classes).
    """
    global_probs = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        global_probs[c] = np.mean(train_labels == c)

    # Test set: encode using full training data
    test_tenc = np.full((len(test_code_ids), n_classes), global_probs, dtype=np.float32)
    code_to_probs: dict[int, np.ndarray] = {}
    for code in np.unique(train_code_ids):
        mask = train_code_ids == code
        count = int(mask.sum())
        probs = np.array([np.mean(train_labels[mask] == c) for c in range(n_classes)])
        code_to_probs[code] = (probs * count + global_probs * smoothing) / (
            count + smoothing
        )
    for i, code in enumerate(test_code_ids):
        code_int = int(code)
        if code_int in code_to_probs:
            test_tenc[i] = code_to_probs[code_int].astype(np.float32)

    # Train set: k-fold to avoid leakage
    train_tenc = np.full(
        (len(train_code_ids), n_classes), global_probs, dtype=np.float32
    )
    indices = np.arange(len(train_code_ids))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    fold_size = len(indices) // n_folds

    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else len(indices)
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        # Build code -> probs mapping from training portion
        fold_codes = train_code_ids[train_idx]
        fold_labels = train_labels[train_idx]
        fold_map: dict[int, np.ndarray] = {}
        for code in np.unique(train_code_ids[val_idx]):
            code_int = int(code)
            mask = fold_codes == code_int
            count = int(mask.sum())
            if count > 0:
                probs = np.array(
                    [np.mean(fold_labels[mask] == c) for c in range(n_classes)]
                )
                fold_map[code_int] = (probs * count + global_probs * smoothing) / (
                    count + smoothing
                )
            else:
                fold_map[code_int] = global_probs

        # Fill in validation portion
        for i in val_idx:
            code_int = int(train_code_ids[i])
            if code_int in fold_map:
                train_tenc[i] = fold_map[code_int].astype(np.float32)

    return train_tenc, test_tenc


def codes_to_single_id(codes: np.ndarray, nbits: int) -> np.ndarray:
    """Convert multi-level trunk codes to a single integer ID per sample."""
    K = 2**nbits
    result = np.zeros(len(codes), dtype=np.int64)
    for level in range(codes.shape[1]):
        result = result * K + codes[:, level].astype(np.int64)
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_comparison(
    dataset_name: str,
    n_seeds: int = 10,
    rq_configs: list[tuple[int, int]] | None = None,
    output_dir: Path | None = None,
    device: str = "cpu",
    clf_names: list[str] | None = None,
) -> dict[str, object]:
    """Run classifiers with and without RQ features on a dataset."""

    if rq_configs is None:
        rq_configs = RECOMMENDED_CONFIGS.get(dataset_name, [(3, 2), (6, 1)])
    if clf_names is None:
        clf_names = ALL_CLASSIFIER_NAMES
    active_clf_names: list[str] = clf_names

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    t0 = time.time()
    train_dataset, _, test_dataset, metadata = load_dataset(dataset_name)
    train_features, train_labels = extract_features(train_dataset)
    test_features, test_labels = extract_features(test_dataset)
    n_classes: int = metadata.n_classes

    print(
        f"  Loaded in {time.time() - t0:.1f}s: "
        f"train={train_features.shape}, test={test_features.shape}, "
        f"classes={n_classes}"
    )

    # Prepare RQ-augmented features for each config
    rq_augmentations: dict[str, dict[str, np.ndarray]] = {}
    for nbits, d_cut in rq_configs:
        K = 2**nbits
        config_name = f"K={K}_d={d_cut}"
        print(f"\n  Fitting RQ: {config_name}...")
        t1 = time.time()

        (
            train_codes,
            test_codes,
            train_recon,
            test_recon,
            train_residual,
            test_residual,
        ) = fit_rq_and_augment(train_features, test_features, nbits, d_cut)

        train_code_ids = codes_to_single_id(train_codes, nbits)
        test_code_ids = codes_to_single_id(test_codes, nbits)
        n_unique = len(set(train_code_ids.tolist()))

        # Target encoding: P(class | trunk_code) via k-fold CV
        train_tenc, test_tenc = compute_target_encoding(
            train_code_ids, train_labels, test_code_ids, n_classes
        )

        print(
            f"    Fitted in {time.time() - t1:.1f}s: "
            f"{n_unique} unique contexts, recon shape={train_recon.shape}"
        )

        rq_augmentations[config_name] = {
            "train_code_ids": train_code_ids.reshape(-1, 1).astype(np.float32),
            "test_code_ids": test_code_ids.reshape(-1, 1).astype(np.float32),
            "train_level_codes": train_codes.astype(np.float32),
            "test_level_codes": test_codes.astype(np.float32),
            "train_recon": train_recon,
            "test_recon": test_recon,
            "train_residual": train_residual,
            "test_residual": test_residual,
            "train_tenc": train_tenc,
            "test_tenc": test_tenc,
        }

    # Define feature sets to compare
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw": (train_features, test_features),
    }
    for config_name, aug in rq_augmentations.items():
        feature_sets[f"raw+codes ({config_name})"] = (
            np.hstack([train_features, aug["train_code_ids"]]),
            np.hstack([test_features, aug["test_code_ids"]]),
        )
        feature_sets[f"raw+recon ({config_name})"] = (
            np.hstack([train_features, aug["train_recon"]]),
            np.hstack([test_features, aug["test_recon"]]),
        )
        feature_sets[f"raw+both ({config_name})"] = (
            np.hstack([train_features, aug["train_code_ids"], aug["train_recon"]]),
            np.hstack([test_features, aug["test_code_ids"], aug["test_recon"]]),
        )
        # Target encoding: P(class | trunk_code) — most informative for trees
        feature_sets[f"raw+tenc ({config_name})"] = (
            np.hstack([train_features, aug["train_tenc"]]),
            np.hstack([test_features, aug["test_tenc"]]),
        )
        # Quantization residual: what the trunk misses
        feature_sets[f"raw+residual ({config_name})"] = (
            np.hstack([train_features, aug["train_residual"]]),
            np.hstack([test_features, aug["test_residual"]]),
        )
        # Target encoding + residual: full RQ decomposition
        feature_sets[f"raw+tenc+resid ({config_name})"] = (
            np.hstack([train_features, aug["train_tenc"], aug["train_residual"]]),
            np.hstack([test_features, aug["test_tenc"], aug["test_residual"]]),
        )

    # Classifier factories
    use_gpu: bool = device.startswith("cuda")

    def make_classifiers(seed: int) -> dict[str, object]:
        all_clfs: dict[str, object] = {
            "XGBoost": xgb.XGBClassifier(
                random_state=seed,
                objective="multi:softmax" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                **({"num_class": n_classes} if n_classes > 2 else {}),
                max_depth=6,
                learning_rate=0.1,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="gpu_hist" if use_gpu else "hist",
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=-1,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
                n_jobs=2,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=seed,
                n_jobs=2,
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=200,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            "LogReg": LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=2,
            ),
            "FTTransformer": FTTransformerClassifier(
                n_epochs=100,
                lr=1e-4,
                batch_size=256,
                patience=16,
                device=device,
                random_state=seed,
            ),
            "ResNet": TabResNetClassifier(
                n_epochs=100,
                lr=1e-3,
                batch_size=256,
                patience=16,
                device=device,
                random_state=seed,
            ),
        }
        return {k: v for k, v in all_clfs.items() if k in active_clf_names}

    # Run across seeds
    seed_list = list(range(42, 42 + n_seeds))
    results_map: dict[tuple[str, str], list[float]] = {}
    for cn in clf_names:
        for fs_name in feature_sets:
            results_map[(cn, fs_name)] = []

    n_evals = len(clf_names) * len(feature_sets)
    print(
        f"\n  Running {len(feature_sets)} feature sets × "
        f"{len(clf_names)} classifiers × {n_seeds} seeds "
        f"({n_evals * n_seeds} total evaluations)..."
    )

    for seed in seed_list:
        t_seed = time.time()
        for fs_name, (X_train, X_test) in feature_sets.items():
            classifiers = make_classifiers(seed)
            for cn, clf in classifiers.items():
                clf.fit(X_train, train_labels)  # pyre-ignore[16]
                preds = clf.predict(X_test)  # pyre-ignore[16]
                preds = np.asarray(preds)
                if n_classes == 2 and cn == "XGBoost":
                    preds = (preds > 0.5).astype(int)
                else:
                    preds = preds.astype(int)
                acc = accuracy_score(test_labels, preds)
                results_map[(cn, fs_name)].append(acc)

        done = seed_list.index(seed) + 1
        elapsed = time.time() - t_seed
        print(f"    Seed {seed} ({done}/{n_seeds}, {elapsed:.0f}s):", end="")
        for cn in clf_names[:3]:  # show first 3 classifiers
            raw_acc = results_map[(cn, "raw")][-1]
            print(f"  {cn}={raw_acc:.4f}", end="")
        print()

    # Results table
    print(f"\n  {'Classifier':14s} {'Features':35s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'-' * 69}")
    for cn in clf_names:
        for fs_name in feature_sets:
            accs = results_map[(cn, fs_name)]
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            print(f"  {cn:14s} {fs_name:35s} {mean_acc:8.4f} {std_acc:8.4f}")
        print()

    # Per-classifier significance: best RQ-augmented vs raw
    sig_results: dict[str, dict[str, object]] = {}
    for cn in clf_names:
        method_accs: dict[str, list[float]] = {}
        ours_set: set[str] = set()
        for fs_name in feature_sets:
            key = f"{cn}/{fs_name}"
            method_accs[key] = results_map[(cn, fs_name)]
            if fs_name != "raw":
                ours_set.add(key)
        sig = compute_pairwise_significance(
            method_accs, ours_set, higher_is_better=True
        )
        if sig:
            sig_results[cn] = sig
            print(f"  Significance ({cn}):")
            log_significance(sig, logging.getLogger(__name__), "accuracy")

    result: dict[str, object] = {
        "dataset": dataset_name,
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "n_features": train_features.shape[1],
        "n_classes": n_classes,
        "n_seeds": n_seeds,
        "classifiers": clf_names,
        "rq_configs": [{"nbits": nb, "d_cut": dc, "K": 2**nb} for nb, dc in rq_configs],
        "results": {
            f"{clf}/{fs}": {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "values": v,
            }
            for (clf, fs), v in results_map.items()
        },
        "significance": sig_results,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"clf_rq_{dataset_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n  Saved to {out_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-classifier + RQ feature augmentation"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(RECOMMENDED_CONFIGS.keys()),
        help="Datasets to run on",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/clf_rq",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for deep learning models (cpu or cuda)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=ALL_CLASSIFIER_NAMES,
        help=f"Classifiers to run. Choices: {ALL_CLASSIFIER_NAMES}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, object]] = {}

    for ds in args.datasets:
        try:
            result = run_comparison(
                ds,
                n_seeds=args.seeds,
                output_dir=output_dir,
                device=args.device,
                clf_names=args.classifiers,
            )
            all_results[ds] = result
        except Exception as e:
            print(f"\n  FAILED {ds}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: Multi-Classifier + RQ Augmentation")
    print(f"{'=' * 80}")

    print(
        f"{'Dataset':20s} {'Classifier':14s} {'Raw':>8s} "
        f"{'Best RQ':>8s} {'Delta':>8s} {'p-value':>8s}"
    )
    print("-" * 72)
    for ds in args.datasets:
        if ds not in all_results:
            print(f"{ds:20s}  FAILED")
            continue
        r = all_results[ds]
        assert isinstance(r, dict)
        results = r["results"]
        assert isinstance(results, dict)
        sig_obj = r.get("significance", {})
        sig = sig_obj if isinstance(sig_obj, dict) else {}
        first_clf = True
        for cn in args.classifiers:
            raw_key = f"{cn}/raw"
            raw_info = results.get(raw_key)
            if raw_info is None:
                continue
            assert isinstance(raw_info, dict)
            raw_mean = raw_info["mean"]
            # Find best RQ-augmented feature set for this classifier
            best_rq_key = None
            best_rq_mean = -1.0
            for key, info in results.items():
                assert isinstance(info, dict)
                if key.startswith(f"{cn}/") and key != raw_key:
                    if info["mean"] > best_rq_mean:
                        best_rq_mean = info["mean"]
                        best_rq_key = key
            if best_rq_key is None:
                continue
            delta = best_rq_mean - raw_mean
            clf_sig_obj = sig.get(cn, {})
            clf_sig = clf_sig_obj if isinstance(clf_sig_obj, dict) else {}
            p_val = clf_sig.get("p_value", float("nan"))
            ds_label = ds if first_clf else ""
            first_clf = False
            print(
                f"{ds_label:20s} {cn:14s} {raw_mean:8.4f} "
                f"{best_rq_mean:8.4f} {delta:+8.4f} {p_val:8.4f}"
            )
        print()

    # Save combined results
    combined_path = output_dir / "clf_rq_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
