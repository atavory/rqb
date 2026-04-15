

"""Batch significance analysis across all experiment result JSONs.

Loads results from results/ (or a custom directory), extracts per-seed
values, runs pairwise significance tests (paired t-test, bootstrap CI,
Cohen's d), and prints summary tables.

Priority comparisons:
  - Contrastive: m-RQ vs Gaussian (best ours vs best baseline)
  - Classifier: each classifier family (XGBoost/LightGBM/...) ours vs raw
  - Label efficiency: RQ-based vs baselines at each budget
  - Bandits: RQ-LinTS vs LinTS (when per-seed data available)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from modules.significance import (
    compute_pairwise_significance,
    log_significance,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


# ── Method classification ────────────────────────────────────────────────

CONTRASTIVE_OURS: set[str] = {
    "m-RQ",
    "m-RQ+Gaussian",
    "RQ-SupCon+Proto",
    "RQ-WeightedSupCon",
    "RQ-HardNegSupCon",
    "RQ-Proto+Weighted",
}

LABEL_EFF_OURS: set[str] = {
    "RQ-Strat-d1",
    "RQ-Prop-d1",
    "RQ-SeedUnc-d1",
    "RQ-DivUnc-d1",
    "RQ-Strat-d3",
    "RQ-Prop-d3",
    "RQ-SeedUnc-d3",
    "RQ-DivUnc-d3",
}

BANDIT_OURS: set[str] = {
    "RQ-LinTS",
    "ProjRQ-4",
    "ProjRQ-8",
    "ProjRQ-16",
    "ProjRQ-32",
    "ProjRQ-48",
    "TS",  # HierarchicalTS uses RQ trunk codes
    "KM-LinTS",  # ablation — treated as ours to allow comparison
}


# ── Analyzers ────────────────────────────────────────────────────────────


def analyze_contrastive(results_dir: Path) -> list[dict[str, object]]:
    """Analyze contrastive experiment results.

    Scans for JSON files containing a 'results' list of per-seed records
    with keys: dataset, seed, method, accuracy.

    Returns a list of significance result dicts (one per dataset).
    """
    all_records: list[dict[str, object]] = []
    for json_path in sorted(results_dir.rglob("mrq_results_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            continue
        # Check it has per-seed records
        if not isinstance(results[0], dict) or "seed" not in results[0]:
            continue
        all_records.extend(results)

    if not all_records:
        return []

    # Group by dataset
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in all_records:
        ds = str(r["dataset"])
        by_dataset[ds].append(r)

    sig_results: list[dict[str, object]] = []
    for dataset in sorted(by_dataset):
        records = by_dataset[dataset]
        # Build method_seed_values: {method: [acc_seed0, acc_seed1, ...]}
        method_seeds: dict[str, dict[int, float]] = defaultdict(dict)
        for r in records:
            method = str(r["method"])
            seed = int(r["seed"])  # pyre-ignore[6]
            acc = float(r["accuracy"])  # pyre-ignore[6]
            method_seeds[method][seed] = acc

        # Align seeds: only include seeds present for ALL methods
        all_seeds_sets = [set(s.keys()) for s in method_seeds.values()]
        if not all_seeds_sets:
            continue
        common_seeds = sorted(set.intersection(*all_seeds_sets))
        if len(common_seeds) < 2:
            logger.warning(
                f"  Contrastive/{dataset}: only {len(common_seeds)} common seeds, skipping"
            )
            continue

        method_seed_values: dict[str, list[float]] = {}
        for method, seed_map in method_seeds.items():
            method_seed_values[method] = [seed_map[s] for s in common_seeds]

        sig = compute_pairwise_significance(
            method_seed_values,
            ours_methods=CONTRASTIVE_OURS,
            higher_is_better=True,
        )
        if sig:
            sig["dataset"] = dataset
            sig["experiment_type"] = "contrastive"
            sig["n_methods"] = len(method_seed_values)
            sig_results.append(sig)

    return sig_results


def analyze_classifier(results_dir: Path) -> list[dict[str, object]]:
    """Analyze classifier augmentation results.

    JSON format: results = {
        "ClassifierName/feature_set": {"mean": float, "std": float, "values": [float]}
    }

    Compares each classifier family: ours (any augmented) vs baseline (raw).
    """
    sig_results: list[dict[str, object]] = []
    for json_path in sorted(results_dir.rglob("clf_rq_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        results = data.get("results", {})
        if not isinstance(results, dict):
            continue

        dataset = str(data.get("dataset", json_path.stem))

        # Group by classifier family
        classifier_families: dict[str, dict[str, list[float]]] = defaultdict(dict)
        for key, val in results.items():
            if not isinstance(val, dict) or "values" not in val:
                continue
            parts = key.split("/", 1)
            if len(parts) != 2:
                continue
            clf_name, feature_set = parts
            classifier_families[clf_name][key] = val["values"]

        for clf_name, method_values in sorted(classifier_families.items()):
            raw_key = f"{clf_name}/raw"
            if raw_key not in method_values:
                continue

            # "ours" = any augmented method, "baseline" = raw
            ours_methods_clf: set[str] = {k for k in method_values if k != raw_key}
            sig = compute_pairwise_significance(
                method_values,
                ours_methods=ours_methods_clf,
                higher_is_better=True,
            )
            if sig:
                sig["dataset"] = dataset
                sig["classifier"] = clf_name
                sig["experiment_type"] = "classifier"
                sig_results.append(sig)

    return sig_results


def analyze_label_efficiency(results_dir: Path) -> list[dict[str, object]]:
    """Analyze label efficiency experiment results.

    JSON format: per_seed_results = [
        {encoder_seed, methods: {method_name: {budget: {mean, std, values}}}}
    ]
    aggregate = {method_name: {budget: {mean, std}}}

    Compares RQ-based methods vs baselines at each label budget.
    """
    sig_results: list[dict[str, object]] = []
    for json_path in sorted(results_dir.rglob("label_efficiency_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        dataset = str(data.get("dataset", json_path.stem))
        per_seed = data.get("per_seed_results", [])
        if not isinstance(per_seed, list) or not per_seed:
            continue

        # Extract all budgets from the first seed
        first_methods = per_seed[0].get("methods", {})
        if not first_methods:
            continue
        first_method_budgets = list(next(iter(first_methods.values())).keys())

        for budget in first_method_budgets:
            # Build method_seed_values across encoder seeds
            method_seed_values: dict[str, list[float]] = defaultdict(list)
            for seed_result in per_seed:
                methods = seed_result.get("methods", {})
                for method_name, budget_data in methods.items():
                    if budget in budget_data:
                        bud_entry = budget_data[budget]
                        if "values" in bud_entry:
                            # values are per-bandit-seed within this encoder seed
                            method_seed_values[method_name].append(
                                float(np.mean(bud_entry["values"]))
                            )
                        elif "mean" in bud_entry:
                            method_seed_values[method_name].append(
                                float(bud_entry["mean"])
                            )

            # Need at least 2 paired observations
            n_obs = min((len(v) for v in method_seed_values.values()), default=0)
            if n_obs < 2:
                continue

            # Trim to same length
            for m in method_seed_values:
                method_seed_values[m] = method_seed_values[m][:n_obs]

            sig = compute_pairwise_significance(
                dict(method_seed_values),
                ours_methods=LABEL_EFF_OURS,
                higher_is_better=True,
            )
            if sig:
                sig["dataset"] = dataset
                sig["budget"] = budget
                sig["experiment_type"] = "label_efficiency"
                sig_results.append(sig)

    return sig_results


def analyze_bandit(results_dir: Path) -> list[dict[str, object]]:
    """Analyze bandit experiment results.

    JSON format varies; older format stores regret_METHOD_nuX_lamY_mean/std
    as lists over rounds (no per-seed data). Newer format may include
    per-seed data. We extract final-round regret for comparison.

    Returns significance results where per-seed data is available,
    or summary notes where it is not.
    """
    sig_results: list[dict[str, object]] = []
    for json_path in sorted(results_dir.rglob("real_nts_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        dataset = str(data.get("dataset", json_path.stem))
        best_hp = data.get("best_hp", {})

        # Try to extract final regret for each method at best HP
        method_final_regret: dict[str, float] = {}
        for method_name, hp in best_hp.items():
            nu = hp.get("nu")
            lam = hp.get("lambda")
            if nu is None or lam is None:
                continue
            # Construct the key pattern
            method_key_map: dict[str, str] = {
                "TS": "ts",
                "Random": "random",
                "NeuralTS": "neuralts",
                "LinTS": "lints",
                "RQ-LinTS": "rq_lints",
                "KM-LinTS": "km_lints",
            }
            # Also handle ProjRQ variants
            for k_val in data.get("pca_k_values", []):
                method_key_map[f"ProjRQ-{k_val}"] = f"projrq_{k_val}"

            key_suffix = method_key_map.get(method_name)
            if key_suffix is None:
                continue
            mean_key = f"regret_{key_suffix}_nu{nu}_lam{lam}_mean"
            if mean_key in data:
                regret_series = data[mean_key]
                if isinstance(regret_series, list) and regret_series:
                    method_final_regret[method_name] = regret_series[-1]

        if method_final_regret:
            # Log summary even without significance
            sig: dict[str, object] = {
                "dataset": dataset,
                "experiment_type": "bandit",
                "note": "no per-seed data available for significance test",
                "method_final_regret": method_final_regret,
                "n_rounds": data.get("n_rounds"),
                "d_cut": data.get("d_cut"),
            }

            # Compute ratio if both RQ-LinTS and LinTS present
            rq_regret = method_final_regret.get("RQ-LinTS")
            lin_regret = method_final_regret.get("LinTS")
            if rq_regret is not None and lin_regret is not None and lin_regret > 0:
                sig["rq_vs_lints_ratio"] = rq_regret / lin_regret

            km_regret = method_final_regret.get("KM-LinTS")
            if km_regret is not None and lin_regret is not None and lin_regret > 0:
                sig["km_vs_lints_ratio"] = km_regret / lin_regret

            sig_results.append(sig)

    return sig_results


def analyze_feature_mode_comparison(results_dir: Path) -> list[dict[str, object]]:
    """Compare results across feature modes (embeddings vs raw vs selected).

    Scans ALL experiment result JSONs for a 'feature_mode' field in the config
    or metadata, groups by (dataset, experiment_type), and reports per-mode
    summary statistics.

    Returns a list of comparison dicts (one per dataset x experiment_type).
    """
    comparisons: list[dict[str, object]] = []

    # Scan bandit results
    for json_path in sorted(results_dir.rglob("real_nts_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        metadata = data.get("metadata", {})
        config = metadata.get("config", data.get("config", {}))
        feature_mode = config.get("feature_mode", "embeddings")
        dataset = str(data.get("dataset", config.get("dataset", json_path.stem)))

        # Extract final regret for key methods
        best_hp = data.get("best_hp", {})
        method_final_regret: dict[str, float] = {}
        for method_name, hp in best_hp.items():
            nu = hp.get("nu")
            lam = hp.get("lambda")
            if nu is None or lam is None:
                continue
            key_map: dict[str, str] = {
                "LinTS": "lints",
                "RQ-LinTS": "rq_lints",
                "KM-LinTS": "km_lints",
            }
            key_suffix = key_map.get(method_name)
            if key_suffix is None:
                continue
            mean_key = f"regret_{key_suffix}_nu{nu}_lam{lam}_mean"
            if mean_key in data:
                regret_series = data[mean_key]
                if isinstance(regret_series, list) and regret_series:
                    method_final_regret[method_name] = regret_series[-1]

        if method_final_regret:
            comparisons.append(
                {
                    "dataset": dataset,
                    "experiment_type": "bandit",
                    "feature_mode": feature_mode,
                    "method_final_regret": method_final_regret,
                }
            )

    # Scan contrastive results
    for json_path in sorted(results_dir.rglob("mrq_results_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        metadata = data.get("metadata", {})
        config = metadata.get("config", data.get("config", {}))
        feature_mode = config.get("feature_mode", "embeddings")

        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            continue

        # Group accuracies by method
        method_accs: dict[str, list[float]] = defaultdict(list)
        for r in results:
            method_accs[str(r.get("method", "?"))].append(float(r.get("accuracy", 0)))

        for method, accs in method_accs.items():
            dataset = str(results[0].get("dataset", "?"))
            comparisons.append(
                {
                    "dataset": dataset,
                    "experiment_type": "contrastive",
                    "feature_mode": feature_mode,
                    "method": method,
                    "mean_accuracy": float(np.mean(accs)),
                    "std_accuracy": float(np.std(accs)),
                    "n_seeds": len(accs),
                }
            )

    return comparisons


def format_feature_mode_table(comparisons: list[dict[str, object]]) -> str:
    """Format feature mode comparison as markdown table."""
    if not comparisons:
        return "## Feature Mode Comparison\n\nNo results found.\n"

    lines: list[str] = ["## Feature Mode Comparison\n"]

    # Group by (dataset, experiment_type)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for c in comparisons:
        key = f"{c['dataset']}|{c['experiment_type']}"
        groups[key].append(c)

    # Bandit comparisons
    bandit_groups = {
        k: v for k, v in groups.items() if v[0]["experiment_type"] == "bandit"
    }
    if bandit_groups:
        lines.append("### Bandit: Feature Mode Impact on Regret\n")
        lines.append("| Dataset | Feature Mode | RQ-LinTS | LinTS | Ratio |")
        lines.append("|---------|-------------|----------|-------|-------|")
        for key in sorted(bandit_groups):
            for c in sorted(
                bandit_groups[key], key=lambda x: str(x.get("feature_mode", ""))
            ):
                regrets = c.get("method_final_regret", {})
                assert isinstance(regrets, dict)
                rq = regrets.get("RQ-LinTS")
                lin = regrets.get("LinTS")
                rq_str = f"{rq:.1f}" if isinstance(rq, (int, float)) else "—"
                lin_str = f"{lin:.1f}" if isinstance(lin, (int, float)) else "—"
                ratio = (
                    f"{float(rq) / float(lin):.3f}"
                    if isinstance(rq, (int, float))
                    and isinstance(lin, (int, float))
                    and lin > 0
                    else "—"
                )
                lines.append(
                    f"| {c['dataset']} | {c['feature_mode']} "
                    f"| {rq_str} | {lin_str} | {ratio} |"
                )
        lines.append("")

    # Contrastive comparisons
    contrastive_groups = {
        k: v for k, v in groups.items() if v[0]["experiment_type"] == "contrastive"
    }
    if contrastive_groups:
        lines.append("### Contrastive: Feature Mode Impact on Accuracy\n")
        lines.append("| Dataset | Feature Mode | Method | Mean Acc | Std | Seeds |")
        lines.append("|---------|-------------|--------|----------|-----|-------|")
        for key in sorted(contrastive_groups):
            for c in sorted(
                contrastive_groups[key],
                key=lambda x: (
                    str(x.get("feature_mode", "")),
                    str(x.get("method", "")),
                ),
            ):
                lines.append(
                    f"| {c['dataset']} | {c['feature_mode']} | {c.get('method', '?')} "
                    f"| {float(c.get('mean_accuracy', 0)):.4f} "
                    f"| {float(c.get('std_accuracy', 0)):.4f} "
                    f"| {c.get('n_seeds', 0)} |"
                )
        lines.append("")

    return "\n".join(lines)


def format_significance_table(results: list[dict[str, object]], title: str) -> str:
    """Format significance results as a markdown table."""
    if not results:
        return f"## {title}\n\nNo results found.\n"

    lines: list[str] = [f"## {title}\n"]

    # Check if any have actual significance data
    has_sig = any("p_value" in r for r in results)

    if has_sig:
        lines.append(
            "| Dataset | Best Ours | Best Baseline | Ours Mean | Base Mean "
            "| Diff | p-value | Cohen's d | Sig? |"
        )
        lines.append(
            "|---------|-----------|---------------|-----------|-----------|"
            "------|---------|-----------|------|"
        )
        for r in results:
            dataset = r.get("dataset", "?")
            extra = ""
            if "budget" in r:
                extra = f" (n={r['budget']})"
            if "classifier" in r:
                extra = f" [{r['classifier']}]"

            best_ours = r.get("best_ours", "?")
            best_base = r.get("best_baseline", "?")
            ours_mean = r.get("ours_mean", 0)
            base_mean = r.get("base_mean", 0)
            mean_diff = r.get("mean_diff", 0)
            p_value = r.get("p_value", 1.0)
            cohens_d = r.get("effect_size_cohens_d", 0)
            sig = "**YES**" if float(p_value) < 0.05 else "no"  # pyre-ignore[6]

            lines.append(
                f"| {dataset}{extra} | {best_ours} | {best_base} "
                f"| {float(ours_mean):.4f} | {float(base_mean):.4f} "  # pyre-ignore[6]
                f"| {float(mean_diff):+.4f} | {float(p_value):.4f} "  # pyre-ignore[6]
                f"| {float(cohens_d):.3f} | {sig} |"  # pyre-ignore[6]
            )
    else:
        # Bandit-style summary without per-seed significance
        lines.append(
            "| Dataset | d_cut | Rounds | RQ-LinTS | LinTS | Ratio | KM-LinTS | KM Ratio |"
        )
        lines.append(
            "|---------|-------|--------|----------|-------|-------|----------|----------|"
        )
        for r in results:
            dataset = r.get("dataset", "?")
            d_cut = r.get("d_cut", "?")
            n_rounds = r.get("n_rounds", "?")
            regrets = r.get("method_final_regret", {})
            assert isinstance(regrets, dict)
            rq = regrets.get("RQ-LinTS", "—")
            lin = regrets.get("LinTS", "—")
            ratio = r.get("rq_vs_lints_ratio", "—")
            km = regrets.get("KM-LinTS", "—")
            km_ratio = r.get("km_vs_lints_ratio", "—")
            rq_str = f"{rq:.1f}" if isinstance(rq, float) else str(rq)
            lin_str = f"{lin:.1f}" if isinstance(lin, float) else str(lin)
            ratio_str = f"{ratio:.3f}" if isinstance(ratio, float) else str(ratio)
            km_str = f"{km:.1f}" if isinstance(km, float) else str(km)
            km_r_str = (
                f"{km_ratio:.3f}" if isinstance(km_ratio, float) else str(km_ratio)
            )
            lines.append(
                f"| {dataset} | {d_cut} | {n_rounds} "
                f"| {rq_str} | {lin_str} | {ratio_str} "
                f"| {km_str} | {km_r_str} |"
            )

    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch significance analysis across experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root results directory (default: results/)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "contrastive",
            "classifier",
            "label_efficiency",
            "bandit",
            "feature_mode",
            "all",
        ],
        default="all",
        help="Which experiment type to analyze (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    output_lines: list[str] = ["# Significance Analysis Report\n"]

    if args.experiment in ("contrastive", "all"):
        contrastive_dir = results_dir / "contrastive"
        if contrastive_dir.exists():
            logger.info("=== Analyzing contrastive results ===")
            sig = analyze_contrastive(contrastive_dir)
            for s in sig:
                log_significance(s, logger, "accuracy")
            table = format_significance_table(sig, "Contrastive Learning")
            output_lines.append(table)
        else:
            logger.info("No contrastive results directory found")

    if args.experiment in ("classifier", "all"):
        classifier_dir = results_dir / "classifier"
        if classifier_dir.exists():
            logger.info("\n=== Analyzing classifier results ===")
            sig = analyze_classifier(classifier_dir)
            for s in sig:
                log_significance(s, logger, f"{s.get('classifier', '?')} accuracy")
            table = format_significance_table(sig, "Classifier Augmentation")
            output_lines.append(table)
        else:
            logger.info("No classifier results directory found")

    if args.experiment in ("label_efficiency", "all"):
        label_dir = results_dir / "label_efficiency"
        if label_dir.exists():
            logger.info("\n=== Analyzing label efficiency results ===")
            sig = analyze_label_efficiency(label_dir)
            for s in sig:
                log_significance(s, logger, f"accuracy@{s.get('budget', '?')}")
            table = format_significance_table(sig, "Label Efficiency")
            output_lines.append(table)
        else:
            logger.info("No label efficiency results directory found")

    if args.experiment in ("bandit", "all"):
        bandit_dir = results_dir / "bandit"
        if bandit_dir.exists():
            logger.info("\n=== Analyzing bandit results ===")
            sig = analyze_bandit(bandit_dir)
            for s in sig:
                regrets = s.get("method_final_regret", {})
                assert isinstance(regrets, dict)
                ratio = s.get("rq_vs_lints_ratio")
                dataset = s.get("dataset", "?")
                d_cut = s.get("d_cut", "?")
                logger.info(
                    f"  {dataset} (d_cut={d_cut}): "
                    f"RQ-LinTS={regrets.get('RQ-LinTS', '—'):.1f}, "
                    f"LinTS={regrets.get('LinTS', '—'):.1f}"
                    + (f", ratio={ratio:.3f}" if isinstance(ratio, float) else "")
                )
            table = format_significance_table(sig, "Contextual Bandits")
            output_lines.append(table)
        else:
            logger.info("No bandit results directory found")

    if args.experiment in ("feature_mode", "all"):
        logger.info("\n=== Analyzing feature mode comparison ===")
        comparisons = analyze_feature_mode_comparison(results_dir)
        if comparisons:
            table = format_feature_mode_table(comparisons)
            output_lines.append(table)
            n_entries = len(comparisons)
            modes = sorted(set(str(c.get("feature_mode", "?")) for c in comparisons))
            logger.info(
                f"  {n_entries} entries across feature modes: {', '.join(modes)}"
            )
        else:
            logger.info("  No feature_mode comparison data found")

    # Write output
    report = "\n".join(output_lines)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"\nReport written to {output_path}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
