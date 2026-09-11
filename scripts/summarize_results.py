#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.evaluation.statistics import paired_bootstrap

PRIMARY_METRICS = (
    "collision",
    "success",
    "minimum_clearance",
    "completion_time",
    "intervention_budget",
    "nominal_modification",
    "filter_modification",
    "authority_total_variation",
    "filter_trigger_rate",
    "upper_infeasible_rate",
    "secondary_fallback_rate",
    "filter_infeasible_rate",
    "first_authority_time",
    "first_filter_time",
    "solve_time_p50",
    "solve_time_p95",
    "solve_time_p99",
)
PAIR_KEYS = ("scenario", "network_condition", "seed", "duration")


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"no experiment rows in {path}")
    return rows


def _numeric(row: dict[str, object], metric: str) -> float:
    return float(row[metric])


def _mean_ci(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    radius = 1.96 * standard_error if confidence == 0.95 else standard_error
    return mean, mean - radius, mean + radius


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, object]], group_keys: tuple[str, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary: dict[str, object] = dict(zip(group_keys, key, strict=True))
        summary["runs"] = len(group)
        for metric in PRIMARY_METRICS:
            if all(metric in row for row in group):
                mean, lower, upper = _mean_ci([_numeric(row, metric) for row in group])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_ci95_lower"] = lower
                summary[f"{metric}_ci95_upper"] = upper
        output.append(summary)
    return output


def _pair_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[key] for key in PAIR_KEYS)


def _paired(rows: list[dict[str, object]], reference: str) -> list[dict[str, object]]:
    by_method: dict[str, dict[tuple[object, ...], dict[str, object]]] = defaultdict(
        dict
    )
    for row in rows:
        by_method[str(row["method"])][_pair_key(row)] = row
    if reference not in by_method:
        raise ValueError(f"reference method {reference!r} is absent")
    output: list[dict[str, object]] = []
    for baseline in sorted(set(by_method) - {reference}):
        keys = sorted(set(by_method[reference]) & set(by_method[baseline]), key=str)
        if not keys:
            continue
        for metric in PRIMARY_METRICS:
            if not all(
                metric in by_method[reference][key]
                and metric in by_method[baseline][key]
                for key in keys
            ):
                continue
            ours = np.array(
                [_numeric(by_method[reference][key], metric) for key in keys]
            )
            other = np.array(
                [_numeric(by_method[baseline][key], metric) for key in keys]
            )
            interval = paired_bootstrap(ours, other, seed=17)
            output.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "pairs": len(keys),
                    "reference_mean": float(np.mean(ours)),
                    "baseline_mean": float(np.mean(other)),
                    "mean_difference_reference_minus_baseline": interval[
                        "mean_difference"
                    ],
                    "ci95_lower": interval["lower"],
                    "ci95_upper": interval["upper"],
                }
            )
    return output


def _merge_counts(target: Counter[str], value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, count in value.items():
        target[str(key)] += int(count)


def _status_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        for field in (
            "upper_status_counts",
            "stage1_status_counts",
            "stage2_status_counts",
            "filter_status_counts",
        ):
            counts: Counter[str] = Counter()
            for row in selected:
                _merge_counts(counts, row.get(field))
            total = sum(counts.values())
            for status, count in sorted(counts.items()):
                output.append(
                    {
                        "method": method,
                        "layer": field.removesuffix("_status_counts"),
                        "status": status,
                        "count": count,
                        "fraction": count / total if total else 0.0,
                    }
                )
    return output


def _write_plot(path: Path, aggregate: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    methods = [str(row["method"]) for row in aggregate]
    metrics = (
        ("collision_mean", "Collision rate"),
        ("intervention_budget_mean", "Autonomous authority"),
        ("filter_modification_mean", "Filter modification"),
        ("filter_infeasible_rate_mean", "Filter infeasible rate"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        values = [float(row.get(metric, 0.0)) for row in aggregate]
        axis.bar(methods, values, color="#277da1")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_conclusions(
    path: Path,
    rows: list[dict[str, object]],
    overall: list[dict[str, object]],
    paired: list[dict[str, object]],
) -> None:
    counts = {
        "runs": len(rows),
        "scenarios": len({row["scenario"] for row in rows}),
        "networks": len({row["network_condition"] for row in rows}),
        "seeds": len({row["seed"] for row in rows}),
    }
    lines = [
        "# E1 experiment summary",
        "",
        (
            f"This report covers {counts['runs']} trajectories: {counts['scenarios']} "
            f"scenarios, {counts['networks']} network conditions, and "
            f"{counts['seeds']} paired seeds. Confidence intervals in the CSV files "
            "use trajectory-level samples."
        ),
        "",
        "## Overall means",
        "",
        "| Method | Collision | Success | Min clearance | Authority | Nominal modification | Filter modification | Upper infeasible | Filter infeasible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(overall, key=lambda item: str(item["method"])):
        lines.append(
            "| {method} | {collision_mean:.3f} | {success_mean:.3f} | "
            "{minimum_clearance_mean:.3f} | {intervention_budget_mean:.3f} | "
            "{nominal_modification_mean:.3f} | {filter_modification_mean:.3f} | "
            "{upper_infeasible_rate_mean:.3f} | {filter_infeasible_rate_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Paired interpretation",
            "",
            "Differences below are Ours minus baseline. An interval excluding zero is a stable paired difference for this experiment matrix.",
            "",
        ]
    )
    selected_metrics = {"collision", "intervention_budget", "filter_modification"}
    for row in paired:
        if row["metric"] in selected_metrics:
            lines.append(
                f"- {row['metric']} versus {row['baseline']}: "
                f"{float(row['mean_difference_reference_minus_baseline']):.4f} "
                f"(95% CI {float(row['ci95_lower']):.4f} to "
                f"{float(row['ci95_upper']):.4f}, n={row['pairs']})."
            )
    lines.extend(
        [
            "",
            "## Reading rule",
            "",
            "Collision and infeasibility results must be read before intervention-efficiency metrics. Lower authority or lower nominal modification is beneficial only when safety is comparable. High hard-QP infeasibility is reported as an empirical limitation, not hidden by slack variables.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-ready summaries from trajectory-level JSONL"
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", default="ours")
    args = parser.parse_args()
    output = args.output_dir or args.results.parent
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.results)
    overall = _aggregate(rows, ("method",))
    stratified = _aggregate(rows, ("scenario", "network_condition", "method"))
    paired = _paired(rows, args.reference)
    statuses = _status_summary(rows)
    _write_csv(output / "e1_overall.csv", overall)
    _write_csv(output / "e1_by_scenario_network.csv", stratified)
    _write_csv(output / "e1_paired_comparisons.csv", paired)
    _write_csv(output / "e1_qp_status.csv", statuses)
    _write_plot(output / "e1_overview.png", overall)
    _write_conclusions(output / "e1_conclusions.md", rows, overall, paired)
    print(f"wrote E1 summaries to {output}")


if __name__ == "__main__":
    main()
