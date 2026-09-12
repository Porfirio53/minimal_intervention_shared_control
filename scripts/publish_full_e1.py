"""Publish the frozen 7000-run E1 experiment as paper-ready artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from minimal_intervention_shared_control.sim2d.scenarios import make_scenario
from minimal_intervention_shared_control.sim2d.simulation import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
)

METHODS = (
    "delay_agreement",
    "single_step",
    "weighted_sum",
    "human_filter",
    "ours",
)
LABELS = {
    "delay_agreement": "Delay agreement",
    "single_step": "Single step",
    "weighted_sum": "Weighted sum (tuned)",
    "human_filter": "Human + filter",
    "ours": "Ours",
}
COLORS = {
    "delay_agreement": "#4477AA",
    "single_step": "#EE7733",
    "weighted_sum": "#228833",
    "human_filter": "#AA3377",
    "ours": "#CC3311",
}
STYLES = {
    "delay_agreement": ":",
    "single_step": "--",
    "weighted_sum": "-.",
    "human_filter": ":",
    "ours": "-",
}
SCENARIOS = (
    "bend",
    "crossing",
    "conflict_both_safe",
    "conflict_human_unsafe",
    "conflict_autonomy_unsafe",
    "conflict_blend_unsafe",
    "network_anomaly",
)
NETWORKS = ("N0", "N1", "N2", "NJ")
SEEDS = tuple(range(50))
RUNS_PER_METHOD = len(SCENARIOS) * len(NETWORKS) * len(SEEDS)
BOOTSTRAP_SAMPLES = 20_000


def read_and_validate(path: Path) -> tuple[list[dict], dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (scenario, network, seed, method)
        for scenario in SCENARIOS
        for network in NETWORKS
        for seed in SEEDS
        for method in METHODS
    }
    keys = [
        (
            row["scenario"],
            row["network_condition"],
            int(row["seed"]),
            row["method"],
        )
        for row in rows
    ]
    counts = Counter(keys)
    if set(counts) != expected or any(value != 1 for value in counts.values()):
        raise ValueError("E1 input is incomplete, duplicated, or outside the formal matrix")
    if len(rows) != 7000 or manifest["completed_runs"] != 7000:
        raise ValueError("E1 input or manifest does not contain 7000 completed runs")
    settings = manifest["settings"]
    if (
        float(settings["duration"]) != 20.0
        or tuple(settings["methods"]) != METHODS
        or tuple(settings["network_conditions"]) != NETWORKS
        or tuple(settings["scenarios"]) != SCENARIOS
        or tuple(settings["seeds"]) != SEEDS
    ):
        raise ValueError("manifest settings do not match the frozen E1 design")
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value in {keys[rows.index(row)]}")
    return rows, manifest


def selected(rows: list[dict], method: str) -> list[dict]:
    return [row for row in rows if row["method"] == method]


def seed_cluster_values(rows: list[dict], method: str, metric: str) -> np.ndarray:
    return np.array(
        [
            np.mean(
                [
                    float(row[metric])
                    for row in rows
                    if row["method"] == method and int(row["seed"]) == seed
                ]
            )
            for seed in SEEDS
        ],
        dtype=float,
    )


def cluster_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def wilson_interval(count: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = count / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return center - radius, center + radius


def summarize(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for method_index, method in enumerate(METHODS):
        group = selected(rows, method)
        collisions = sum(bool(row["collision"]) for row in group)
        successes = sum(bool(row["success"]) for row in group)
        metrics = {}
        for metric_index, metric in enumerate(
            ("intervention_budget", "executed_modification")
        ):
            values = seed_cluster_values(rows, method, metric)
            metrics[metric] = {
                "mean": float(values.mean()),
                "seed_cluster_bootstrap_95": cluster_interval(
                    values, 1000 + 10 * method_index + metric_index
                ),
            }
        output[method] = {
            "runs": len(group),
            "collisions": collisions,
            "collision_rate": collisions / len(group),
            "collision_wilson_95": wilson_interval(collisions, len(group)),
            "successes": successes,
            "success_rate": successes / len(group),
            "success_wilson_95": wilson_interval(successes, len(group)),
            **metrics,
        }
    return output


def paired_comparisons(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for baseline_index, baseline in enumerate(METHODS[:-1]):
        output[baseline] = {}
        for metric_index, metric in enumerate(
            ("collision", "success", "intervention_budget", "executed_modification")
        ):
            difference = seed_cluster_values(rows, "ours", metric) - seed_cluster_values(
                rows, baseline, metric
            )
            output[baseline][metric] = {
                "ours_minus_baseline": float(difference.mean()),
                "seed_cluster_bootstrap_95": cluster_interval(
                    difference, 2000 + 10 * baseline_index + metric_index
                ),
            }
    return output


def interval_text(mean: float, interval: tuple[float, float]) -> str:
    return f"{mean:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"


def write_table(summary: dict, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Overall E1 results over seven scenarios, four network conditions, and 50 paired seeds (1400 trajectories per method). Binary entries are rate [95\% Wilson interval]; continuous entries are mean [95\% seed-cluster bootstrap interval]. Lower is better except for success. Collision and success take precedence over intervention measures.}",
        r"\label{tab:e1-overall}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Collision [\%] & Success [\%] & $I_\alpha$ & $I_{\rm exec}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        values = summary[method]
        collision_interval = tuple(100 * x for x in values["collision_wilson_95"])
        success_interval = tuple(100 * x for x in values["success_wilson_95"])
        label = r"\textbf{Ours}" if method == "ours" else LABELS[method]
        lines.append(
            f"{label} & "
            f"{100 * values['collision_rate']:.1f} "
            f"[{collision_interval[0]:.1f}, {collision_interval[1]:.1f}] & "
            f"{100 * values['success_rate']:.1f} "
            f"[{success_interval[0]:.1f}, {success_interval[1]:.1f}] & "
            f"{interval_text(values['intervention_budget']['mean'], tuple(values['intervention_budget']['seed_cluster_bootstrap_95']))} & "
            f"{interval_text(values['executed_modification']['mean'], tuple(values['executed_modification']['seed_cluster_bootstrap_95']))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rerun_case(
    rows: list[dict], scenario: str, network: str, seed: int, methods: tuple[str, ...]
) -> dict[str, SimulationResult]:
    results = {
        method: run_simulation(
            scenario,
            SimulationConfig(
                duration=20.0,
                method=method,
                seed=seed,
                network_condition=network,
            ),
        )
        for method in methods
    }
    for method, result in results.items():
        reference = next(
            row
            for row in rows
            if row["scenario"] == scenario
            and row["network_condition"] == network
            and int(row["seed"]) == seed
            and row["method"] == method
        )
        for metric in (
            "collision",
            "success",
            "intervention_budget",
            "executed_modification",
            "minimum_clearance",
        ):
            actual = result.metrics[metric]
            expected = reference[metric]
            if isinstance(actual, bool):
                matches = actual == expected
            else:
                matches = math.isclose(float(actual), float(expected), abs_tol=1e-10)
            if not matches:
                raise ValueError(
                    f"representative trace drift for {scenario}/{network}/{seed}/"
                    f"{method}/{metric}: {actual} != {expected}"
                )
    return results


def obstacle_positions(scenario: str, steps: int, dt: float = 0.02) -> np.ndarray:
    obstacle = make_scenario(scenario).obstacles[0].instantiate()
    positions = []
    for _ in range(steps):
        obstacle.step(dt)
        positions.append(obstacle.position.copy())
    return np.asarray(positions)


def plot_paths(
    axis: plt.Axes,
    scenario: str,
    results: dict[str, SimulationResult],
    *,
    dynamic_obstacle: bool,
) -> None:
    scene = make_scenario(scenario)
    axis.plot(
        scene.human_path.points[:, 0],
        scene.human_path.points[:, 1],
        color="0.55",
        linestyle="--",
        linewidth=1.0,
        label="Human reference",
    )
    axis.plot(
        scene.autonomous_path.points[:, 0],
        scene.autonomous_path.points[:, 1],
        color="0.25",
        linestyle=":",
        linewidth=1.0,
        label="Autonomy reference",
    )
    for method, result in results.items():
        states = np.asarray([row["state"] for row in result.records], dtype=float)
        axis.plot(
            states[:, 0],
            states[:, 1],
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=2.1 if method == "ours" else 1.45,
            label=LABELS[method],
        )
    obstacle = scene.obstacles[0].instantiate()
    if dynamic_obstacle:
        longest = max(len(result.records) for result in results.values())
        positions = obstacle_positions(scenario, longest)
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color="0.35",
            linestyle="-.",
            linewidth=1.2,
            label="Pedestrian path",
        )
        closest = int(np.argmin(np.linalg.norm(positions - np.array([1.5, 0.0]), axis=1)))
        obstacle_center = positions[closest]
    else:
        obstacle_center = obstacle.position
    axis.add_patch(
        Circle(
            obstacle_center,
            obstacle.radius,
            facecolor="0.70",
            edgecolor="0.25",
            linewidth=0.8,
            alpha=0.75,
        )
    )
    axis.scatter(*scene.initial_state[:2], marker="o", s=24, c="black", zorder=7)
    axis.scatter(*scene.goal[:2], marker="*", s=90, c="black", zorder=7)
    axis.set(xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def plot_time_series(
    axis: plt.Axes, results: dict[str, SimulationResult], field: str, ylabel: str
) -> None:
    for method, result in results.items():
        times = np.asarray([row["time"] for row in result.records], dtype=float)
        values = np.asarray([row["filtered"] for row in result.records], dtype=float)
        component = 0 if field == "speed" else 1
        axis.plot(
            times,
            values[:, component],
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=2.0 if method == "ours" else 1.3,
            label=LABELS[method],
        )
    axis.set(xlabel="Time (s)", ylabel=ylabel)
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=7.5, framealpha=0.9, loc="best")


def save_case_figure(
    rows: list[dict],
    output: Path,
    *,
    scenario: str,
    methods: tuple[str, ...],
    field: str,
    ylabel: str,
    dynamic_obstacle: bool,
) -> dict[str, SimulationResult]:
    results = rerun_case(rows, scenario, "N0", 17, methods)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.45), layout="constrained")
    plot_paths(axes[0], scenario, results, dynamic_obstacle=dynamic_obstacle)
    axes[0].set_xlim(-0.15, 4.2)
    axes[0].set_ylim((-1.2, 1.45) if dynamic_obstacle else (-1.75, 2.05))
    axes[0].set_title("(a) Closed-loop trajectories", loc="left")
    plot_time_series(axes[1], results, field, ylabel)
    axes[1].set_title("(b) Executed command", loc="left")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return results


def serialize_traces(cases: dict[str, dict[str, SimulationResult]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for scenario, results in cases.items():
            for method, result in results.items():
                stream.write(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "network_condition": "N0",
                            "seed": 17,
                            "method": method,
                            "metrics": result.metrics,
                            "records": result.records,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/fullresult/full2/e1_runs.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/paper/full")
    )
    args = parser.parse_args()
    rows, manifest = read_and_validate(args.input)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    values = summarize(rows)
    paired = paired_comparisons(rows)
    write_table(values, tables / "e1_overall.tex")
    crossing = save_case_figure(
        rows,
        figures / "e1_crossing_mechanism",
        scenario="crossing",
        methods=("single_step", "weighted_sum", "ours"),
        field="speed",
        ylabel="Executed speed (m/s)",
        dynamic_obstacle=True,
    )
    conflict = save_case_figure(
        rows,
        figures / "e1_conflict_mechanism",
        scenario="conflict_human_unsafe",
        methods=("human_filter", "single_step", "weighted_sum", "ours"),
        field="turn_rate",
        ylabel="Executed turn rate (rad/s)",
        dynamic_obstacle=False,
    )
    serialize_traces(
        {"crossing": crossing, "conflict_human_unsafe": conflict},
        args.output_dir / "mechanism_traces.jsonl",
    )

    ours = values["ours"]
    weighted = values["weighted_sum"]
    single = values["single_step"]
    payload = {
        "scope": "formal E1; 7 scenarios x 4 networks x 50 paired seeds x 5 methods",
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "manifest": manifest,
        "bootstrap": {
            "unit": "seed; every resample retains all scenarios and networks",
            "samples": BOOTSTRAP_SAMPLES,
        },
        "methods": values,
        "paired_ours_minus_baseline": paired,
        "headline_effects": {
            "ours_vs_weighted_sum_success_percentage_points": 100
            * (ours["success_rate"] - weighted["success_rate"]),
            "ours_vs_weighted_sum_authority_reduction_percent": 100
            * (
                1
                - ours["intervention_budget"]["mean"]
                / weighted["intervention_budget"]["mean"]
            ),
            "ours_vs_weighted_sum_execution_modification_reduction_percent": 100
            * (
                1
                - ours["executed_modification"]["mean"]
                / weighted["executed_modification"]["mean"]
            ),
            "ours_vs_single_step_success_percentage_points": 100
            * (ours["success_rate"] - single["success_rate"]),
            "ours_vs_single_step_execution_modification_reduction_percent": 100
            * (
                1
                - ours["executed_modification"]["mean"]
                / single["executed_modification"]["mean"]
            ),
        },
        "representative_traces": {
            "network_condition": "N0",
            "seed": 17,
            "note": "rerun metrics were checked against the matching formal rows",
        },
    }
    (args.output_dir / "e1_full_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# Full E1 paper artifacts\n\n"
        "The single LaTeX table is computed from all 7000 formal runs. The two "
        "figures are fixed N0/seed-17 mechanism traces whose metrics are checked "
        "against the corresponding formal rows; they do not repeat aggregate table "
        "values. PDF files are for the paper and PNG files are previews. The table "
        "requires `booktabs` and `graphicx`.\n\n"
        "Reproduce in under two minutes with:\n\n"
        "```bash\n"
        ".venv/bin/python scripts/publish_full_e1.py\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"validated {len(rows)} runs and wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
