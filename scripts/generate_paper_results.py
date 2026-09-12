#!/usr/bin/env python3
"""Build publication figures and LaTeX tables from the frozen E1 results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.sim2d.scenarios import make_scenario
from minimal_intervention_shared_control.sim2d.simulation import (
    SimulationConfig,
    run_simulation,
)

METHODS = (
    "delay_agreement",
    "single_step",
    "weighted_sum",
    "human_filter",
    "ours",
)
METHOD_LABELS = {
    "delay_agreement": "Delay agreement",
    "single_step": "Single step",
    "weighted_sum": "Weighted sum",
    "human_filter": "Human + filter",
    "ours": "Ours",
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
SCENARIO_LABELS = {
    "bend": "Bend",
    "crossing": "Crossing",
    "conflict_both_safe": "Both safe",
    "conflict_human_unsafe": "Human unsafe",
    "conflict_autonomy_unsafe": "Autonomy unsafe",
    "conflict_blend_unsafe": "Blend unsafe",
    "network_anomaly": "Network anomaly",
}
NETWORKS = ("N0", "N1", "N2", "NJ")
FORMAL_SEEDS = tuple(range(50))
PAIR_METRICS = (
    "collision",
    "success",
    "intervention_budget",
    "nominal_modification",
    "filter_modification",
    "authority_total_variation",
    "upper_infeasible_rate",
    "filter_infeasible_rate",
)
COLORS = {
    "delay_agreement": "#4C78A8",
    "single_step": "#F58518",
    "weighted_sum": "#54A24B",
    "human_filter": "#B279A2",
    "ours": "#D1495B",
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _key(row: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(row["scenario"]),
        str(row["network_condition"]),
        int(row["seed"]),
        str(row["method"]),
    )


def _load_formal_rows(base_path: Path, weighted_path: Path) -> list[dict[str, object]]:
    base = _read_jsonl(base_path)
    weighted_input = _read_jsonl(weighted_path)
    if base_path.resolve() == weighted_path.resolve():
        weighted = [row for row in weighted_input if row["method"] == "weighted_sum"]
    else:
        weighted = weighted_input
        if any(row["method"] != "weighted_sum" for row in weighted):
            raise ValueError("weighted-sum override contains another method")
    weighted_by_key = {_key(row): row for row in weighted}
    if len(weighted_by_key) != 1400:
        raise ValueError("weighted-sum override must contain 1400 unique rows")
    combined = [row for row in base if row["method"] != "weighted_sum"] + weighted
    expected = {
        (scenario, network, seed, method)
        for scenario in SCENARIOS
        for network in NETWORKS
        for seed in FORMAL_SEEDS
        for method in METHODS
    }
    counts = Counter(_key(row) for row in combined)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        missing = expected - set(counts)
        extra = set(counts) - expected
        duplicates = [key for key, count in counts.items() if count != 1]
        raise ValueError(
            f"invalid formal matrix: missing={len(missing)}, extra={len(extra)}, "
            f"duplicates={len(duplicates)}"
        )
    for row in combined:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value in {_key(row)}")
    return combined


def _select(
    rows: list[dict[str, object]],
    *,
    method: str | None = None,
    scenario: str | None = None,
    network: str | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if (method is None or row["method"] == method)
        and (scenario is None or row["scenario"] == scenario)
        and (network is None or row["network_condition"] == network)
    ]


def _mean(rows: list[dict[str, object]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - radius, center + radius


def _seed_cluster_values(
    rows: list[dict[str, object]], metric: str, method: str
) -> np.ndarray:
    return np.array(
        [
            np.mean(
                [
                    float(row[metric])
                    for row in rows
                    if row["method"] == method and int(row["seed"]) == seed
                ]
            )
            for seed in FORMAL_SEEDS
        ],
        dtype=float,
    )


def _cluster_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int = 10_000
) -> tuple[float, float]:
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_cluster(
    rows: list[dict[str, object]], metric: str, baseline: str, seed: int = 17
) -> tuple[float, float, float]:
    ours = _seed_cluster_values(rows, metric, "ours")
    other = _seed_cluster_values(rows, metric, baseline)
    differences = ours - other
    lower, upper = _cluster_ci(differences, np.random.default_rng(seed))
    return float(np.mean(differences)), lower, upper


def _escape(value: str) -> str:
    return value.replace("_", r"\_").replace("+", r"+")


def _write_tex(path: Path, lines: list[str]) -> None:
    content = "\n".join(lines).replace("\\\\%", "\\%") + "\n"
    path.write_text(content, encoding="utf-8")


def _overall_table(rows: list[dict[str, object]], output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Overall E1 results across 7 scenarios, 4 network conditions, and 50 paired seeds. Collision and success are trajectory counts; continuous entries are means. Lower is better except for success and clearance.}",
        r"\label{tab:e1-overall}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Method & Collision & Success & Clearance [m] & $I_\alpha$ & $I_{\rm nominal}$ & $I_{\rm filter}$ & $TV_\alpha$ & Upper inf. & Filter inf. & P99 [ms] \\",
        r"\midrule",
    ]
    for method in METHODS:
        selected = _select(rows, method=method)
        collision = sum(bool(row["collision"]) for row in selected)
        success = sum(bool(row["success"]) for row in selected)
        label = r"\textbf{Ours}" if method == "ours" else _escape(METHOD_LABELS[method])
        lines.append(
            f"{label} & {collision}/1400 & {success}/1400 & "
            f"{_mean(selected, 'minimum_clearance'):.3f} & "
            f"{_mean(selected, 'intervention_budget'):.3f} & "
            f"{_mean(selected, 'nominal_modification'):.3f} & "
            f"{_mean(selected, 'filter_modification'):.3f} & "
            f"{_mean(selected, 'authority_total_variation'):.3f} & "
            rf"{100 * _mean(selected, 'upper_infeasible_rate'):.1f}\% & "
            rf"{100 * _mean(selected, 'filter_infeasible_rate'):.1f}\% & "
            f"{1000 * _mean(selected, 'solve_time_p99'):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    _write_tex(output, lines)


def _paired_table(rows: list[dict[str, object]], output: Path) -> None:
    metric_labels = {
        "collision": "Collision",
        "success": "Success",
        "intervention_budget": r"$I_\alpha$",
        "nominal_modification": r"$I_{\rm nominal}$",
        "filter_modification": r"$I_{\rm filter}$",
        "authority_total_variation": r"$TV_\alpha$",
        "upper_infeasible_rate": "Upper infeasible",
        "filter_infeasible_rate": "Filter infeasible",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired differences (Ours minus baseline) with 95\% seed-cluster bootstrap intervals. Each of 50 seed clusters retains all scenario and network observations.}",
        r"\label{tab:e1-paired}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Baseline & Metric & Difference & 95\% CI lower & 95\% CI upper \\",
        r"\midrule",
    ]
    for baseline_index, baseline in enumerate(METHODS[:-1]):
        for metric in PAIR_METRICS:
            difference, lower, upper = _paired_cluster(rows, metric, baseline)
            lines.append(
                f"{_escape(METHOD_LABELS[baseline])} & {metric_labels[metric]} & "
                f"{difference:.4f} & {lower:.4f} & {upper:.4f} \\\\"
            )
        if baseline_index != len(METHODS[:-1]) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    _write_tex(output, lines)


def _scenario_table(rows: list[dict[str, object]], output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Scenario-stratified collision/success counts over four network conditions and 50 seeds (200 trajectories per cell).}",
        r"\label{tab:e1-scenarios}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Scenario & " + " & ".join(_escape(METHOD_LABELS[method]) for method in METHODS) + r" \\",
        r"\midrule",
    ]
    for scenario in SCENARIOS:
        values = []
        for method in METHODS:
            selected = _select(rows, method=method, scenario=scenario)
            collision = sum(bool(row["collision"]) for row in selected)
            success = sum(bool(row["success"]) for row in selected)
            values.append(f"{collision}/{success}")
        lines.append(
            f"{_escape(SCENARIO_LABELS[scenario])} & " + " & ".join(values) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\multicolumn{6}{l}{\footnotesize Each entry is collision count / success count.} \\",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    _write_tex(output, lines)


def _network_table(rows: list[dict[str, object]], output: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Success rate (\%) by network condition, pooled over seven scenarios and 50 seeds.}",
        r"\label{tab:e1-network}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & $\mathcal{N}_0$ & $\mathcal{N}_1$ & $\mathcal{N}_2$ & $\mathcal{N}_J$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        values = [100 * _mean(_select(rows, method=method, network=network), "success") for network in NETWORKS]
        label = r"\textbf{Ours}" if method == "ours" else _escape(METHOD_LABELS[method])
        lines.append(f"{label} & " + " & ".join(f"{value:.1f}" for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_tex(output, lines)


def _event_table(rows: list[dict[str, object]], output: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Event-aware timing. Event times are averaged only over trajectories in which the event occurred; completion time is averaged only over successful trajectories.}",
        r"\label{tab:e1-events}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Authority event & $t_\alpha$ [s] & Filter event & $t_F$ [s] & Success time [s] \\",
        r"\midrule",
    ]
    for method in METHODS:
        selected = _select(rows, method=method)
        authority = [row for row in selected if float(row["intervention_budget"]) > 1e-7]
        filtered = [row for row in selected if float(row["filter_modification"]) > 1e-7]
        successful = [row for row in selected if bool(row["success"])]
        authority_time = (
            f"{_mean(authority, 'first_authority_time'):.2f}" if authority else "--"
        )
        filter_time = f"{_mean(filtered, 'first_filter_time'):.2f}" if filtered else "--"
        label = r"\textbf{Ours}" if method == "ours" else _escape(METHOD_LABELS[method])
        lines.append(
            f"{label} & {len(authority)}/1400 & "
            f"{authority_time} & {len(filtered)}/1400 & {filter_time} & "
            f"{_mean(successful, 'completion_time'):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_tex(output, lines)


def _selection_table(selection_path: Path, output: Path) -> None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Weighted-sum baseline selection on 56 development trajectories per weight; formal seeds 0--49 were not inspected.}",
        r"\label{tab:weighted-selection}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$w_\alpha$ & Collision [\%] & Success [\%] & $I_\alpha$ & $I_{\rm filter}$ & Upper inf. [\%] \\",
        r"\midrule",
    ]
    chosen = float(selection["selected"]["weight"])
    for row in selection["candidates"]:
        values = [
            f"{float(row['weight']):g}",
            f"{100 * float(row['collision_rate']):.1f}",
            f"{100 * float(row['success_rate']):.1f}",
            f"{float(row['intervention_budget']):.3f}",
            f"{float(row['filter_modification']):.3f}",
            f"{100 * float(row['upper_infeasible_rate']):.1f}",
        ]
        if float(row["weight"]) == chosen:
            values = [rf"\textbf{{{value}}}" for value in values]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_tex(output, lines)


def _save_figure(figure: object, base: Path) -> None:
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")


def _plot_heatmaps(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    row_labels = [
        f"{SCENARIO_LABELS[scenario]} / {network}"
        for scenario in SCENARIOS
        for network in NETWORKS
    ]
    collision = np.empty((len(row_labels), len(METHODS)))
    success = np.empty_like(collision)
    for row_index, (scenario, network) in enumerate(product(SCENARIOS, NETWORKS)):
        for method_index, method in enumerate(METHODS):
            selected = _select(rows, method=method, scenario=scenario, network=network)
            collision[row_index, method_index] = 100 * _mean(selected, "collision")
            success[row_index, method_index] = 100 * _mean(selected, "success")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], collision, "Collision rate (%)", "Reds"),
        (axes[1], success, "Success rate (%)", "YlGn"),
    ):
        image = axis.imshow(values, aspect="auto", vmin=0, vmax=100, cmap=cmap)
        axis.set_title(title)
        axis.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS], rotation=35, ha="right")
        axis.set_yticks(range(len(row_labels)), row_labels if axis is axes[0] else [])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                axis.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color="white" if value > 55 else "black")
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    _save_figure(figure, output / "e1_scenario_network_heatmaps")
    plt.close(figure)


def _bootstrap_method_ci(
    rows: list[dict[str, object]], metric: str, method: str, seed: int
) -> tuple[float, float, float]:
    values = _seed_cluster_values(rows, metric, method)
    lower, upper = _cluster_ci(values, np.random.default_rng(seed))
    return float(np.mean(values)), lower, upper


def _plot_tradeoff(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for method in METHODS:
        selected = _select(rows, method=method)
        collision = 100 * _mean(selected, "collision")
        success = 100 * _mean(selected, "success")
        authority = _mean(selected, "intervention_budget")
        filter_modification = _mean(selected, "filter_modification")
        style = {"s": 90 if method == "ours" else 55, "color": COLORS[method], "zorder": 3 if method == "ours" else 2}
        axes[0].scatter(authority, collision, **style)
        axes[1].scatter(filter_modification, success, **style)
        axes[0].annotate(METHOD_LABELS[method], (authority, collision), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axes[1].annotate(METHOD_LABELS[method], (filter_modification, success), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel=r"Autonomous authority $I_\alpha$", ylabel="Collision rate (%)")
    axes[1].set(xlabel=r"Filter modification $I_{\rm filter}$", ylabel="Success rate (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save_figure(figure, output / "e1_safety_efficiency_tradeoff")
    plt.close(figure)


def _plot_quality(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    panels = (
        ("nominal_modification", r"Nominal modification $I_{\rm nominal}$", 1.0),
        ("filter_modification", r"Filter modification $I_{\rm filter}$", 1.0),
        ("authority_total_variation", r"Authority variation $TV_\alpha$", 1.0),
        ("upper_infeasible_rate", "Upper infeasible rate (%)", 100.0),
        ("filter_infeasible_rate", "Filter infeasible rate (%)", 100.0),
        ("solve_time_p99", "Trajectory P99 solve time (ms)", 1000.0),
    )
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
    x = np.arange(len(METHODS))
    for panel_index, (axis, (metric, title, scale)) in enumerate(zip(axes.flat, panels, strict=True)):
        estimates = [_bootstrap_method_ci(rows, metric, method, 100 + panel_index) for method in METHODS]
        means = np.array([value[0] for value in estimates]) * scale
        errors = np.array([[value[0] - value[1], value[2] - value[0]] for value in estimates]).T * scale
        axis.bar(x, means, color=[COLORS[method] for method in METHODS], width=0.72)
        axis.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=2, linewidth=0.8)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output / "e1_intervention_feasibility_runtime")
    plt.close(figure)


def _plot_network(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for method in METHODS:
        success = [100 * _mean(_select(rows, method=method, network=network), "success") for network in NETWORKS]
        authority = [_mean(_select(rows, method=method, network=network), "intervention_budget") for network in NETWORKS]
        axes[0].plot(NETWORKS, success, marker="o", label=METHOD_LABELS[method], color=COLORS[method], linewidth=2 if method == "ours" else 1.3)
        axes[1].plot(NETWORKS, authority, marker="o", label=METHOD_LABELS[method], color=COLORS[method], linewidth=2 if method == "ours" else 1.3)
    axes[0].set(ylabel="Success rate (%)", xlabel="Network condition")
    axes[1].set(ylabel=r"Autonomous authority $I_\alpha$", xlabel="Network condition")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    _save_figure(figure, output / "e1_network_robustness")
    plt.close(figure)


def _plot_selection(selection_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    data = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = sorted(data["candidates"], key=lambda row: float(row["weight"]))
    weights = np.array([float(row["weight"]) for row in candidates])
    figure, axes = plt.subplots(1, 2, figsize=(8.7, 3.7), constrained_layout=True)
    axes[0].plot(weights, [100 * float(row["collision_rate"]) for row in candidates], marker="o", label="Collision")
    axes[0].plot(weights, [100 * float(row["success_rate"]) for row in candidates], marker="s", label="Success")
    axes[1].plot(weights, [float(row["intervention_budget"]) for row in candidates], marker="o", label=r"$I_\alpha$")
    axes[1].plot(weights, [float(row["filter_modification"]) for row in candidates], marker="s", label=r"$I_{\rm filter}$")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel(r"Weighted-sum coefficient $w_\alpha$")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Rate (%)")
    axes[1].set_ylabel("Normalized intervention")
    _save_figure(figure, output / "weighted_sum_development_selection")
    plt.close(figure)


def _obstacle_positions(scenario_name: str, steps: int, dt: float) -> np.ndarray:
    scene = make_scenario(scenario_name)
    obstacle = scene.obstacles[0].instantiate()
    positions = []
    for _ in range(steps):
        obstacle.step(dt)
        positions.append(obstacle.position.copy())
    return np.asarray(positions)


def _write_trace(path: Path, method: str, result: object) -> None:
    payload = {
        "scenario": "crossing",
        "network_condition": "N0",
        "seed": 17,
        "method": method,
        "metrics": result.metrics,
        "records": result.records,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _representative_figure(output: Path, trace_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    trace_path.unlink(missing_ok=True)
    results = {
        method: run_simulation(
            "crossing",
            SimulationConfig(duration=12.0, method=method, seed=17, network_condition="N0"),
        )
        for method in ("single_step", "weighted_sum", "ours")
    }
    for method, result in results.items():
        _write_trace(trace_path, method, result)
    scene = make_scenario("crossing")
    longest = max(len(result.records) for result in results.values())
    obstacle = _obstacle_positions("crossing", longest, 0.02)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    trajectory_axis = axes[0, 0]
    for method, result in results.items():
        states = np.asarray([row["state"] for row in result.records])
        trajectory_axis.plot(states[:, 0], states[:, 1], color=COLORS[method], label=METHOD_LABELS[method], linewidth=2 if method == "ours" else 1.3)
    trajectory_axis.plot(scene.human_path.points[:, 0], scene.human_path.points[:, 1], "--", color="#777777", linewidth=1, label="Human path")
    trajectory_axis.plot(scene.autonomous_path.points[:, 0], scene.autonomous_path.points[:, 1], ":", color="#222222", linewidth=1, label="Autonomy path")
    trajectory_axis.plot(obstacle[:, 0], obstacle[:, 1], color="#8C564B", linewidth=2, alpha=0.8, label="THOR pedestrian")
    crossing_index = int(np.argmin(np.linalg.norm(obstacle - np.array([1.5, 0.0]), axis=1)))
    trajectory_axis.add_patch(Circle(obstacle[crossing_index], 0.27, facecolor="#8C564B", alpha=0.2, edgecolor="#8C564B"))
    trajectory_axis.set(xlabel="x [m]", ylabel="y [m]", title="Crossing trajectories")
    trajectory_axis.set_aspect("equal")
    trajectory_axis.legend(fontsize=7, ncol=2)
    for axis, metric, title in (
        (axes[0, 1], "alpha", r"Autonomous authority $\alpha$"),
        (axes[1, 0], "filter", "Filter correction"),
        (axes[1, 1], "clearance", "Physical clearance"),
    ):
        for method, result in results.items():
            times = np.asarray([row["time"] for row in result.records])
            if metric == "filter":
                nominal = np.asarray([row["nominal"] for row in result.records])
                filtered = np.asarray([row["filtered"] for row in result.records])
                values = np.linalg.norm((filtered - nominal) / np.array([1.2, 1.8]), axis=1)
            else:
                values = np.asarray([row[metric] for row in result.records], dtype=float)
            axis.plot(times, values, color=COLORS[method], label=METHOD_LABELS[method], linewidth=1.8 if method == "ours" else 1.1)
        axis.set(xlabel="Time [s]", title=title)
        axis.grid(alpha=0.25)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_ylabel("Clearance [m]")
    _save_figure(figure, output / "e1_representative_crossing")
    plt.close(figure)


def _write_machine_summary(rows: list[dict[str, object]], output: Path) -> None:
    payload: dict[str, object] = {"runs": len(rows), "methods": {}}
    for method in METHODS:
        selected = _select(rows, method=method)
        collisions = sum(bool(row["collision"]) for row in selected)
        successes = sum(bool(row["success"]) for row in selected)
        payload["methods"][method] = {
            "runs": len(selected),
            "collisions": collisions,
            "collision_wilson_95": _wilson(collisions, len(selected)),
            "successes": successes,
            "success_wilson_95": _wilson(successes, len(selected)),
            **{metric: _mean(selected, metric) for metric in PAIR_METRICS[2:]},
            "minimum_clearance": _mean(selected, "minimum_clearance"),
            "solve_time_p99": _mean(selected, "solve_time_p99"),
        }
    payload["paired_ours_minus_baseline"] = {
        baseline: {
            metric: dict(zip(("difference", "ci95_lower", "ci95_upper"), _paired_cluster(rows, metric, baseline), strict=True))
            for metric in PAIR_METRICS
        }
        for baseline in METHODS[:-1]
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("results/full/e1_runs.jsonl"))
    parser.add_argument(
        "--weighted-sum",
        type=Path,
        default=Path("results/paper/weighted_sum_formal.jsonl"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/paper/weighted_sum_sensitivity.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    args = parser.parse_args()
    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_formal_rows(args.base, args.weighted_sum)
    _overall_table(rows, table_dir / "e1_overall.tex")
    _paired_table(rows, table_dir / "e1_paired_comparisons.tex")
    _scenario_table(rows, table_dir / "e1_by_scenario.tex")
    _network_table(rows, table_dir / "e1_by_network.tex")
    _event_table(rows, table_dir / "e1_event_timing.tex")
    _selection_table(args.selection, table_dir / "weighted_sum_selection.tex")
    _plot_heatmaps(rows, figure_dir)
    _plot_tradeoff(rows, figure_dir)
    _plot_quality(rows, figure_dir)
    _plot_network(rows, figure_dir)
    _plot_selection(args.selection, figure_dir)
    _representative_figure(
        figure_dir, args.output_dir / "representative_crossing_records.jsonl"
    )
    _write_machine_summary(rows, args.output_dir / "e1_paper_summary.json")
    print(f"wrote paper artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
