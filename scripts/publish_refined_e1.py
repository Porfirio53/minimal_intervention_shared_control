"""Publish the frozen small E1 holdout as two mechanism figures and one table."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from minimal_intervention_shared_control.sim2d.scenarios import make_scenario

METHODS = ("delay_agreement", "single_step", "weighted_sum", "human_filter", "ours")
LABELS = {
    "delay_agreement": "Delay agreement",
    "single_step": "Single step",
    "weighted_sum": "Weighted sum (tuned)",
    "human_filter": "Human + filter",
    "ours": "Ours",
}
COLORS = dict(
    zip(METHODS, ("#4477AA", "#EE7733", "#228833", "#AA3377", "#CC3311"), strict=True)
)
STYLES = dict(zip(METHODS, (":", "--", "--", "-.", "-"), strict=True))


def read_rows(paths: list[Path], selection: dict) -> list[dict]:
    rows = []
    manifests = []
    for path in paths:
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
        manifests.append(json.loads(path.with_suffix(".manifest.json").read_text()))
    if any(m["source_sha256"] != manifests[0]["source_sha256"] for m in manifests):
        raise ValueError(
            "holdout batches do not share identical source/configuration hashes"
        )
    for manifest in manifests:
        if any(
            manifest["risk"][key] != value
            for key, value in selection["selected"].items()
        ):
            raise ValueError("holdout risk settings differ from the frozen selection")
    expected = set(
        product(
            selection["scenarios"],
            selection["holdout_networks"],
            selection["holdout_seeds"],
            METHODS,
        )
    )
    keys = [
        (r["scenario"], r["network_condition"], r["seed"], r["method"]) for r in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("holdout matrix is incomplete or duplicated")
    if any(r["duration"] != selection["duration"] or r["overrides"] for r in rows):
        raise ValueError("holdout contains an unfrozen duration or parameter override")
    return rows


def summary(rows: list[dict]) -> dict:
    output = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        output[method] = {
            "runs": len(group),
            "collisions": sum(r["collision"] for r in group),
            "successes": sum(r["success"] for r in group),
            **{
                key: float(np.mean([r[key] for r in group]))
                for key in (
                    "intervention_budget",
                    "executed_modification",
                    "nominal_modification",
                    "filter_modification",
                    "authority_total_variation",
                    "upper_infeasible_rate",
                    "filter_infeasible_rate",
                )
            },
            "per_seed": {
                str(seed): {
                    "runs": sum(r["seed"] == seed for r in group),
                    "collisions": sum(
                        r["collision"] for r in group if r["seed"] == seed
                    ),
                    "successes": sum(r["success"] for r in group if r["seed"] == seed),
                    "intervention_budget": float(
                        np.mean(
                            [
                                r["intervention_budget"]
                                for r in group
                                if r["seed"] == seed
                            ]
                        )
                    ),
                    "executed_modification": float(
                        np.mean(
                            [
                                r["executed_modification"]
                                for r in group
                                if r["seed"] == seed
                            ]
                        )
                    ),
                }
                for seed in sorted({r["seed"] for r in group})
            },
        }
    return output


def write_table(values: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Local E1 holdout: seven scenarios, four network conditions and two unseen network seeds (56 runs per method; 20 s limit). Lower intervention is interpreted together with collision and task success.}",
        r"\label{tab:e1-refined}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Collisions & Successes & $I_\alpha$ & $I_{\rm exec}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = values[method]
        label = r"\textbf{Ours}" if method == "ours" else LABELS[method]
        lines.append(
            f"{label} & {row['collisions']}/{row['runs']} & {row['successes']}/{row['runs']} & {row['intervention_budget']:.3f} & {row['executed_modification']:.3f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def case_figure(
    rows: list[dict], scenario: str, selected: tuple[str, ...], path: Path
) -> None:
    scene = make_scenario(scenario)
    group = {
        r["method"]: r
        for r in rows
        if r["scenario"] == scenario
        and r["network_condition"] == "N0"
        and r["seed"] == 950001
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), layout="constrained")
    for method in selected:
        records = group[method]["records"]
        states = np.array([r["state"] for r in records])
        times = np.array([r["time"] for r in records])
        kwargs = {
            "color": COLORS[method],
            "linestyle": STYLES[method],
            "linewidth": 2 if method == "ours" else 1.4,
            "label": LABELS[method],
        }
        axes[0].plot(states[:, 0], states[:, 1], **kwargs)
        if scenario == "crossing":
            values = np.array([r["filtered"][0] for r in records])
        else:
            values = np.array([r["alpha"] for r in records])
        axes[1].plot(times, values, **kwargs)
    axes[0].plot(
        scene.human_path.points[:, 0],
        scene.human_path.points[:, 1],
        color="0.65",
        linestyle=":",
        linewidth=1,
    )
    axes[0].plot(
        scene.autonomous_path.points[:, 0],
        scene.autonomous_path.points[:, 1],
        color="0.65",
        linestyle=":",
        linewidth=1,
    )
    axes[0].scatter(*scene.goal[:2], marker="*", s=90, c="black", zorder=6)
    obstacle = scene.obstacles[0].instantiate()
    if scenario == "crossing":
        positions = []
        for _ in range(250):
            obstacle.step(0.02)
            positions.append(obstacle.position.copy())
        positions = np.array(positions)
        axes[0].plot(
            positions[:, 0], positions[:, 1], color="0.4", linestyle="-.", linewidth=1.2
        )
        axes[0].annotate(
            "Pedestrian",
            (1.5, -0.6),
            xytext=(2.25, -0.95),
            arrowprops={"arrowstyle": "->", "color": "0.4"},
            fontsize=8,
        )
        axes[0].set_ylim(-1.2, 1.65)
        axes[1].set(xlabel="Time (s)", ylabel="Executed speed (m/s)", xlim=(0, 10))
    else:
        axes[0].add_patch(
            Circle(
                obstacle.position,
                obstacle.radius,
                facecolor="0.6",
                edgecolor="0.2",
                alpha=0.8,
            )
        )
        axes[0].set_ylim(-1.85, 2.05)
        axes[1].set(
            xlabel="Time (s)",
            ylabel=r"Autonomous authority $\alpha$",
            ylim=(-0.04, 1.04),
        )
    axes[0].set(xlabel="x (m)", ylabel="y (m)", xlim=(-0.15, 4.3))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("(a) Closed-loop trajectories", loc="left")
    axes[1].set_title("(b) Executed response", loc="left")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path("results/refinement/holdout_950001.jsonl"),
            Path("results/refinement/holdout_950002.jsonl"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    args = parser.parse_args()
    selection = json.loads(Path("artifacts/e1/refinement_selection.json").read_text())
    rows = read_rows(args.inputs, selection)
    values = summary(rows)
    output = args.output_dir
    for directory in (output / "figures", output / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    write_table(values, output / "tables/e1_refined.tex")
    case_figure(
        rows,
        "crossing",
        ("single_step", "weighted_sum", "ours"),
        output / "figures/e1_crossing_mechanism",
    )
    case_figure(
        rows,
        "conflict_human_unsafe",
        ("human_filter", "delay_agreement", "weighted_sum", "ours"),
        output / "figures/e1_conflict_mechanism",
    )
    payload = {
        "scope": "local held-out validation; two network seeds, fixed scenario geometries",
        "runs": len(rows),
        "selection": selection,
        "methods": values,
        "by_scenario": {
            scenario: summary([r for r in rows if r["scenario"] == scenario])
            for scenario in selection["scenarios"]
        },
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.inputs
        },
    }
    (output / "e1_refined_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    (output / "README.md").write_text(
        "# E1 精简论文产物\n\n"
        "两幅机制图、一张总体结果表；PDF 用于论文，PNG 用于预览。LaTeX 表需要 `booktabs`。\n\n"
        "数据为冻结参数后两个新网络 seed 的局部验证（7 场景 × 4 网络 × 2 seed × 5 方法）。"
        "不是旧 7000 条全量实验的重新汇总，也不把两个 seed 当成 56 个独立随机场景。\n\n"
        "主表保留碰撞、成功、自主权预算和最终执行修改；机制图展示具体轨迹和时间过程，未重复总体统计。"
        "其他诊断和所有失败试验保存在 `results/refinement/` 及本目录 JSON 中。\n\n"
        "复现图表：`.venv/bin/python scripts/publish_refined_e1.py`。"
        "实验写作见 `docs/Mypaper.md`，调优记录见 `docs/MyE1-refinement.md`。\n"
    )
    print(json.dumps({"runs": len(rows), "methods": values}, indent=2))


if __name__ == "__main__":
    main()
