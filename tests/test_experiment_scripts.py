import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_config_expands_seed_count_and_networks() -> None:
    module = _load_script("run_experiments.py")
    settings = module._resolve_settings(
        Namespace(
            config=Path("configs/experiments/paper.yaml"),
            scenarios=None,
            methods=None,
            networks=None,
            network=None,
            duration=None,
            seeds=None,
            seed_start=100,
        )
    )
    assert settings["seeds"] == list(range(100, 150))
    assert settings["network_conditions"] == ["N0", "N1", "N2", "NJ"]
    assert "conflict_blend_unsafe" in settings["scenarios"]


def test_resume_key_reads_completed_experiment(tmp_path: Path) -> None:
    module = _load_script("run_experiments.py")
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps(
            {
                "scenario": "bend",
                "method": "ours",
                "seed": 4,
                "network_condition": "N1",
                "duration": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert module._existing_keys(path) == {("bend", "ours", 4, "N1", 3.0)}


def test_summary_writes_paired_paper_outputs(tmp_path: Path) -> None:
    module = _load_script("summarize_results.py")
    rows = []
    for seed in (0, 1):
        for method, collision in (("ours", False), ("human_filter", True)):
            rows.append(
                {
                    "scenario": "crossing",
                    "network_condition": "N1",
                    "seed": seed,
                    "duration": 3.0,
                    "method": method,
                    **{metric: 0.0 for metric in module.PRIMARY_METRICS},
                    "collision": collision,
                }
            )
    paired = module._paired(rows, "ours")
    collision = next(row for row in paired if row["metric"] == "collision")
    assert collision["mean_difference_reference_minus_baseline"] == -1.0
    overall = module._aggregate(rows, ("method",))
    module._write_conclusions(tmp_path / "summary.md", rows, overall, paired)
    assert "Ours minus baseline" in (tmp_path / "summary.md").read_text(
        encoding="utf-8"
    )
