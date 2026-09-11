# Minimal-intervention shared control

`docs/paper.md` is the canonical source for algorithm definitions and variable semantics.
`docs/paper-code-audit.md` maps the paper equations to the implementation and records the
remaining approximations and author-confirmation items.

Research baseline for delayed differential-drive teleoperation. The WSL side contains a
deterministic 2-D simulator, timestamp-aware bidirectional network model, predictors,
multi-step chance constraints, lexicographic authority allocation, an independent robust
CBF safety filter, controlled baselines, metrics, and public-dataset adapters.

## WSL quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
.venv/bin/misc-run --scenario crossing --method ours --seed 7
```

Run a paired smoke matrix and write trajectory-level JSON records:

```bash
.venv/bin/python scripts/run_experiments.py \
  --scenarios bend crossing conflict network_anomaly \
  --methods delay_agreement single_step weighted_sum human_filter ours \
  --seeds 0 1 2 --output results/smoke.jsonl
```

Run the controlled E2 replay with exactly the same nominal-command log:

```bash
.venv/bin/python scripts/run_safety_isolation.py --scenario network_anomaly
```

Run the leakage-controlled E0 fit after preprocessing the selected SCAND Jackal bags and
the THÖR TSV files:

```bash
.venv/bin/python scripts/run_e0.py
```

This writes full predictions under ignored `results/e0/` and publishes only the small,
reproducible model parameters and metric summaries under `artifacts/e0/`. The 2-D simulator
and the platform-neutral `SharedControlRuntime.step(StepInput)` load those same artifacts.
The crossing scenario replays a fixed held-out THÖR track when the processed data exists;
otherwise its records explicitly report `synthetic-fallback`.

The default is intentionally a fast smoke configuration. Paper-scale runs should increase
the duration and paired seeds through experiment YAML files. Raw SCAND and THOR data are
not redistributed; see `datasets/*/README.md` for placement and conversion commands.

After E0 preprocessing, select compact predictor settings without touching the held-out
test partitions:

```bash
.venv/bin/python scripts/tune_e0.py
```

Run the complete resumable E1 matrix from `configs/experiments/paper.yaml` and generate
paper-ready summaries under `results/full/`:

```bash
.venv/bin/python scripts/run_full_wsl.py --jobs 6
```

## Platform boundary

Everything under `src/minimal_intervention_shared_control`, dataset preprocessing, tests,
batch experiments, and analysis is designed to run in WSL. Windows is reserved for native
Webots, the differential-drive world/controller bridge, gamepad acquisition, and human
participant trials. No ROS installation is required for this version.

Windows must use its own Python virtual environment; a native Windows process cannot import
the Linux binaries in WSL's `.venv`. Clone/pull the same commit on Windows and install with
`py -m pip install -e .`. Source and calibrated artifacts then stay aligned through Git;
the multi-gigabyte source datasets do not need to be copied to Windows. See
`docs/platform-boundary.md` for the exact controller contract.
