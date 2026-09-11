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

The default is intentionally a fast smoke configuration. Paper-scale runs should increase
the duration and paired seeds through experiment YAML files. Raw SCAND and THOR data are
not redistributed; see `datasets/*/README.md` for placement and conversion commands.

## Platform boundary

Everything under `src/minimal_intervention_shared_control`, dataset preprocessing, tests,
batch experiments, and analysis is designed to run in WSL. Windows is reserved for native
Webots, the differential-drive world/controller bridge, gamepad acquisition, and human
participant trials. No ROS installation is required for this version.
