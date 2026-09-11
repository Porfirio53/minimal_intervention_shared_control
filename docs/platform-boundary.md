# Platform boundary

## WSL (implemented in this repository)

- Python package, differential-drive dynamics, paths and local control.
- Independent timestamped downlink/uplink FIFO channels with delay, jitter, correlation,
  packet loss, and causal command age.
- Transparent human AR/behavior-cloning models, constant-velocity obstacle prediction,
  covariance calibration, prediction tubes, and chance margins.
- Multi-step sensitivity constraints, lexicographic OSQP allocation, four controlled
  baselines, and an execution-side hard robust CBF QP that cannot consume human covariance.
- Paper-to-code traceability, deterministic equation tests, explicit upper/filter
  infeasibility metrics, and numerical/physical-bound validation.
- Four deterministic 2-D scenario families, paired batch runner, trajectory-level metrics,
  bootstrap utility, and plotting hook.
- SCAND ROS-bag extraction and THOR TSV loading. Dataset downloads remain manual because
  the large source archives and their licenses are external to this Git repository.
- Leakage-controlled E0 fitting, versioned calibrated model artifacts, a held-out THÖR
  crossing replay, and the simulator-independent `SharedControlRuntime.step` entry point.

## Windows (not executable from this WSL-only checkout)

### One-time setup

1. Install 64-bit Python 3.11-3.13 and Git for Windows. Do not reuse `.venv` from WSL;
   native Windows and Linux Python packages are binary-incompatible.
2. Clone this repository to an ordinary Windows path and check out the same commit used by
   WSL. Keeping a second clone is more robust for Webots than importing through `\\wsl$`.
3. In PowerShell, create and install the Windows environment:

```powershell
git clone <repository-url> C:\work\minimal_intervention_shared_control
Set-Location C:\work\minimal_intervention_shared_control
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --upgrade pip
.\.venv-win\Scripts\python.exe -m pip install -e ".[test]"
.\.venv-win\Scripts\python.exe -m pytest
```

The tracked `artifacts/e0/` directory contains the small SCAND/THÖR model parameters and
summaries, so Windows does not need the raw bags or 3-D TSV files. Sync code and artifacts
with normal Git commits. Do not commit `results/`, raw datasets, participant data, or keys.

### Webots controller contract

Create one `SharedControlRuntime(RuntimeConfig(...))` when the Webots controller starts and
call `runtime.step(StepInput(...))` every `T_s=0.02 s`. The controller must provide:

- `now`: Windows monotonic time in seconds;
- `state`: current `[p_x, p_y, psi]` in metres/radians;
- newly received `CommandEnvelope(Control, s_m, g_m, r_m)` objects;
- one local-planner candidate sequence shaped `(N, 2)` in `[v, omega]`;
- current obstacle mean position/velocity for upper prediction;
- one independently timestamped `RelativeStateObservation` per obstacle for the execution
  filter, with its own `tau^S` and covariance blocks.

Apply only `StepOutput.filtered_control` to Webots. Log `nominal_control`,
`filtered_control`, `alpha`, `human_information_age`, both QP statuses, timestamps, state,
and obstacle observations. Never pass `risk_tube.human_intent.covariance` into the safety
observation or filter.

The order of `obstacles` and `safety_observations` must match. All axes, signs, units, robot
radius, obstacle radii, and speed limits must be checked against the Webots world before a
trial. The current Python configuration values are research placeholders until measured on
the Windows platform.

### Remaining Windows work

- Install native Webots and create/validate the differential-drive UGV world with dynamic
  pedestrians and obstacles.
- Import and exercise the packaged `SharedControlRuntime` from a native Webots Python
  controller. IPC is unnecessary with the recommended separate Windows clone/environment.
- Implement the same application-level downlink/uplink timestamp FIFOs in the Webots
  controller; validate timestamps against Windows' monotonic clock.
- Bridge one timestamped relative-state observation per obstacle, including the source-time
  position/velocity covariance blocks and `tau^S`; do not pass human-intent covariance into
  the execution filter.
- Measure and configure the real linear/angular speed, rate, wheel-speed, command-hold, and
  obstacle-speed limits. All actuator limits must be inside the optimization rather than in
  an unreported post-optimization clamp.
- Calibrate execution covariance, relative-state covariance, acceleration, clock, model, and
  actuation error bounds from held-out platform logs; verify initial `H^-` and empirical
  confidence coverage before making any safety-probability claim.
- Connect an Xbox/DualSense gamepad and verify axis mapping, dead zones, emergency stop,
  command saturation, focus-loss behavior, and trial logging.
- Pilot and then run the participant protocol (B1, B4, Ours), after institutional ethics
  review/informed consent where required. Randomize/counterbalance conditions and retain
  one complete log per participant/trial.

ROS 2 and CARLA are intentionally outside the first version described in `docs/design.md`.
