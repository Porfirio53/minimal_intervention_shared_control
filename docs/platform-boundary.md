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

## Windows (not executable from this WSL-only checkout)

- Install native Webots and create/validate the differential-drive UGV world with dynamic
  pedestrians and obstacles.
- Point a Webots Python controller at this shared algorithm package, or expose a narrow
  IPC bridge if Windows and WSL use separate Python installations.
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
