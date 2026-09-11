# SCAND data

Download the Jackal ROS bags from the official SCAND data page and place them in `raw/`.
Raw and processed data are intentionally ignored by Git. Convert each run with:

```bash
python scripts/preprocess_scand.py datasets/scand/raw/RUN.bag \
  datasets/scand/processed/RUN.csv --run-id RUN --driver-id DRIVER \
  --command-topic /cmd_vel --odometry-topic /odometry/filtered
```

Inspect the bag's topic list and override both topic options when a release uses different
names. Splitting is performed by complete run (or leave-one-driver-out), never by randomly
shuffling individual timestamps.
