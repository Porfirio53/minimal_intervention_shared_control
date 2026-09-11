# SCAND data

Download the Jackal ROS bags from the official SCAND data page and place them in `raw/`.
The Texas Data Repository currently returns HTTP 403 to this WSL environment, so use a
Windows browser for the download:

1. Open https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH.
2. Accept any displayed repository terms and download the files belonging to the
   Clearpath Jackal. Keep the original file names and run boundaries; do not mix Spot
   bags into this first pipeline.
3. Copy the downloaded bags into this WSL directory. From WSL, for example:

```bash
mkdir -p datasets/scand/raw
cp -av /mnt/c/Users/<WindowsUser>/Downloads/<RUN.bag> datasets/scand/raw/
```

Raw and processed data are intentionally ignored by Git. Before conversion, inspect the
bag's actual topics because release-specific names can differ:

```bash
.venv/bin/python - datasets/scand/raw/<RUN.bag> <<'PY'
import sys
from pathlib import Path
from rosbags.highlevel import AnyReader

with AnyReader([Path(sys.argv[1])]) as reader:
    for connection in sorted(reader.connections, key=lambda item: item.topic):
        print(connection.topic, connection.msgtype)
PY
```

Convert each run only after identifying the command and odometry topics:

```bash
python scripts/preprocess_scand.py datasets/scand/raw/RUN.bag \
  datasets/scand/processed/RUN.csv --run-id RUN --driver-id DRIVER \
  --command-topic /bluetooth_teleop/joy --command-format jackal-ps4-joy \
  --odometry-topic /jackal_velocity_controller/odom
```

Override both topic options when a release uses different names. The Jackal Joy conversion
uses Clearpath's PS4 configuration from the collection period: axes 1/0 are linear/angular,
buttons 4/5 are normal/turbo enable, and scales are 0.4/2.0 m/s and 1.4 rad/s. Splitting is
performed by complete run (or leave-one-driver-out), never by randomly shuffling individual
timestamps.
