#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from minimal_intervention_shared_control.datasets.scand import (
    causal_hold_indices,
    jackal_ps4_joy_to_command,
)


def stamp_seconds(timestamp: int) -> float:
    return timestamp / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract normalized command/odometry rows from one SCAND ROS bag"
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--driver-id", required=True)
    parser.add_argument("--command-topic", default="/cmd_vel")
    parser.add_argument(
        "--command-format",
        choices=("twist", "jackal-ps4-joy"),
        default="twist",
        help="interpret command-topic as Twist or SCAND Jackal PS4 Joy",
    )
    parser.add_argument("--odometry-topic", default="/odometry/filtered")
    args = parser.parse_args()
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise SystemExit(
            "install the data extra first: pip install -e '.[data]'"
        ) from error
    commands: list[tuple[float, float, float]] = []
    odometry: list[tuple[float, float, float, float]] = []
    with AnyReader([args.bag]) as reader:
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {args.command_topic, args.odometry_topic}
        ]
        if len({connection.topic for connection in selected}) < 2:
            available = sorted({connection.topic for connection in reader.connections})
            raise SystemExit(
                f"requested topics were not both present; available topics: {available}"
            )
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            stamp = stamp_seconds(timestamp)
            if connection.topic == args.command_topic:
                if args.command_format == "jackal-ps4-joy":
                    v, omega = jackal_ps4_joy_to_command(message.axes, message.buttons)
                else:
                    twist = getattr(message, "twist", message)
                    v, omega = float(twist.linear.x), float(twist.angular.z)
                commands.append((stamp, v, omega))
            else:
                pose = message.pose.pose
                quaternion = pose.orientation
                yaw = __import__("math").atan2(
                    2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                    1 - 2 * (quaternion.y**2 + quaternion.z**2),
                )
                odometry.append(
                    (stamp, float(pose.position.x), float(pose.position.y), yaw)
                )
    if not commands or not odometry:
        raise SystemExit("bag did not contain usable command and odometry messages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command_indices = causal_hold_indices(
        [command[0] for command in commands],
        [sample[0] for sample in odometry],
    )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["run_id", "driver_id", "timestamp", "x", "y", "yaw", "v", "omega"]
        )
        for (timestamp, x, y, yaw), command_index in zip(odometry, command_indices):
            if command_index < 0:
                continue
            _, v, omega = commands[command_index]
            writer.writerow(
                [args.run_id, args.driver_id, timestamp, x, y, yaw, v, omega]
            )


if __name__ == "__main__":
    main()
