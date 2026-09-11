#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
                twist = getattr(message, "twist", message)
                commands.append((stamp, float(twist.linear.x), float(twist.angular.z)))
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
    command_index = 0
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["run_id", "driver_id", "timestamp", "x", "y", "yaw", "v", "omega"]
        )
        for timestamp, x, y, yaw in odometry:
            while command_index + 1 < len(commands) and abs(
                commands[command_index + 1][0] - timestamp
            ) <= abs(commands[command_index][0] - timestamp):
                command_index += 1
            _, v, omega = commands[command_index]
            writer.writerow(
                [args.run_id, args.driver_id, timestamp, x, y, yaw, v, omega]
            )


if __name__ == "__main__":
    main()
