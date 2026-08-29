#!/usr/bin/env python3
"""Minimal MCP + ROS2 face-plugin smoke client for Jetson validation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


def mcp_call(url: str, arguments: dict, *, timeout: float) -> dict:
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {"name": "face", "arguments": arguments},
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        rpc = json.loads(response.read())
    if "error" in rpc:
        raise RuntimeError(rpc["error"])
    return json.loads(rpc["result"]["content"][0]["text"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--topic", default="/codex/face/smoke")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--restart-each", action="store_true")
    parser.add_argument("--dds-wait", type=float, default=0.0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--expected-person-id")
    args = parser.parse_args()
    if args.repeat < 1 or args.dds_wait < 0:
        parser.error("--repeat must be >= 1 and --dds-wait must be >= 0")

    status = mcp_call(
        args.mcp_url,
        {"action": "start", "input_topic": args.topic},
        timeout=args.timeout,
    )
    output_topic = status.get("output") or f"{args.topic}/face"
    qos_input = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
    qos_output = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    rclpy.init(args=None)
    node = rclpy.create_node(f"face_smoke_{int(time.time() * 1000)}")
    publisher = node.create_publisher(CompressedImage, args.topic, qos_input)
    received: list[dict] = []

    def on_result(message: String) -> None:
        received.append(json.loads(message.data))

    node.create_subscription(String, output_topic, on_result, qos_output)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    latencies: list[float] = []
    correct = 0
    try:
        encoded = list(args.image.read_bytes())
        for index in range(args.repeat):
            if index > 0 and args.restart_each:
                status = mcp_call(
                    args.mcp_url,
                    {"action": "start", "input_topic": args.topic},
                    timeout=args.timeout,
                )
                output_topic = status.get("output") or output_topic
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                executor.spin_once(timeout_sec=0.1)
                if publisher.get_subscription_count() and node.count_publishers(output_topic):
                    break
            else:
                raise TimeoutError("face ROS2 publisher/subscriber did not connect")
            if args.dds_wait:
                time.sleep(args.dds_wait)
            message = CompressedImage()
            message.header.stamp = node.get_clock().now().to_msg()
            message.format = args.image.suffix.lstrip(".") or "jpeg"
            message.data = encoded
            started = time.perf_counter()
            publisher.publish(message)
            expected_count = index + 1
            while time.time() < deadline and len(received) < expected_count:
                executor.spin_once(timeout_sec=0.05)
            if len(received) < expected_count:
                raise TimeoutError(f"face result {expected_count} was not received")
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            payload = received[-1]
            identity = payload.get("identity") or {}
            if (
                args.expected_person_id is not None
                and identity.get("person_id") == args.expected_person_id
            ):
                correct += 1
            if not args.summary_only:
                print(
                    json.dumps(
                        {"case": expected_count, "latency_ms": round(latency_ms, 3), **payload},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if args.restart_each:
                mcp_call(args.mcp_url, {"action": "stop"}, timeout=args.timeout)
    finally:
        try:
            mcp_call(args.mcp_url, {"action": "stop"}, timeout=args.timeout)
        finally:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    summary = {
        "type": "summary",
        "cases": len(latencies),
        "latency_mean_ms": round(statistics.mean(latencies), 3),
        "latency_p50_ms": round(statistics.median(latencies), 3),
        "latency_p95_ms": round(ordered[p95_index], 3),
        "latency_max_ms": round(max(latencies), 3),
    }
    if args.expected_person_id is not None:
        summary["expected_person_id"] = args.expected_person_id
        summary["correct"] = correct
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.expected_person_id is not None and correct != len(latencies):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
