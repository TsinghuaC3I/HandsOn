#!/usr/bin/env python3
"""Exercise real ROS/DDS, Wuji retargeting, calibration and raw UDP recording.

Source the ROS environment and MANUS workspace before running this script.
All topics are remapped and UDP destinations are private ephemeral loopback
ports. No hardware is enabled. --shared-executor provides a negative control
where raw work shares one serial executor with hand/control callbacks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import socket
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--shared-executor", action="store_true")
    options = parser.parse_args()
    if options.seconds < 8:
        parser.error("--seconds must be at least 8")

    import numpy as np
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import Pose, TransformStamped
    from manus_ros2_msgs.msg import ManusErgonomics, ManusGlove, ManusRawNode
    from manus_input_py import manus_input_node as input_module
    from tf2_msgs.msg import TFMessage
    import yaml

    from teleoperation.manus import manus_wuji_ros2_bridge as bridge
    from handson.data.manus_raw_recording import decode_manus_raw_event

    vendor = ROOT / "teleoperation/manus/vendor/wuji-hand-teleop"
    stop = threading.Event()
    block_raw = threading.Event()
    raw_blocked = threading.Event()
    release_raw = threading.Event()
    failures: list[str] = []
    worker_errors: list[str] = []
    timer_starts: list[float] = []
    timer_gate = threading.Lock()

    class MeasuredBridge(bridge.ManusWujiHandsOnBridge):
        def _raw_glove_cb(self, message, topic):
            if block_raw.is_set() and not raw_blocked.is_set():
                raw_blocked.set()
                if not release_raw.wait(2.0):
                    worker_errors.append("raw blocking probe timed out")
            super()._raw_glove_cb(message, topic)

        def _timer_cb(self):
            if not timer_gate.acquire(blocking=False):
                worker_errors.append("control timer reentered")
                return
            try:
                timer_starts.append(time.monotonic())
                super()._timer_cb()
            finally:
                timer_gate.release()

    def worker(function):
        try:
            function()
        except Exception as exc:
            worker_errors.append(f"{function.__name__}: {type(exc).__name__}: {exc}")
            stop.set()

    sockets = {}
    packets = {name: [] for name in ("right", "left", "tracker", "raw")}
    threads = []
    nodes = []
    executors = []
    controller = None
    result = {}
    namespace = f"/handson_callback_load_{os.getpid()}"
    ros_args = ["--ros-args"]
    for topic in ("hand_input", "manus_glove_0", "manus_glove_1", "tf", "tf_static"):
        ros_args.extend(("-r", f"/{topic}:={namespace}/{topic}"))
    rclpy.init(args=ros_args)
    try:
        with tempfile.TemporaryDirectory(prefix="handson-manus-load-") as directory:
            temp = Path(directory)
            mapping = bridge.load_yaml(ROOT / "teleoperation/manus/config/handson_manus_mapping.yaml")
            mapping["vr_to_mujoco"] = np.eye(4).tolist()
            calibration = mapping["startup_calibration"]
            calibration["settle_sec"] = 0.6
            calibration["motion_sec"] = 0.8
            mapping_path = temp / "mapping.yaml"
            mapping_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
            for name in packets:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                sock.bind(("127.0.0.1", 0))
                sockets[name] = sock
            args = bridge.build_argparser().parse_args([
                "--hand", "both", "--hand-type", "wuji1", "--send-tracker",
                "--frequency", "60", "--mapping-config", str(mapping_path),
                "--manus-wuji-root", str(vendor), "--force-recalibrate",
                "--openvr-config", str(mapping_path),
                "--calibration-cache", str(temp / "cache.json"),
                "--calibration-cache-output", str(temp / "cache.json"),
                "--tracker-control-port", "0",
                "--right-hand-port", str(sockets["right"].getsockname()[1]),
                "--left-hand-port", str(sockets["left"].getsockname()[1]),
                "--tracker-port", str(sockets["tracker"].getsockname()[1]),
                "--raw-recording-port", str(sockets["raw"].getsockname()[1]),
            ])
            controller = MeasuredBridge(args)
            nodes.append(controller)
            input_node = input_module.ManusInputNode(input_module.ManusInputConfig(publish_rate_hz=120.0))
            nodes.append(input_node)
            source = Node("handson_manus_load_source")
            nodes.append(source)
            qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
            publishers = [source.create_publisher(ManusGlove, f"/manus_glove_{i}", qos) for i in range(2)]
            tf_publisher = source.create_publisher(TFMessage, "/tf", qos)
            if options.shared_executor:
                executor = SingleThreadedExecutor()
                for node in (controller, *controller.receiver_nodes):
                    executor.add_node(node)
            else:
                executor = bridge.BridgeExecutor(controller)
            input_executor = SingleThreadedExecutor()
            input_executor.add_node(input_node)
            executors.extend((executor, input_executor))

            messages = []
            for index, side in enumerate(("Right", "Left")):
                msg = ManusGlove(glove_id=index + 1, side=side)
                for point, node_id in enumerate(input_module.MEDIAPIPE_TO_MANUS):
                    node = ManusRawNode(node_id=node_id, parent_node_id=1, joint_type="Joint", chain_type="Finger")
                    if point:
                        finger, joint = divmod(point - 1, 4)
                        node.pose.position.x = float((finger - 2) * 0.025 * (1 if index == 0 else -1))
                        node.pose.position.y = float(-0.035 - joint * 0.022)
                        node.pose.position.z = 0.005
                    node.pose.orientation.w = 1.0
                    msg.raw_nodes.append(node)
                msg.raw_node_count = len(msg.raw_nodes)
                msg.ergonomics = [ManusErgonomics(type=f"Joint{i}", value=float(i)) for i in range(20)]
                msg.ergonomics_count = len(msg.ergonomics)
                msg.raw_sensor = [Pose() for _ in range(6)]
                msg.raw_sensor_count = len(msg.raw_sensor)
                msg.raw_sensor_orientation.w = 1.0
                messages.append(msg)

            published = [0, 0]

            def publish():
                deadline = time.monotonic()
                phase = "anchor"
                phase_started = deadline
                while not stop.is_set():
                    now = time.monotonic()
                    new_phase = controller._calib_phase
                    if new_phase != phase:
                        phase, phase_started = new_phase, now
                    delta = np.zeros(3)
                    if phase in ("forward", "left", "up"):
                        delta[("forward", "left", "up").index(phase)] = min(0.20, (now - phase_started) * 0.25)
                    transforms = []
                    stamp = source.get_clock().now().to_msg()
                    for index, side in enumerate(("right", "left")):
                        publishers[index].publish(messages[index])
                        published[index] += 1
                        tf = TransformStamped()
                        tf.header.stamp = stamp
                        tf.header.frame_id = "world"
                        tf.child_frame_id = f"{side}_wrist"
                        tf.transform.translation.x = float(delta[0])
                        tf.transform.translation.y = float((-0.3 if index == 0 else 0.3) + delta[1])
                        tf.transform.translation.z = float(1.0 + delta[2])
                        tf.transform.rotation.w = 1.0
                        transforms.append(tf)
                    tf_publisher.publish(TFMessage(transforms=transforms))
                    deadline += 1.0 / 120.0
                    stop.wait(max(0.0, deadline - time.monotonic()))

            def receive():
                names = {sock: name for name, sock in sockets.items()}
                while not stop.is_set():
                    ready, _, _ = select.select(list(names), [], [], 0.05)
                    for sock in ready:
                        packet = sock.recv(65535)
                        name = names[sock]
                        value = decode_manus_raw_event(packet) if name == "raw" else time.monotonic()
                        packets[name].append(value)

            for function in (receive, executor.spin, input_executor.spin, publish):
                thread = threading.Thread(target=worker, args=(function,), daemon=True)
                threads.append(thread)
                thread.start()
            started = time.monotonic()
            # Allow discovery and initial solver warmup before the blocking probe.
            stop.wait(2.0)
            block_raw.set()
            if not raw_blocked.wait(2.0):
                failures.append("raw callback never started")
            before = (controller._msg_count, controller._timer_count)
            stop.wait(0.25)
            during = (controller._msg_count, controller._timer_count)
            release_raw.set()
            if during[0] <= before[0] or during[1] <= before[1]:
                failures.append("raw callback blocked hand input or control timer")
            stop.wait(max(0.0, options.seconds - (time.monotonic() - started)))
            elapsed = time.monotonic() - started
            steady_ticks = [tick for tick in timer_starts if tick >= started + 4.0]
            max_gap = max(np.diff(steady_ticks), default=0.0)
            hand_rates = {side: sum(t >= started + 4.0 for t in packets[side]) / max(0.01, elapsed - 4.0) for side in ("right", "left")}
            if max_gap > 0.2 or len(steady_ticks) / max(0.01, elapsed - 4.0) < 50:
                failures.append("control timer missed its steady-state latency/rate budget")
            if any(rate < 50 for rate in hand_rates.values()):
                failures.append("hand UDP output below 50 Hz")
            if controller._calib_phase != "done":
                failures.append("guided calibration did not complete")
            raw_counts = [sum(row["source_topic"] == f"/manus_glove_{i}" for row in packets["raw"]) for i in range(2)]
            if any(count / elapsed < 100 for count in raw_counts):
                failures.append("raw UDP throughput below 100 Hz per glove")
            sequence_gaps = []
            for i in range(2):
                seq = [row["source_sequence"] for row in packets["raw"] if row["source_topic"] == f"/manus_glove_{i}"]
                sequence_gaps.extend((a, b) for a, b in zip(seq, seq[1:]) if b != a + 1)
            if sequence_gaps:
                failures.append("raw sidecar sequence gaps")
            result = {
                "shared_executor": options.shared_executor, "elapsed_sec": round(elapsed, 3),
                "input_module": input_module.__file__,
                "source_published": published.copy(), "hand_received": controller._msg_count,
                "hand_udp_hz": hand_rates, "timer_hz": len(steady_ticks) / max(0.01, elapsed - 4.0),
                "timer_max_gap_ms": float(max_gap * 1000),
                "calibration_phase": controller._calib_phase,
                "calibration_samples": {phase: {side: len(rows) for side, rows in sides.items()} for phase, sides in controller._guided_samples.items()},
                "raw_udp_received": raw_counts, "raw_sequence_gap_count": len(sequence_gaps),
                "progress_during_raw_block": {"hand": during[0] - before[0], "timer": during[1] - before[1]},
            }
    finally:
        stop.set()
        release_raw.set()
        for executor in executors:
            executor.shutdown()
        for thread in threads:
            thread.join(timeout=3.0)
        if controller is not None:
            controller.close()
        for node in nodes:
            node.destroy_node()
        for sock in sockets.values():
            sock.close()
        rclpy.shutdown()
    result["failures"] = failures + worker_errors
    print(json.dumps(result, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
