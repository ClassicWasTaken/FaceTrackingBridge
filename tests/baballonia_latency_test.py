# Copyright (c) 2026 Face Tracking Bridge contributors.
# SPDX-License-Identifier: Apache-2.0

"""Timestamp real Baballonia OSC input -> captured VRChat output on loopback.

Supply the official module ZIP separately. Every process, input port, avatar,
module copy and data profile is isolated. This measures the bridge/module path,
not camera inference, headset transport, avatar animation or display latency.
No upstream-app performance claim is inferred from this benchmark.
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import random
import shutil
import socket
import statistics
import tempfile
import threading
import time

from integration_test import osc_message, osc_string, read_osc, udp_socket
from native_integration_test import native_payloads
from provider_integration_test import Fixture, wait_for


class TimedCapture:
    def __init__(self, capture, native, legacy_prefix):
        self.capture = capture
        self.native = native
        self.legacy_prefix = legacy_prefix
        self.events = deque(maxlen=16000)
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.failure = None
        capture.settimeout(.2)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            while not self.stopped.is_set():
                try:
                    packet, _ = self.capture.recvfrom(65535)
                    received = time.perf_counter_ns()  # Before decoding or main-thread scheduling.
                except socket.timeout:
                    continue
                values = {}
                if self.native:
                    for payload in native_payloads(packet):
                        values.update(jaw=payload[36] / 255, eye=1 - payload[9] / 255)
                else:
                    for address, value in read_osc(packet):
                        if address == f"/avatar/parameters/{self.legacy_prefix}/JawOpen":
                            values["jaw"] = value
                        elif address == f"/avatar/parameters/{self.legacy_prefix}/EyeOpenLeft":
                            values["eye"] = value
                if values:
                    with self.condition:
                        self.events.append((received, values))
                        self.condition.notify_all()
        except Exception as error:
            if not self.stopped.is_set():
                self.failure = repr(error)

    def response(self, sent, previous, target, timeout=.5):
        found = {channel: {} for channel in ("jaw", "eye")}
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                for timestamp, values in self.events:
                    if timestamp < sent:
                        continue
                    for channel, value in values.items():
                        progress = (value - previous) / (target - previous)
                        for threshold in (.1, .5, .9):
                            if progress >= threshold and threshold not in found[channel]:
                                found[channel][threshold] = (timestamp - sent) / 1e6
                if all(len(result) == 3 for result in found.values()):
                    return found
                if self.failure:
                    raise AssertionError(self.failure)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"Missing timely step response: {found}")
                self.condition.wait(min(.1, remaining))

    def wait_for_values(self, expected, timeout=3):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                latest = {}
                for _, values in self.events:
                    latest.update(values)
                if all(abs(latest.get(channel, -10) - expected) <= .005 for channel in ("jaw", "eye")):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.failure:
                    raise AssertionError(f"Output route did not finish warm-up: {latest}; {self.failure}")
                self.condition.wait(min(.1, remaining))

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1)
        assert not self.thread.is_alive(), "Capture thread did not stop."


def summary(values):
    ordered = sorted(values)
    return {"samples": len(values), "mean_ms": round(statistics.mean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(ordered[math.ceil(.95 * len(ordered)) - 1], 3),
            "max_ms": round(ordered[-1], 3)}


def run(executable, archive, samples, mode, native, root, legacy_prefix="v2", legacy_eye_addresses=False):
    fixture = Fixture(root, archive)
    if legacy_prefix != "v2":
        avatar_path = next((root / "LocalLow").rglob("avtr_provider_test.json"))
        avatar = json.loads(avatar_path.read_text(encoding="utf-8"))
        for parameter in avatar["parameters"]:
            parameter["name"] = parameter["name"].replace("v2/", legacy_prefix + "/", 1)
            parameter["input"]["address"] = "/avatar/parameters/" + parameter["name"]
        avatar_path.write_text(json.dumps(avatar), encoding="utf-8")
    eye_port = fixture.input_port
    face_port = eye_port
    extra = ["--input-port", eye_port]
    if mode == "split":
        face_dir = root / "face-module"
        shutil.copytree(fixture.module_dir, face_dir)
        face_module = face_dir / fixture.manifest["DllFileName"]
        reservation = udp_socket()
        face_port = reservation.getsockname()[1]
        reservation.close()
        extra += ["--face-module", face_module, "--face-kind", "BaballoniaDesktop",
                  "--face-name", "Isolated Baballonia face", "--face-input-port", face_port]
    capture = TimedCapture(fixture.capture, native, legacy_prefix)
    worker = fixture.worker(executable, "BaballoniaDesktop", extra)
    def send(value):
        sent = time.perf_counter_ns()
        fixture.send(face_port, {"/jawOpen": value})
        eye_names = ("/LeftEyeLid", "/RightEyeLid") if legacy_eye_addresses else ("/leftEyeLid", "/rightEyeLid")
        fixture.send(eye_port, {name: value for name in eye_names})
        return sent
    results = {channel: {str(threshold): [] for threshold in (.1, .5, .9)} for channel in ("jaw", "eye")}
    try:
        wait_for(worker, "latency fixture readiness", lambda: worker.any(
            lambda record: record.get("type") == "ready" and
            record.get("source") == ("Combined" if mode == "split" else "BaballoniaDesktop")))
        wait_for(worker, "mapper and avatar listener are running", lambda: worker.any(
            lambda record: record.get("type") == "status" and record.get("trackingRate", 0) > 0))
        # Announce once. Repeated avatar changes would reset routing and distort latency.
        fixture.transmit.sendto(osc_message("/avatar/change", "avtr_provider_test"),
                                ("127.0.0.1", fixture.receive_port))
        if native:
            fixture.transmit.sendto(osc_string("/avatar/avatarfullfacetracking") + osc_string(",T"),
                                    ("127.0.0.1", fixture.receive_port))
        wait_for(worker, "avatar route was explicitly bound", lambda: worker.any(
            lambda record: record.get("type") == "log" and
            record.get("message", "").startswith("Avatar: Provider Test.")))
        for index in range(12):
            send(.2 if index % 2 == 0 else .8)
            time.sleep(.08)
        capture.wait_for_values(.8)
        previous = .8
        rng = random.Random(13017)
        # Each step remains present long enough for a complete response; jitter
        # prevents a convenient phase relationship with either sampling clock.
        for index in range(samples):
            target = rng.uniform(.12, .3) if index % 2 == 0 else rng.uniform(.7, .88)
            sent = send(target)
            response = capture.response(sent, previous, target)
            for channel, thresholds in response.items():
                for threshold, duration in thresholds.items():
                    results[channel][str(threshold)].append(duration)
            previous = target
            time.sleep(rng.uniform(.04, .075))
            assert worker.process.poll() is None, worker.diagnostics()
        return {"mode": mode, "output": "native" if native else "legacy", "legacy_prefix": legacy_prefix,
                "legacy_eye_addresses": legacy_eye_addresses,
                "step_response": {channel: {f"t{int(float(threshold) * 100)}": summary(values)
                                             for threshold, values in thresholds.items()}
                                  for channel, thresholds in results.items()}}
    except Exception as error:
        raise AssertionError(str(error) + "\n" + worker.diagnostics()) from error
    finally:
        worker.close()
        capture.close()
        fixture.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--module-zip", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--mode", choices=("shared", "split", "both"), default="both")
    parser.add_argument("--output", choices=("legacy", "native", "both"), default="both")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--legacy-prefix", choices=("v2", "FT/v2"), default="v2")
    parser.add_argument("--legacy-eye-addresses", action="store_true",
                        help="Use the uppercase eye addresses required by the older 3.1 module.")
    args = parser.parse_args()
    if not 8 <= args.samples <= 1000:
        parser.error("Use 8..1000 steps per scenario.")
    modes = ("shared", "split") if args.mode == "both" else (args.mode,)
    formats = (False, True) if args.output == "both" else (args.output == "native",)
    results = []
    def save_report():
        if args.report:
            report = {"worker": str(args.worker.resolve()), "module_zip": str(args.module_zip.resolve()),
                      "scope": "Synthetic OSC input to loopback VRChat packet, no camera/avatar/display latency",
                      "results": results}
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="ftb-baballonia-latency-") as temporary:
        for mode in modes:
            for native in formats:
                root = Path(temporary) / f"{mode}-{'native' if native else 'legacy'}"
                result = run(args.worker.resolve(strict=True), args.module_zip.resolve(strict=True),
                             args.samples, mode, native, root, args.legacy_prefix, args.legacy_eye_addresses)
                results.append(result)
                save_report()
                print(json.dumps(result), flush=True)
