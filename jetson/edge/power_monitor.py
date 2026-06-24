#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Power / energy sampling for NVIDIA Jetson via ``tegrastats``.

Jetson boards expose on-board power rails (mW) through ``tegrastats``. This
module samples the total-board rail during a measurement window and integrates
it to report average power (W), peak power (W) and energy (Joules). Dividing the
energy by the number of processed flows yields the energy-per-inference (mJ) —
a key efficiency metric for an edge IDS that the latency/throughput numbers
alone do not capture.

Graceful by design: if ``tegrastats`` is absent (e.g. running off-Jetson or in a
container), ``available`` is False and the metrics come back as ``None`` so the
benchmark still runs everywhere.

Usage:
    pm = PowerMonitor(interval_ms=200).start()
    ... workload ...
    stats = pm.stop()   # {avg_power_w, peak_power_w, energy_j, samples, ...}
"""
import re
import shutil
import statistics
import subprocess
import threading
import time

# Rail naming differs across Jetson generations; try the whole-board input rail
# first, then fall back to summing the main compute rails.
_PREFERRED_TOTAL_RAILS = ("VDD_IN", "POM_5V_IN", "VDD_SYS_5V0")
_RAIL_RE = re.compile(r"([A-Za-z0-9_]+)\s+(\d+)mW/\d+mW")


def _parse_power_w(line: str):
    """Return instantaneous total-board power in Watts from a tegrastats line."""
    rails = {k: int(v) for k, v in _RAIL_RE.findall(line)}
    if not rails:
        return None
    for key in _PREFERRED_TOTAL_RAILS:
        if key in rails:
            return rails[key] / 1000.0
    # Fall back: sum the dominant compute/input rails.
    total = sum(mw for k, mw in rails.items()
                if k.endswith("_IN") or "GPU" in k or "CPU" in k or "SOC" in k)
    return total / 1000.0 if total else None


class PowerMonitor:
    def __init__(self, interval_ms: int = 200, tegrastats: str = "tegrastats"):
        self.interval_ms = interval_ms
        self.tegrastats = tegrastats
        self.available = shutil.which(tegrastats) is not None
        self._proc = None
        self._thread = None
        self._samples = []          # instantaneous power (W)
        self._stop = threading.Event()
        self._t0 = None

    def _reader(self):
        try:
            self._proc = subprocess.Popen(
                [self.tegrastats, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except Exception:
            self.available = False
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            p = _parse_power_w(line)
            if p is not None:
                self._samples.append(p)

    def start(self):
        if not self.available:
            return self
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        duration_s = (time.time() - self._t0) if self._t0 else 0.0
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2)

        if not self.available or not self._samples:
            return {
                "power_available": False,
                "avg_power_w": None, "peak_power_w": None,
                "energy_j": None, "power_samples": 0,
                "duration_s": round(duration_s, 2),
            }
        avg_w = statistics.mean(self._samples)
        return {
            "power_available": True,
            "avg_power_w": round(avg_w, 2),
            "peak_power_w": round(max(self._samples), 2),
            # Energy = average power × wall-clock duration of the window.
            "energy_j": round(avg_w * duration_s, 2),
            "power_samples": len(self._samples),
            "duration_s": round(duration_s, 2),
        }
