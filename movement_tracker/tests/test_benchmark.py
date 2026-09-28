"""
Performance Benchmarking Tests for Prediction & Filtering Engine.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import TelemetryInput
from tracker import EntityTracker


def test_throughput_performance() -> None:
    """
    Measures prediction operations per second for high-frequency processing.
    """
    tracker = EntityTracker(filter_enabled=True)
    telemetry = TelemetryInput(
        entity_id="bench_target",
        x=1.0,
        y=2.0,
        z=3.0,
        vx=0.5,
        vy=0.1,
        vz=0.0,
        timestamp=1000.0,
    )

    iterations = 10_000
    start_time = time.perf_counter()

    for i in range(iterations):
        telemetry.timestamp += 0.1
        tracker.process_telemetry(telemetry, dt=1.0)

    elapsed_time = time.perf_counter() - start_time
    ops_per_sec = iterations / elapsed_time

    # Ensure system completes at least 2,000 full filter+prediction steps per second
    assert ops_per_sec > 2000.0
