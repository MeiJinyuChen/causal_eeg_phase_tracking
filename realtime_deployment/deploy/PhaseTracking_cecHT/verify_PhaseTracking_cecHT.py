"""Local verification of the generated ``PhaseTracking_cecHT.py`` plugin.

Checks, in order:

1. the plugin imports standalone (registered in ``sys.modules`` exactly like
   the host loader does, under a dotted ``user_algorithms.*`` name);
2. the in-host kernel self-test passes against the baked golden;
3. a non-contiguous block is accepted (regression for the strided-view
   rejection that the host probe measured), and the bare core still rejects
   the same strided block (proving the staging copy is what fixes it);
4. per-sample phase, ``ready`` flags and pseudo-events are identical to the
   frozen package implementation driven with the same input;
5. ``process`` returns the input object unchanged;
6. wrong sampling rate and oversize blocks degrade to pass-through instead of
   raising;
7. per-callback cost is measured against the host's ~41.5 ms callback interval.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from time import perf_counter_ns

import numpy as np


HERE = Path(__file__).resolve().parent
PLUGIN_PATH = HERE / "PhaseTracking_cecHT.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

HOST_MODULE_NAME = "user_algorithms.PhaseTracking_cecHT"
FS = 250.0
BLOCK_PATTERN = (10, 11, 10, 12, 9, 10, 11, 10)
WARMUP_SAMPLES = 2000
TOTAL_SAMPLES = 6000
PARITY_TOL = 1e-12

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def load_plugin():
    """Import the plugin the way the host does: dotted name, registered."""

    sys.modules.setdefault("user_algorithms", types.ModuleType("user_algorithms"))
    specification = importlib.util.spec_from_file_location(HOST_MODULE_NAME, PLUGIN_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[HOST_MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


def make_signal(total: int) -> np.ndarray:
    """Deterministic slow-wave-like signal with a frequency drift."""

    index = np.arange(total, dtype=np.float64)
    time = index / FS
    frequency = 0.8 + 0.05 * np.sin(2.0 * np.pi * 0.01 * time)
    phase = 2.0 * np.pi * np.cumsum(frequency) / FS
    signal = 40.0 * np.sin(phase) + 8.0 * np.sin(2.0 * np.pi * 2.3 * time)
    return signal.astype(np.float64)


def channel_views(signal: np.ndarray, channel: int = 3, channels: int = 8):
    """Yield (offset, width, view) blocks shaped like the host's per-channel view.

    The host hands ``process()`` a ``data[:, ch]`` slice of a C-contiguous
    ``(samples, channels)`` array, which is a genuinely strided (non
    C-contiguous) view.  The host probe measured that in 997 of 1000 calls.
    """

    matrix = np.zeros((signal.size, channels), dtype=np.float64)
    matrix[:, channel] = signal
    offset = 0
    position = 0
    while offset < signal.size:
        width = min(BLOCK_PATTERN[position % len(BLOCK_PATTERN)], signal.size - offset)
        yield offset, width, matrix[offset : offset + width, channel]
        offset += width
        position += 1


def reference_stream(signal: np.ndarray):
    """Drive the frozen package implementation over the same blocks."""

    sys.path.insert(0, str(REPO_ROOT))
    from causal_phase_tracking.causal_cecHT.pseudo_trigger import (  # noqa: PLC0415
        CausalCecHTTrigger,
        PseudoTriggerConfig,
        PseudoTriggerOutputBuffers,
    )
    from causal_phase_tracking.causal_cecHT.streaming_core import (  # noqa: PLC0415
        CausalCecHTCore,
        CecHTBlockContext,
        CecHTOutputBuffers,
        CecHTReasonCode,
        CecHTVerifiedField,
    )

    plugin = load_plugin()
    core = CausalCecHTCore(plugin.deploy_core_config())
    trigger = CausalCecHTTrigger(plugin.deploy_trigger_config())
    outputs = CecHTOutputBuffers.allocate(plugin.MAX_BLOCK_SAMPLES)
    trigger_outputs = PseudoTriggerOutputBuffers.allocate(plugin.MAX_BLOCK_SAMPLES)

    total = signal.size
    phase_est = np.zeros(total, dtype=np.float64)
    ready_flags = np.zeros(total, dtype=np.bool_)
    index_trace = np.arange(total, dtype=np.int64)
    block_start = 0
    events = 0
    position = 0
    while block_start < total:
        width = min(BLOCK_PATTERN[position % len(BLOCK_PATTERN)], total - block_start)
        block = np.ascontiguousarray(signal[block_start : block_start + width])
        context = CecHTBlockContext(
            stream_epoch=0,
            first_relative_index=block_start,
            declared_sampling_rate_hz=FS,
            unit_code=1,
            channel_code=7,
            polarity_code=1,
            verified_fields_mask=CecHTVerifiedField.ALL,
        )
        status = core.process_block_into(block, context, outputs)
        if status.name != "OK":
            raise RuntimeError(f"reference stream fault: {status}")
        trigger.process_block_into(
            index_trace[block_start : block_start + width],
            outputs.phase_rad[:width],
            outputs.valid[:width],
            outputs.ready[:width],
            trigger_outputs,
        )
        events += int(np.count_nonzero(trigger_outputs.event[:width]))
        phase_est[block_start : block_start + width] = outputs.phase_rad[:width]
        ready_flags[block_start : block_start + width] = outputs.ready[:width]
        block_start += width
        position += 1
    return {
        "phase": phase_est,
        "ready": ready_flags,
        "events": events,
        "last_index": int(index_trace[-1]),
        "module": plugin,
    }


def plugin_stream(signal: np.ndarray, plugin):
    """Drive the generated plugin with strided views, as the host does."""

    algorithm = plugin.CustomSignalAlgorithm(FS)
    identity_ok = True
    intervals = []
    label = np.zeros(0, dtype=np.uint8)
    for _offset, _width, block in channel_views(signal):
        if block.flags.c_contiguous:
            raise AssertionError("test harness failed to build a strided view")
        started = perf_counter_ns()
        returned = algorithm.process(block, label)
        intervals.append(perf_counter_ns() - started)
        identity_ok = identity_ok and returned is block
    # NOTE: do not reset here -- reset() clears counters and the phase ring by
    # design, so the caller must read the state first.
    return algorithm, identity_ok, np.asarray(intervals, dtype=np.float64)


def main() -> int:
    print(f"plugin: {PLUGIN_PATH}")
    plugin = load_plugin()
    check(
        "standalone import under a dotted host-style module name",
        plugin.__name__ == HOST_MODULE_NAME,
        f"__name__={plugin.__name__}",
    )
    check(
        "module registered in sys.modules (required for dataclasses)",
        sys.modules.get(HOST_MODULE_NAME) is plugin,
    )

    self_test = plugin.run_kernel_self_test()
    check(
        "in-host kernel self-test against the baked golden",
        self_test.get("status") == "PASS",
        json.dumps(
            {
                "status": self_test.get("status"),
                "worst_relative_deviation": self_test.get("worst_relative_deviation"),
                "failed_paths": self_test.get("failed_paths"),
            },
            ensure_ascii=False,
        ),
    )

    # 3. non-contiguous regression
    from causal_phase_tracking.causal_cecHT.streaming_core import (  # noqa: PLC0415
        CausalCecHTCore,
        CecHTBlockContext,
        CecHTOutputBuffers,
        CecHTVerifiedField,
    )

    signal = make_signal(TOTAL_SAMPLES)
    _offset, _width, strided = next(channel_views(signal))
    check(
        "test harness really produced a strided (non C-contiguous) view",
        not strided.flags.c_contiguous,
        f"strides={strided.strides}",
    )
    bare_core = CausalCecHTCore(plugin.deploy_core_config())
    bare_status = bare_core.process_block_into(
        strided,
        CecHTBlockContext(
            stream_epoch=0,
            first_relative_index=0,
            declared_sampling_rate_hz=FS,
            unit_code=1,
            channel_code=7,
            polarity_code=1,
            verified_fields_mask=CecHTVerifiedField.ALL,
        ),
        CecHTOutputBuffers.allocate(plugin.MAX_BLOCK_SAMPLES),
    )
    check(
        "bare core still rejects a strided block (documents the failure mode)",
        bare_status.name == "STRUCTURAL_REJECT",
        f"status={bare_status.name}",
    )

    reference = reference_stream(signal)
    algorithm, identity_ok, intervals = plugin_stream(signal, plugin)
    snapshot = algorithm.snapshot()
    check("process() returns the input object unchanged", identity_ok)
    check(
        "no structural rejects / stream faults / internal errors in the plugin",
        snapshot["structural_rejects"] == 0
        and snapshot["stream_faults"] == 0
        and snapshot["internal_errors"] == 0,
        json.dumps(
            {
                key: snapshot[key]
                for key in ("structural_rejects", "stream_faults", "internal_errors")
            }
        ),
    )
    check(
        "plugin actually produced ready endpoints",
        snapshot["ready_samples"] > 0,
        f"ready_samples={snapshot['ready_samples']} of {snapshot['samples']} fed",
    )

    history = algorithm.phase_history()
    history_index = history["relative_sample_index"]
    if history_index.size == 0:
        check("phase ring populated", False, "empty history")
    else:
        check(
            "phase ring populated",
            history_index.size > 0,
            f"{history_index.size} samples of history",
        )
        reference_phase = reference["phase"][history_index]
        reference_ready = reference["ready"][history_index]
        max_phase_diff = float(np.max(np.abs(history["phase_rad"] - reference_phase)))
        check(
            "per-sample phase matches the frozen package implementation",
            max_phase_diff <= PARITY_TOL,
            f"max|dphase|={max_phase_diff:.3e} rad over {history_index.size} samples",
        )
        check(
            "ready flags match the frozen package implementation",
            bool(np.array_equal(history["ready"], reference_ready)),
        )
    check(
        "pseudo-event count matches the frozen package implementation",
        snapshot["events"] == reference["events"],
        f"plugin={snapshot['events']} reference={reference['events']}",
    )

    # reset semantics: the host calls reset() when the algorithm is unloaded or
    # the sampling rate changes, and the plugin must clear its streaming state.
    check(
        "plugin reports itself as healthy (not degraded, self-test PASS)",
        (not snapshot["degraded"]) and snapshot["self_test_status"] == "PASS",
        f"self_test={snapshot['self_test_status']} degraded={snapshot['degraded']}",
    )
    log_path = Path(algorithm._log_path)
    algorithm.reset()
    after_reset = algorithm.snapshot()
    check(
        "reset() clears counters and the phase ring",
        after_reset["attempts"] == 0 and algorithm.phase_history()["phase_rad"].size == 0,
        f"attempts={after_reset['attempts']}",
    )
    reset_logged = False
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("event") == "reset_summary":
                    reset_logged = True
                    break
            except json.JSONDecodeError:
                continue
    check(
        "side-channel log contains a reset_summary event",
        reset_logged,
        f"log={log_path}",
    )

    # 6. degradation paths
    wrong_fs = plugin.CustomSignalAlgorithm(500.0)
    returned = wrong_fs.process(strided, np.zeros(10, dtype=np.uint8))
    check(
        "unsupported sampling rate degrades to pass-through",
        wrong_fs.snapshot()["degraded"] and returned is strided,
        f"reason={wrong_fs.snapshot()['degrade_reason']}",
    )
    oversize = plugin.CustomSignalAlgorithm(FS)
    big = np.zeros(plugin.MAX_BLOCK_SAMPLES + 1, dtype=np.float64)
    returned_big = oversize.process(big, np.zeros(big.size, dtype=np.uint8))
    check(
        "oversize block degrades to pass-through instead of raising",
        oversize.snapshot()["degraded"] and returned_big is big,
        f"reason={oversize.snapshot()['degrade_reason']}",
    )

    # 7. cost
    mean_us = float(intervals.mean()) / 1e3
    max_us = float(intervals.max()) / 1e3
    print(
        f"[info] plugin process(): mean={mean_us:.1f} us  max={max_us:.1f} us  "
        f"n={intervals.size} (host callback interval ~41.5 ms)"
    )
    print(
        f"[info] mean cost is {mean_us / 41500.0 * 100.0:.3f}% of the callback budget"
    )

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
