# =====================================================================
# Hand-written deployment wrapper.  Everything above this line is generated
# verbatim from the frozen sources; only this section is project glue.
# =====================================================================

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns
from uuid import uuid4

import numpy as np


PLUGIN_VERSION = "phase-tracking-cecht-plugin-v1"

# Frozen canonical deployment configuration (approved 8 s / f0 = 0.8 Hz).
DEPLOY_FS_HZ = 250.0
DEPLOY_WINDOW_S = 8.0
DEPLOY_WINDOW_SAMPLES = 2000
DEPLOY_N_FFT = 2000
DEPLOY_F0_HZ = 0.8
DEPLOY_BAND_HZ = (0.5, 1.25)
DEPLOY_FILTER_ORDER = 2
DEPLOY_TARGET_PHASE_RAD = 0.0
DEPLOY_REFRACTORY_S = 0.5
MAX_BLOCK_SAMPLES = 1024
PHASE_RING_SAMPLES = 2500  # 10 s of history at 250 Hz

SELF_TEST_REL_TOL = 1e-9
SELF_TEST_ABS_TOL = 1e-12

_LOG_LOCK = Lock()
_LOG_ENV = "CLAS_CECHT_PHASE_LOG"
_DEFAULT_LOG = Path(__file__).resolve().parent / "probe_logs" / "PhaseTracking_cecHT.jsonl"
_SELF_TEST_CACHE: dict | None = None


def deploy_core_config():
    """The frozen estimator configuration used by every channel instance."""

    return CecHTCoreConfig(
        sampling_rate_hz=DEPLOY_FS_HZ,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=DEPLOY_WINDOW_SAMPLES,
        l_freq=DEPLOY_BAND_HZ[0],
        h_freq=DEPLOY_BAND_HZ[1],
        filter_order=DEPLOY_FILTER_ORDER,
        filter_type="butter",
        n_fft=DEPLOY_N_FFT,
        fft_mode="exact",
        calibrate=True,
        f0=DEPLOY_F0_HZ,
        parameter_version="deploy-canonical-8s-v1",
        config_version="deploy-canonical-8s-v1",
        max_block_samples=MAX_BLOCK_SAMPLES,
    )


def deploy_trigger_config():
    """Pseudo-trigger configuration; no real stimulation is ever commanded."""

    return PseudoTriggerConfig(
        sampling_rate_hz=DEPLOY_FS_HZ,
        target_phase_rad=DEPLOY_TARGET_PHASE_RAD,
        refractory_s=DEPLOY_REFRACTORY_S,
        parameter_version="deploy-canonical-8s-v1",
        config_version="deploy-canonical-8s-v1",
        max_block_samples=MAX_BLOCK_SAMPLES,
    )


def kernel_statistics(kernel: np.ndarray) -> dict:
    """Environment-sensitive numbers baked into GOLDEN_KERNEL by the builder."""

    values = np.asarray(kernel, dtype=np.complex128)
    return {
        "kernel_len": int(values.size),
        "max_abs": float(np.max(np.abs(values))),
        "l1": float(np.sum(np.abs(values))),
        "at": {
            str(index): {
                "re": float(values[index].real),
                "im": float(values[index].imag),
            }
            for index in (0, 1, values.size // 2, values.size - 1)
        },
    }


def run_kernel_self_test() -> dict:
    """Recompute the endpoint kernel in-host and compare against the golden.

    This is the deployment-time environment check: ``GOLDEN_KERNEL`` was
    produced on the development machine, so a mismatch means the host's
    numpy/scipy produced a numerically different kernel.
    """

    global _SELF_TEST_CACHE
    if _SELF_TEST_CACHE is not None:
        return _SELF_TEST_CACHE

    try:
        kernel = CausalCecHTCore(deploy_core_config()).endpoint_kernel
        observed = kernel_statistics(kernel)
        worst = 0.0
        worst_path = None
        failures = []

        def compare(reference, candidate, path):
            nonlocal worst, worst_path
            if isinstance(reference, dict):
                for key, value in reference.items():
                    compare(value, candidate.get(key), f"{path}.{key}")
                return
            difference = abs(float(candidate) - float(reference))
            relative = difference / abs(float(reference)) if reference else difference
            if difference > SELF_TEST_ABS_TOL and relative > SELF_TEST_REL_TOL:
                failures.append(path)
                if relative > worst:
                    worst, worst_path = relative, path

        compare(GOLDEN_KERNEL, observed, "kernel")
        _SELF_TEST_CACHE = {
            "status": "PASS" if not failures else "FAIL",
            "failed_paths": failures,
            "worst_relative_deviation": worst,
            "worst_path": worst_path,
            "observed": observed,
            "expected": GOLDEN_KERNEL,
            "rel_tol": SELF_TEST_REL_TOL,
            "abs_tol": SELF_TEST_ABS_TOL,
        }
    except BaseException as exc:  # noqa: BLE001 - diagnostics must never raise
        _SELF_TEST_CACHE = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return _SELF_TEST_CACHE


class CechtPhaseCore:
    """Causal cecHT phase estimator with a host-shaped block interface.

    Responsibilities:

    * stage each incoming block into a pre-allocated C-contiguous float64
      buffer (the host hands over strided views; the streaming core rejects
      non-contiguous input);
    * keep one absolute sample counter per instance so the core never sees a
      gap or a duplicate index;
    * run the pseudo-trigger on ``valid & ready`` phase endpoints;
    * hand the phase history to the companion side channel;
    * return the input waveform unchanged (the phase is not a waveform).
    """

    def __init__(self, fs: float) -> None:
        self.constructor_fs = fs
        self._instance_id = uuid4().hex
        configured_log = os.environ.get(_LOG_ENV)
        self._log_path = (
            Path(configured_log).expanduser().resolve() if configured_log else _DEFAULT_LOG
        )
        self._log_errors = 0
        self._initialized = False
        self._degraded = False
        self._degrade_reason = None
        self._core = None
        self._trigger = None
        self._self_test = None

        # Pre-allocated hot-path storage (no allocation inside ``process``).
        self._block = np.empty(MAX_BLOCK_SAMPLES, dtype=np.float64)
        self._relative_index = np.empty(MAX_BLOCK_SAMPLES, dtype=np.int64)
        self._offset = np.arange(MAX_BLOCK_SAMPLES, dtype=np.int64)
        self._core_outputs = CecHTOutputBuffers.allocate(MAX_BLOCK_SAMPLES)
        self._trigger_outputs = PseudoTriggerOutputBuffers.allocate(MAX_BLOCK_SAMPLES)
        self._phase_ring = np.zeros(PHASE_RING_SAMPLES, dtype=np.float64)
        self._index_ring = np.zeros(PHASE_RING_SAMPLES, dtype=np.int64)
        self._ready_ring = np.zeros(PHASE_RING_SAMPLES, dtype=np.bool_)

        self._zero_counters()
        if fs == DEPLOY_FS_HZ:
            try:
                self._core = CausalCecHTCore(deploy_core_config())
                self._trigger = CausalCecHTTrigger(deploy_trigger_config())
                self._self_test = run_kernel_self_test()
            except BaseException as exc:  # noqa: BLE001 - degrade instead of crashing
                self._degrade(f"setup_failed: {type(exc).__name__}: {exc}")
        else:
            self._degrade(f"unsupported_sampling_rate: {fs}")

        self._initialized = True
        self._write_event(
            "loaded",
            {
                "plugin_version": PLUGIN_VERSION,
                "constructor_fs_declared_by_host": fs,
                "module_name": __name__,
                "module_present_in_sys_modules": bool(
                    __name__ in sys.modules and sys.modules.get(__name__) is not None
                ),
                "module_path": str(Path(__file__).resolve()),
                "numpy_version": np.__version__,
                "scipy_version": _scipy_version(),
                "degraded": self._degraded,
                "degrade_reason": self._degrade_reason,
                "self_test": self._self_test,
                "config": {
                    "sampling_rate_hz": DEPLOY_FS_HZ,
                    "window_s": DEPLOY_WINDOW_S,
                    "window_samples": DEPLOY_WINDOW_SAMPLES,
                    "n_fft": DEPLOY_N_FFT,
                    "f0_hz": DEPLOY_F0_HZ,
                    "band_hz": list(DEPLOY_BAND_HZ),
                    "filter_order": DEPLOY_FILTER_ORDER,
                    "calibrated": True,
                    "target_phase_rad": DEPLOY_TARGET_PHASE_RAD,
                    "refractory_s": DEPLOY_REFRACTORY_S,
                },
                "stimulation_enabled": False,
                "process_file_io": False,
            },
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        if self._initialized and self.attempts:
            self._write_event("reset_summary", self.snapshot())
        if self._core is not None:
            self._core.reset(CecHTReasonCode.EXPLICIT_RESET)
        if self._trigger is not None:
            self._trigger.reset(PseudoTriggerReasonCode.EXPLICIT_RESET)
        self._zero_counters()

    def _zero_counters(self) -> None:
        self.attempts = 0
        self.calls = 0
        self.samples = 0
        self.ready_samples = 0
        self.events = 0
        self.internal_errors = 0
        self.structural_rejects = 0
        self.stream_faults = 0
        self.oversize_blocks = 0
        self.max_process_ns = 0
        self.total_process_ns = 0
        self.epoch = 0
        self._sample_index = 0
        self._ring_pos = 0
        self._ring_filled = 0
        self._last_ready_phase = None
        self._last_ready_index = None
        self._last_event_index = None
        self._last_event_crossing = None
        self._first_fault_logged = False
        self._first_reject_logged = False

    def _degrade(self, reason: str) -> None:
        self._degraded = True
        self._degrade_reason = reason

    # -- hot path ----------------------------------------------------------

    def process(self, x, stim_label):
        started = perf_counter_ns()
        self.attempts += 1
        try:
            if self._degraded or self._core is None:
                return x
            try:
                block = x if isinstance(x, np.ndarray) else np.asarray(x)
            except BaseException:  # noqa: BLE001
                self.internal_errors += 1
                return x
            if block.ndim != 1:
                self.structural_rejects += 1
                self._degrade(f"unexpected_block_ndim: {block.ndim}")
                return x
            count = int(block.size)
            if count == 0:
                return x
            if count > MAX_BLOCK_SAMPLES:
                self.oversize_blocks += 1
                self._degrade(f"block_larger_than_capacity: {count}")
                return x

            # The host hands over a strided view; the core requires C-contiguous
            # float64 input, so stage it into our own pre-allocated buffer.
            self._block[:count] = block
            staged = self._block[:count]

            np.add(self._offset[:count], self._sample_index, out=self._relative_index[:count])
            context = CecHTBlockContext(
                stream_epoch=self.epoch,
                first_relative_index=self._sample_index,
                declared_sampling_rate_hz=DEPLOY_FS_HZ,
                unit_code=1,
                channel_code=7,
                polarity_code=1,
                verified_fields_mask=CecHTVerifiedField.ALL,
            )
            status = self._core.process_block_into(staged, context, self._core_outputs)
            self.calls += 1
            self.samples += count

            if status is CecHTProcessStatus.STRUCTURAL_REJECT:
                self.structural_rejects += 1
                if not self._first_reject_logged:
                    self._first_reject_logged = True
                    self._write_event(
                        "structural_reject",
                        {
                            "reason": str(self._core.last_reason_code),
                            "block_size": count,
                            "sample_index": self._sample_index,
                        },
                    )
                self._degrade(f"structural_reject: {self._core.last_reason_code}")
                self._sample_index += count
                return x
            if status is CecHTProcessStatus.STREAM_FAULT:
                self.stream_faults += 1
                if not self._first_fault_logged:
                    self._first_fault_logged = True
                    self._write_event(
                        "stream_fault",
                        {
                            "reason": str(self._core.last_reason_code),
                            "block_size": count,
                            "sample_index": self._sample_index,
                        },
                    )
                # The core has already reset itself; keep running so a single
                # transient glitch cannot disable the algorithm for the session.
                self._sample_index += count
                return x

            self._run_trigger(count)
            self._push_history(count)
            self._sample_index += count
        except BaseException as exc:  # noqa: BLE001 - never break the host callback
            self.internal_errors += 1
            if self.internal_errors == 1:
                self._write_event(
                    "internal_error",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
        finally:
            elapsed = perf_counter_ns() - started
            self.total_process_ns += elapsed
            self.max_process_ns = max(self.max_process_ns, elapsed)
        return x

    def _run_trigger(self, count: int) -> None:
        if self._trigger is None:
            return
        status = self._trigger.process_block_into(
            self._relative_index[:count],
            self._core_outputs.phase_rad[:count],
            self._core_outputs.valid[:count],
            self._core_outputs.ready[:count],
            self._trigger_outputs,
        )
        if status is PseudoTriggerStatus.OK:
            fired = int(np.count_nonzero(self._trigger_outputs.event[:count]))
            if fired:
                self.events += fired
                positions = np.flatnonzero(self._trigger_outputs.event[:count])
                last = int(positions[-1])
                self._last_event_index = int(self._relative_index[last])
                self._last_event_crossing = float(
                    self._trigger_outputs.crossing_time_s[last]
                )

    def _push_history(self, count: int) -> None:
        ready = self._core_outputs.ready[:count]
        if not ready.any():
            return
        positions = np.flatnonzero(ready)
        self.ready_samples += int(positions.size)
        last = int(positions[-1])
        self._last_ready_phase = float(self._core_outputs.phase_rad[last])
        self._last_ready_index = int(self._relative_index[last])

        capacity = PHASE_RING_SAMPLES
        position = self._ring_pos
        end = position + count
        if end <= capacity:
            self._phase_ring[position:end] = self._core_outputs.phase_rad[:count]
            self._index_ring[position:end] = self._relative_index[:count]
            self._ready_ring[position:end] = ready
        else:
            head = capacity - position
            self._phase_ring[position:] = self._core_outputs.phase_rad[:head]
            self._index_ring[position:] = self._relative_index[:head]
            self._ready_ring[position:] = ready[:head]
            tail = count - head
            self._phase_ring[:tail] = self._core_outputs.phase_rad[head:count]
            self._index_ring[:tail] = self._relative_index[head:count]
            self._ready_ring[:tail] = ready[head:count]
        self._ring_pos = end % capacity
        self._ring_filled = min(capacity, self._ring_filled + count)

    # -- reporting ---------------------------------------------------------

    def latest_phase(self) -> dict:
        """Read-only view of the newest ready phase endpoint."""

        return {
            "ready": self._last_ready_phase is not None,
            "phase_rad": self._last_ready_phase,
            "relative_sample_index": self._last_ready_index,
            "ready_samples": self.ready_samples,
        }

    def phase_history(self) -> dict:
        """Snapshot of the phase ring for a companion app."""

        count = self._ring_filled
        if count == 0:
            return {
                "phase_rad": np.empty(0, dtype=np.float64),
                "relative_sample_index": np.empty(0, dtype=np.int64),
                "ready": np.empty(0, dtype=np.bool_),
            }
        capacity = PHASE_RING_SAMPLES
        if count < capacity:
            return {
                "phase_rad": self._phase_ring[:count].copy(),
                "relative_sample_index": self._index_ring[:count].copy(),
                "ready": self._ready_ring[:count].copy(),
            }
        order = np.concatenate(
            (
                np.arange(self._ring_pos, capacity),
                np.arange(0, self._ring_pos),
            )
        )
        return {
            "phase_rad": self._phase_ring[order],
            "relative_sample_index": self._index_ring[order],
            "ready": self._ready_ring[order],
        }

    def snapshot(self) -> dict:
        mean_ns = self.total_process_ns / self.attempts if self.attempts else None
        return {
            "plugin_version": PLUGIN_VERSION,
            "instance_id": self._instance_id,
            "constructor_fs_declared_by_host": self.constructor_fs,
            "degraded": self._degraded,
            "degrade_reason": self._degrade_reason,
            "self_test_status": (self._self_test or {}).get("status"),
            "attempts": self.attempts,
            "calls": self.calls,
            "samples": self.samples,
            "ready_samples": self.ready_samples,
            "events": self.events,
            "internal_errors": self.internal_errors,
            "structural_rejects": self.structural_rejects,
            "stream_faults": self.stream_faults,
            "oversize_blocks": self.oversize_blocks,
            "last_reason_code": (
                str(self._core.last_reason_code) if self._core is not None else None
            ),
            "last_trigger_reason_code": (
                str(self._trigger.last_reason_code) if self._trigger is not None else None
            ),
            "last_ready_phase_rad": self._last_ready_phase,
            "last_ready_index": self._last_ready_index,
            "last_event_index": self._last_event_index,
            "last_event_crossing_s": self._last_event_crossing,
            "mean_process_ns": mean_ns,
            "max_process_ns": self.max_process_ns if self.attempts else None,
            "raw_samples_retained": False,
            "returned_object_identity_preserved": True,
            "stimulation_enabled": False,
        }

    def _write_event(self, event: str, payload: dict) -> None:
        record = {
            "schema_version": 1,
            "event": event,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "instance_id": self._instance_id,
            "pid": os.getpid(),
            "thread_id": get_ident(),
            "payload": payload,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with _LOG_LOCK, self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except BaseException:  # noqa: BLE001 - logging must not break the host
            self._log_errors += 1


def _scipy_version() -> str:
    try:
        import scipy

        return str(scipy.__version__)
    except BaseException:  # noqa: BLE001
        return "unavailable"


# NOTE: this class deliberately has no docstring.  The host builds each
# algorithm-list entry as '<file stem> (<first line of the class docstring>)',
# so a docstring here would be appended to the name shown in the GUI.
class CustomSignalAlgorithm(CechtPhaseCore):
    pass


__all__ = [
    "CustomSignalAlgorithm",
    "CechtPhaseCore",
    "deploy_core_config",
    "deploy_trigger_config",
    "kernel_statistics",
    "run_kernel_self_test",
    "PLUGIN_VERSION",
]
