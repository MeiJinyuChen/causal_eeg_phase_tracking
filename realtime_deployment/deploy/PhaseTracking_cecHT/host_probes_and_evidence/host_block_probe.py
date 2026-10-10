"""Same-shape host block-interface probe for target-host validation.

This module is deliberately non-actuating: it applies no filter, classifier,
gate, phase tracker, or stimulus command.  It records aggregate interface facts
without retaining EEG samples and always returns the original block object.

The canonical source can be copied as one standalone file into the host's
``user_algorithms`` directory.  It depends only on the Python standard library
and NumPy, both of which are present in the inspected host bundle.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import numpy as np


_LOG_LOCK = Lock()
_LOG_ENV = "CLAS_HOST_PROBE_LOG"
_DEFAULT_LOG = Path(__file__).resolve().parent / "probe_logs" / "interface_probe.jsonl"
_PROCESS_SAMPLE_CALL = 1000


class PassThroughContractProbe:
    """Observe block shape/type/timing without retaining EEG samples."""

    def __init__(self, fs: float) -> None:
        self.constructor_fs = fs
        self.reset()

    def reset(self) -> None:
        self.attempts = 0
        self.calls = 0
        self.samples = 0
        self.internal_errors = 0
        self.nonfinite_calls = 0
        self.first_process_observed = False
        self.process_sample_trigger_call = _PROCESS_SAMPLE_CALL
        self.process_sample_written = False
        self.value_min = None
        self.value_max = None
        self.max_process_ns = 0
        self.total_process_ns = 0
        self.input_ndims: Counter[int] = Counter()
        self.input_dtypes: Counter[str] = Counter()
        self.input_lengths: Counter[int] = Counter()
        self.input_shapes: Counter[str] = Counter()
        self.input_writeable: Counter[bool] = Counter()
        self.input_c_contiguous: Counter[bool] = Counter()
        self.label_types: Counter[str] = Counter()
        self.label_shapes: Counter[str] = Counter()
        self.process_thread_ids: Counter[int] = Counter()
        self.callback_interval_count = 0
        self.callback_interval_sum_ns = 0
        self.callback_interval_min_ns: int | None = None
        self.callback_interval_max_ns: int | None = None
        self._last_call_started_ns: int | None = None

    def process(self, x: Any, stim_label: Any) -> Any:
        """Observe aggregate facts and return ``x`` by object identity."""

        started = perf_counter_ns()
        self.attempts += 1
        try:
            signal = np.asarray(x)
            label = np.asarray(stim_label)
            self.calls += 1
            self.samples += int(signal.size)
            self.first_process_observed = True
            if self.calls == 1:
                self._on_first_process_observed()
            self.input_ndims[int(signal.ndim)] += 1
            self.input_dtypes[str(signal.dtype)] += 1
            self.input_lengths[int(signal.size)] += 1
            self.input_shapes[str(tuple(signal.shape))] += 1
            self.input_writeable[bool(signal.flags.writeable)] += 1
            self.input_c_contiguous[bool(signal.flags.c_contiguous)] += 1
            self.label_types[type(stim_label).__name__] += 1
            self.label_shapes[str(tuple(label.shape))] += 1
            self.process_thread_ids[int(get_ident())] += 1
            if self._last_call_started_ns is not None:
                interval = started - self._last_call_started_ns
                self.callback_interval_count += 1
                self.callback_interval_sum_ns += interval
                self.callback_interval_min_ns = (
                    interval
                    if self.callback_interval_min_ns is None
                    else min(self.callback_interval_min_ns, interval)
                )
                self.callback_interval_max_ns = (
                    interval
                    if self.callback_interval_max_ns is None
                    else max(self.callback_interval_max_ns, interval)
                )
            self._last_call_started_ns = started
            try:
                if not np.isfinite(signal).all():
                    self.nonfinite_calls += 1
            except (TypeError, ValueError):
                self.nonfinite_calls += 1
            try:
                block_min = np.min(signal)
                block_max = np.max(signal)
                self.value_min = (
                    block_min if self.value_min is None else min(self.value_min, block_min)
                )
                self.value_max = (
                    block_max if self.value_max is None else max(self.value_max, block_max)
                )
            except (TypeError, ValueError):
                pass
        except Exception:
            # Malformed observations must not terminate the host callback.
            self.internal_errors += 1
        finally:
            elapsed = perf_counter_ns() - started
            self.total_process_ns += elapsed
            self.max_process_ns = max(self.max_process_ns, elapsed)
            if (
                self.calls == self.process_sample_trigger_call
                and not self.process_sample_written
            ):
                self.process_sample_written = True
                self._on_process_sample_due()
        return x

    def _on_first_process_observed(self) -> None:
        """Optional sidecar hook; this base class performs no hot-path I/O."""

    def _on_process_sample_due(self) -> None:
        """Optional bounded-snapshot hook; this base class performs no I/O."""

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate observations; no raw samples are included."""

        mean_ns = self.total_process_ns / self.attempts if self.attempts else None
        mean_callback_interval_ns = (
            self.callback_interval_sum_ns / self.callback_interval_count
            if self.callback_interval_count
            else None
        )
        return {
            "probe_status": "passthrough_observation_only",
            "constructor_fs_declared_by_host": self.constructor_fs,
            "constructor_fs_is_verified_sample_rate": False,
            "channel_identity_observable": False,
            "numeric_unit_observable": False,
            "device_timestamps_observable": False,
            "sample_indices_observable": False,
            "stim_label_semantics_observable": False,
            "attempts": self.attempts,
            "calls": self.calls,
            "first_process_observed": getattr(self, "first_process_observed", False),
            "samples": self.samples,
            "internal_errors": self.internal_errors,
            "nonfinite_calls": self.nonfinite_calls,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "input_ndims": dict(self.input_ndims),
            "input_dtypes": dict(self.input_dtypes),
            "input_lengths": dict(self.input_lengths),
            "input_shapes": dict(self.input_shapes),
            "input_writeable": dict(self.input_writeable),
            "input_c_contiguous": dict(self.input_c_contiguous),
            "label_types": dict(self.label_types),
            "label_shapes": dict(self.label_shapes),
            "process_thread_ids": dict(self.process_thread_ids),
            "callback_interval_mean_ns": mean_callback_interval_ns,
            "callback_interval_min_ns": self.callback_interval_min_ns,
            "callback_interval_max_ns": self.callback_interval_max_ns,
            "mean_process_ns": mean_ns,
            "max_process_ns": self.max_process_ns if self.attempts else None,
            "raw_samples_retained": False,
            "stimulation_enabled": False,
        }


class CustomSignalAlgorithm(PassThroughContractProbe):
    """CLAS host 接口探针（仅回放/调试，原样透传）。

    该算法不滤波、不分类、不开放门控，也不产生刺激指令。它只统计宿主实际
    传入的采样率声明、块形状、dtype、调用线程和调用耗时，并始终原样返回
    输入对象。首次 ``process()`` 调用只设置内存标志；切换算法、重载算法或
    改变采样率触发 ``reset()`` 时写汇总到 ``probe_logs/interface_probe.jsonl``。
    首次 ``process()`` 调用会同步写一条 ``first_process_observed`` 证据；此后
    高频 ``process()`` 内不进行文件 I/O，直到第 1000 次调用时写一条有界
    ``process_sample`` 快照。
    """

    def __init__(self, fs: float) -> None:
        self._instance_id = uuid4().hex
        configured_log = os.environ.get(_LOG_ENV)
        self._log_path = (
            Path(configured_log).expanduser().resolve()
            if configured_log
            else _DEFAULT_LOG
        )
        self._initialized = False
        self._log_errors = 0
        super().__init__(fs)
        self._initialized = True
        self._write_event(
            "loaded",
            {
                "constructor_fs_declared_by_host": fs,
                "numpy_version": np.__version__,
                "module_path": str(Path(__file__).resolve()),
                "process_file_io": "first_call_and_bounded_sample_only",
                "stimulation_enabled": False,
            },
        )

    def reset(self) -> None:
        """Persist the completed diagnostic segment, then clear all state."""

        if getattr(self, "_initialized", False):
            self._write_event("reset_summary", self.snapshot())
        super().reset()

    def _on_first_process_observed(self) -> None:
        """Write one sidecar record on the first host process() call."""

        self._write_event(
            "first_process_observed",
            {
                "process_attempts": self.attempts,
                "process_calls": self.calls,
            },
        )

    def _on_process_sample_due(self) -> None:
        """Write one bounded aggregate snapshot at the configured call count."""

        self._write_event("process_sample", self.snapshot())

    def _write_event(self, event: str, payload: dict[str, Any]) -> None:
        """Best-effort constructor/reset logging outside the high-rate path."""

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
        except Exception:
            # A diagnostic sidecar must never terminate or alter host signal flow.
            self._log_errors += 1

    def snapshot(self) -> dict[str, Any]:
        """Add plugin-level logging facts to the in-memory observation."""

        result = super().snapshot()
        result.update(
            {
                "instance_id": self._instance_id,
                "sidecar_log_path": str(self._log_path),
                "sidecar_log_errors": self._log_errors,
                "sidecar_write_policy": "constructor_first_process_bounded_sample_and_reset_only",
            }
        )
        return result
