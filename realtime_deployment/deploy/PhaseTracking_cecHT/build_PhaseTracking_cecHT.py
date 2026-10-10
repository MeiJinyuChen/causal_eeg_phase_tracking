"""Build the single-file host plugin ``PhaseTracking_cecHT.py``.

The plugin is generated (never hand-edited) from the frozen sources plus the
hand-written wrapper, so the deployed estimator cannot drift away from the code
that was validated by the P0/P1/P2/P2b/P3 experiments.

Sources, inlined in this order:

1. ``causal_cecHT/vendor/eikeosmers_cecht/phase.py``   (BSD-3, ECHT core)
2. ``causal_cecHT/streaming_core.py``                  (causal streaming core)
3. ``causal_cecHT/pseudo_trigger.py``                  (pseudo-trigger)

Transformations: drop the ``from __future__`` lines (one is emitted at the top
of the generated file) and delete the two relative imports, which are replaced
by the inlined definitions.  Any other change in those two import lines aborts
the build, so a refactor cannot silently produce a wrong plugin.

Usage:

    python build_PhaseTracking_cecHT.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
CECHT_DIR = REPO_ROOT / "causal_phase_tracking" / "causal_cecHT"

SOURCE_PHASE = CECHT_DIR / "vendor" / "eikeosmers_cecht" / "phase.py"
SOURCE_CORE = CECHT_DIR / "streaming_core.py"
SOURCE_TRIGGER = CECHT_DIR / "pseudo_trigger.py"
WRAPPER = HERE / "PhaseTracking_cecHT_wrapper.py"
OUTPUT = HERE / "PhaseTracking_cecHT.py"

FUTURE_LINE = "from __future__ import annotations"
RELATIVE_IMPORTS = (
    "from .vendor.eikeosmers_cecht.phase import ECHT",
    "from .streaming_core import wrap_phase_rad",
)
# Each inlined module must contribute exactly the relative import(s) listed
# here; anything else means the sources were refactored and the build must stop.
EXPECTED_RELATIVE = {
    "phase": (),
    "streaming_core": ("from .vendor.eikeosmers_cecht.phase import ECHT",),
    "pseudo_trigger": ("from .streaming_core import wrap_phase_rad",),
    "wrapper": (),
}

HEADER_DOCSTRING = '''"""CLAS 因果 cecHT 实时相位估计（host 外部算法，仅估计相位，不发刺激）。

Generated single-file plugin.  Do not edit by hand: rebuild with
``realtime_deployment/deploy/PhaseTracking_cecHT/build_PhaseTracking_cecHT.py``.

Frozen deployment configuration: 250 Hz, 8 s window (2000 samples),
``n_fft = N`` with ``fft_mode='exact'``, Butterworth order 2 band 0.5-1.25 Hz,
calibrated at f0 = 0.8 Hz, pseudo-trigger target 0 rad with 0.5 s refractory.

The waveform returned to the host is the input block unchanged; the estimated
phase lives in the plugin's in-memory ring buffer and side-channel log, because
the host's ``process()`` contract only accepts a same-shape waveform.
"""'''

IMPORT_BLOCK = f"""{FUTURE_LINE}

# ---- merged imports (union of the inlined modules plus the wrapper) --------
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum, IntFlag
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len
from scipy.signal import bessel, butter, cheby1, cheby2, ellip, sosfreqz
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    """Return a repo-relative POSIX path so the artifact carries no machine path."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _inline(text: str, label: str) -> str:
    lines = []
    seen_relative = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == FUTURE_LINE:
            continue
        if stripped in RELATIVE_IMPORTS:
            seen_relative.add(stripped)
            lines.append(f"# [build] removed relative import: {stripped}")
            continue
        lines.append(line)
    expected = set(EXPECTED_RELATIVE[label])
    missing = expected - seen_relative
    unexpected = seen_relative - expected
    if missing or unexpected:
        raise SystemExit(
            f"build aborted for {label}: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    return "\n".join(lines).strip("\n")


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def build_golden() -> tuple[dict, dict]:
    """Compute the golden kernel statistics from the frozen package sources."""

    sys.path.insert(0, str(REPO_ROOT))
    from causal_phase_tracking.causal_cecHT.streaming_core import (  # noqa: PLC0415
        CausalCecHTCore,
        CecHTCoreConfig,
    )

    wrapper = _load_module(WRAPPER, "phase_tracking_cecht_wrapper_for_build")
    # The wrapper is normally imported after the inlined modules; give it the
    # names it expects so the golden uses its own key structure verbatim.
    wrapper.np = np
    wrapper.CecHTCoreConfig = CecHTCoreConfig
    wrapper.CausalCecHTCore = CausalCecHTCore
    config = wrapper.deploy_core_config()
    core = CausalCecHTCore(config)
    golden = wrapper.kernel_statistics(core.endpoint_kernel)
    return golden, {
        "window_samples": int(config.window_samples),
        "n_fft": int(config.n_fft),
        "f0_hz": float(config.f0),
        "l_freq": float(config.l_freq),
        "h_freq": float(config.h_freq),
        "filter_order": int(config.filter_order),
        "calibrate": bool(config.calibrate),
    }


def compose(golden: dict, config_summary: dict) -> str:
    phase_body = _inline(SOURCE_PHASE.read_text(encoding="utf-8"), "phase")
    core_body = _inline(SOURCE_CORE.read_text(encoding="utf-8"), "streaming_core")
    trigger_body = _inline(SOURCE_TRIGGER.read_text(encoding="utf-8"), "pseudo_trigger")
    wrapper_body = _inline(WRAPPER.read_text(encoding="utf-8"), "wrapper")

    provenance = "\n".join(
        f"#   {label:<16} {digest}  {_repo_relative(path)}"
        for label, path, digest in (
            ("phase.py", SOURCE_PHASE, _sha256(SOURCE_PHASE)),
            ("streaming_core.py", SOURCE_CORE, _sha256(SOURCE_CORE)),
            ("pseudo_trigger.py", SOURCE_TRIGGER, _sha256(SOURCE_TRIGGER)),
            ("wrapper", WRAPPER, _sha256(WRAPPER)),
        )
    )
    golden_literal = json.dumps(golden, indent=4, sort_keys=True, ensure_ascii=True)
    config_literal = json.dumps(config_summary, sort_keys=True)

    return "\n".join(
        [
            HEADER_DOCSTRING,
            "",
            "# ---------------------------------------------------------------------",
            "# GENERATED FILE - do not edit by hand.",
            "# Source SHA-256 at build time:",
            provenance,
            "# ---------------------------------------------------------------------",
            "",
            IMPORT_BLOCK,
            "",
            "# ---------------------------------------------------------------------",
            "# begin inlined vendor/eikeosmers_cecht/phase.py (BSD-3-Clause)",
            "# ---------------------------------------------------------------------",
            phase_body,
            "",
            "# ---------------------------------------------------------------------",
            "# begin inlined causal_cecHT/streaming_core.py",
            "# ---------------------------------------------------------------------",
            core_body,
            "",
            "# ---------------------------------------------------------------------",
            "# begin inlined causal_cecHT/pseudo_trigger.py",
            "# ---------------------------------------------------------------------",
            trigger_body,
            "",
            "# ---------------------------------------------------------------------",
            "# begin generated golden constants (development-machine reference)",
            f"# config: {config_literal}",
            "# ---------------------------------------------------------------------",
            f"GOLDEN_KERNEL = {golden_literal}",
            "",
            wrapper_body,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild into memory and fail if the on-disk plugin differs",
    )
    args = parser.parse_args()

    golden, config_summary = build_golden()
    generated = compose(golden, config_summary)

    if args.check:
        if not OUTPUT.exists():
            print(f"[fail] {OUTPUT} does not exist")
            return 1
        existing = OUTPUT.read_text(encoding="utf-8")
        if existing != generated:
            print("[fail] on-disk plugin differs from a fresh build")
            return 1
        print("[ok] on-disk plugin matches a fresh build")
        return 0

    OUTPUT.write_text(generated, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"  bytes            : {len(generated.encode('utf-8'))}")
    print(f"  plugin sha256    : {_sha256(OUTPUT)}")
    print(f"  golden kernel l1 : {golden['l1']!r}")
    print(f"  golden max|c|    : {golden['max_abs']!r}")
    print(f"  config           : {config_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
