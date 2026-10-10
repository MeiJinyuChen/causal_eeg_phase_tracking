from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_simulation_evaluation.evaluation_manifest import (
    choose_development_segments,
)


def test_segment_choice_prefers_long_stage4_blocks_and_spaces_them() -> None:
    blocks = [(100.0, 400.0, "Sleep stage 4"), (500.0, 800.0, "Sleep stage 4"), (900.0, 1200.0, "Sleep stage 4")]
    segments = choose_development_segments(blocks, segment_s=180.0, spacing_s=30.0, count=3)
    assert len(segments) == 3
    starts = [item[0] for item in segments]
    ends = [item[1] for item in segments]
    assert all(end - start == 180.0 for start, end in zip(starts, ends))
    assert all(ends[i] + 30.0 <= starts[i + 1] for i in range(len(segments) - 1))


def test_segment_choice_drops_nights_with_fewer_than_three_stage4_blocks() -> None:
    blocks = [(100.0, 350.0, "Sleep stage 4"), (500.0, 700.0, "Sleep stage 2")]
    assert choose_development_segments(blocks, segment_s=180.0, spacing_s=30.0, count=3) == []
