"""Pre-registered development manifest generation helpers."""

from __future__ import annotations

from typing import Iterable


def choose_development_segments(
    blocks: Iterable[tuple[float, float, str]],
    *,
    segment_s: float = 180.0,
    spacing_s: float = 30.0,
    count: int = 3,
) -> list[tuple[float, float]]:
    """Choose middle 180 s slices from the longest Stage 4 blocks.

    A night is ineligible when it has fewer than ``count`` Stage 4 blocks of at
    least ``segment_s`` seconds.  Selection is deterministic and based only on
    sleep stage, never on cecHT output.
    """

    eligible = [
        (start, end, end - start)
        for start, end, description in blocks
        if description.strip().lower() in {"sleep stage 3", "sleep stage 4"}
        and end - start >= segment_s
    ]
    if len(eligible) < count:
        return []

    eligible.sort(key=lambda item: item[2], reverse=True)
    chosen: list[tuple[float, float]] = []
    for start, end, _ in eligible:
        middle = (start + end) / 2.0
        segment_start = middle - segment_s / 2.0
        segment_end = middle + segment_s / 2.0
        if any(
            segment_start < other_end + spacing_s
            and other_start < segment_end + spacing_s
            for other_start, other_end in chosen
        ):
            continue
        chosen.append((segment_start, segment_end))
        if len(chosen) == count:
            break
    chosen.sort(key=lambda item: item[0])
    return chosen
