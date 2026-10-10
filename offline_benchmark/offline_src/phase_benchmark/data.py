from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offline_src.config import DATA_DIR, EPOCH_DURATION
from offline_src.data_loader import (
    EEGRecordingPaths,
    discover_local_recordings as _discover_local_recordings,
    is_hypnogram_file,
    load_local_recording,
)
from offline_src.epoching import create_sleep_epochs, has_sleep_stage_annotations


@dataclass
class BenchmarkEpoch:
    subject_id: int
    recording_id: int
    epoch_index: int
    signal: np.ndarray
    sfreq: float
    crop_start_sample: int = 0
    crop_stop_sample: int | None = None
    fieldtrip_buffer_sec: float = 0.0


def discover_local_recordings(data_dir: Path | None = None) -> list[EEGRecordingPaths]:
    """Return local EDF/EDF+/BDF/BDF+ recordings for phase benchmarking."""
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    return _discover_local_recordings(data_dir)


def _choose_epoch_indices(epochs, annotated: bool, epochs_per_subject: int) -> np.ndarray:
    if annotated:
        deep_indices = np.flatnonzero(epochs.events[:, 2] == 4)
        if len(deep_indices):
            return deep_indices[:epochs_per_subject]
    return np.arange(min(epochs_per_subject, len(epochs)), dtype=int)


def load_benchmark_epochs(
    n_subjects: int,
    epochs_per_subject: int,
    fieldtrip_buffer_sec: float = 0.0,
    edf_paths: list[Path] | None = None,
) -> list[BenchmarkEpoch]:
    """Load local EDF/EDF+/BDF/BDF+ epochs for cecHT/classic PLL benchmarking."""
    if edf_paths is None:
        selected_recordings = discover_local_recordings()[:n_subjects]
    else:
        selected_recordings = []
        for path in edf_paths:
            path = Path(path)
            if is_hypnogram_file(path):
                print(f"Skipping hypnogram file (paired automatically with its PSG): {path.name}")
                continue
            selected_recordings.append(
                EEGRecordingPaths(
                    subject_id=len(selected_recordings) + 1,
                    recording_id=1,
                    edf_path=path,
                )
            )
        selected_recordings = selected_recordings[:n_subjects]

    if len(selected_recordings) < n_subjects:
        print(f"Warning: requested {n_subjects} recordings but found {len(selected_recordings)} locally.")

    epochs_out: list[BenchmarkEpoch] = []
    for rec_idx, path_info in enumerate(selected_recordings, start=1):
        print(
            f"[Phase data {rec_idx}/{len(selected_recordings)}] "
            f"file={path_info.edf_path.name}",
            flush=True,
        )
        record = load_local_recording(path_info.edf_path)
        epochs = create_sleep_epochs(record.raw)
        annotated = has_sleep_stage_annotations(record.raw)
        chosen_indices = _choose_epoch_indices(epochs, annotated, epochs_per_subject)
        sfreq = float(record.raw.info["sfreq"])
        epoch_samples = int(round(EPOCH_DURATION * sfreq))
        buffer_samples = int(round(fieldtrip_buffer_sec * sfreq))

        for epoch_index in chosen_indices:
            epoch_start = int(epochs.events[int(epoch_index), 0])
            start = epoch_start - buffer_samples
            stop = epoch_start + epoch_samples + buffer_samples
            if start < 0 or stop > record.raw.n_times:
                continue
            signal = record.raw.get_data(picks=[record.raw.ch_names[0]], start=start, stop=stop)[0]
            epochs_out.append(
                BenchmarkEpoch(
                    subject_id=path_info.subject_id,
                    recording_id=path_info.recording_id,
                    epoch_index=int(epoch_index),
                    signal=signal,
                    sfreq=sfreq,
                    crop_start_sample=buffer_samples,
                    crop_stop_sample=buffer_samples + epoch_samples,
                    fieldtrip_buffer_sec=float(fieldtrip_buffer_sec),
                )
            )
        del record

    if not epochs_out:
        raise RuntimeError("No usable EDF/BDF epochs found for phase benchmark.")
    return epochs_out
