from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne

from offline_src.config import EEG_CHANNELS


SUPPORTED_RECORDING_SUFFIXES = {".edf", ".edf+", ".bdf", ".bdf+"}
# Sleep-EDF splits one night into a PSG file and a separate hypnogram file
# (``SC4001E0-PSG.edf`` / ``SC4001EC-Hypnogram.edf``). The night suffix differs
# between the two names, so the shared subject prefix is the pairing key.
HYPNOGRAM_NAME_HINT = "hypnogram"
SUBJECT_PREFIX_LENGTH = 6


@dataclass
class EEGRecording:
    """One loaded EEG recording with metadata."""

    raw: mne.io.BaseRaw
    annotations: mne.Annotations
    subject_id: int
    recording_id: int
    edf_path: Path
    annotation_path: Path | None = None
    source_kind: str = "local"

    @property
    def recording_path(self) -> Path:
        return self.edf_path

    @property
    def psg_path(self) -> Path:
        return self.edf_path

    @property
    def hypnogram_path(self) -> Path | None:
        return self.annotation_path


@dataclass
class EEGRecordingPaths:
    """One EEG recording path pair with simple metadata."""

    subject_id: int
    recording_id: int
    edf_path: Path
    annotation_path: Path | None = None
    source_kind: str = "local"

    @property
    def recording_path(self) -> Path:
        return self.edf_path

    @property
    def psg_path(self) -> Path:
        return self.edf_path

    @property
    def hypnogram_path(self) -> Path | None:
        return self.annotation_path


def _normalise_channel_name(name: str) -> str:
    return (
        name.strip()
        .replace("EEG ", "")
        .replace("eeg ", "")
        .replace(" ", "")
        .lower()
    )


def _match_requested_channels(raw: mne.io.BaseRaw, requested: Iterable[str]) -> list[str]:
    available = {_normalise_channel_name(ch): ch for ch in raw.ch_names}
    selected = []
    for target in requested:
        key = _normalise_channel_name(target)
        if key in available:
            selected.append(available[key])
            continue
        suffix_matches = [
            ch for norm, ch in available.items() if norm.endswith(key) or key.endswith(norm)
        ]
        if suffix_matches:
            selected.append(suffix_matches[0])
    return selected


def _infer_eeg_channels(raw: mne.io.BaseRaw, requested_channels: list[str] | None) -> list[str]:
    if requested_channels:
        selected = _match_requested_channels(raw, requested_channels)
        missing = [ch for ch in requested_channels if ch not in selected]
        if selected:
            if missing:
                print(f"Warning: some requested channels were not found: {missing}")
            return selected

    selected = _match_requested_channels(raw, EEG_CHANNELS)
    if selected:
        return selected

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks):
        return [raw.ch_names[idx] for idx in eeg_picks[: min(2, len(eeg_picks))]]

    non_annotation_channels = [
        ch for ch in raw.ch_names if raw.get_channel_types(picks=[ch])[0] != "stim"
    ]
    if not non_annotation_channels:
        raise ValueError("No usable EEG/data channels were found in the EEG recording.")
    return non_annotation_channels[: min(2, len(non_annotation_channels))]


def is_supported_recording_file(path: str | Path) -> bool:
    """Return True for EDF/EDF+/BDF/BDF+ files.

    EDF+ normally uses the .edf suffix, and BDF+ normally uses .bdf.
    """
    return Path(path).suffix.lower() in SUPPORTED_RECORDING_SUFFIXES


def is_hypnogram_file(path: str | Path) -> bool:
    """Return True for hypnogram companion files such as ``SC4001EC-Hypnogram.edf``."""
    return HYPNOGRAM_NAME_HINT in Path(path).stem.lower()


def find_hypnogram_path(psg_path: str | Path) -> Path | None:
    """Return the hypnogram belonging to a Sleep-EDF style PSG file, if present.

    Pairing uses the shared subject prefix (``SC4001``) instead of the full stem,
    because the night suffix differs between the PSG and hypnogram file names.
    """
    psg_path = Path(psg_path)
    folder = psg_path.parent
    if not folder.is_dir():
        return None
    prefix = psg_path.name[:SUBJECT_PREFIX_LENGTH].lower()
    for candidate in sorted(folder.iterdir()):
        if candidate == psg_path or not candidate.is_file():
            continue
        if not is_supported_recording_file(candidate) or not is_hypnogram_file(candidate):
            continue
        if candidate.name[:SUBJECT_PREFIX_LENGTH].lower() == prefix:
            return candidate
    return None


def load_raw_recording(
    recording_path: str | Path,
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Load a local EDF/EDF+/BDF/BDF+ file as an MNE Raw object."""
    recording_path = Path(recording_path)
    if not recording_path.exists():
        raise FileNotFoundError(f"EEG recording file not found: {recording_path}")

    suffix = recording_path.suffix.lower()
    if suffix in {".edf", ".edf+"}:
        return mne.io.read_raw_edf(recording_path, preload=preload, stim_channel=None, verbose=False)
    if suffix in {".bdf", ".bdf+"}:
        return mne.io.read_raw_bdf(recording_path, preload=preload, stim_channel=None, verbose=False)
    raise ValueError(
        f"Unsupported EEG recording format: {recording_path.suffix}. "
        "Supported formats are .edf/.edf+ and .bdf/.bdf+."
    )


def load_raw_eeg_file(
    recording_path: str | Path,
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Compatibility wrapper for older code."""
    return load_raw_recording(recording_path, preload=preload)


def load_local_recording(
    recording_path: str | Path,
    requested_channels: list[str] | None = None,
    subject_id: int = 0,
    recording_id: int = 1,
    preload: bool = True,
    annotation_path: str | Path | None = None,
) -> EEGRecording:
    """Load one local EDF/EDF+/BDF/BDF+ file and pick requested or inferred EEG channels.

    When a Sleep-EDF style hypnogram sits next to the recording it is attached
    automatically, which is what allows downstream epoch selection to pick
    deep-sleep (N3/N4) segments instead of falling back to the record start.
    """
    recording_path = Path(recording_path)
    if annotation_path is None:
        annotation_path = find_hypnogram_path(recording_path)
    raw = load_raw_recording(recording_path, preload=preload)
    if annotation_path is not None:
        annotation_path = Path(annotation_path)
        raw.set_annotations(mne.read_annotations(annotation_path), emit_warning=False)
    selected = _infer_eeg_channels(raw, requested_channels)
    raw.pick(selected)

    return EEGRecording(
        raw=raw,
        annotations=raw.annotations,
        subject_id=subject_id,
        recording_id=recording_id,
        edf_path=recording_path,
        annotation_path=annotation_path,
        source_kind="local",
    )


def load_local_edf_recording(
    edf_path: str | Path,
    requested_channels: list[str] | None = None,
    subject_id: int = 0,
    recording_id: int = 1,
    preload: bool = True,
) -> EEGRecording:
    """Compatibility wrapper for older local-EDF naming."""
    return load_local_recording(
        edf_path,
        requested_channels=requested_channels,
        subject_id=subject_id,
        recording_id=recording_id,
        preload=preload,
    )


def discover_local_recordings(data_dir: str | Path) -> list[EEGRecordingPaths]:
    """Discover EDF/EDF+/BDF/BDF+ files in a local folder."""
    data_dir = Path(data_dir)
    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and is_supported_recording_file(path) and not is_hypnogram_file(path)
    )
    return [
        EEGRecordingPaths(subject_id=index, recording_id=1, edf_path=path)
        for index, path in enumerate(files, start=1)
    ]
