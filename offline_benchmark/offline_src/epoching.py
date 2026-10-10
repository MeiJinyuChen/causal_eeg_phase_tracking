from __future__ import annotations

import mne

from offline_src.config import EPOCH_DURATION


SLEEP_STAGE_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": 6,
    "Movement time": 7,
}


def has_sleep_stage_annotations(raw: mne.io.BaseRaw) -> bool:
    descriptions = set(raw.annotations.description)
    return bool(descriptions.intersection(SLEEP_STAGE_EVENT_ID))


def create_sleep_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Create 30-second epochs from EDF+/BDF+ sleep annotations or fixed windows."""
    if has_sleep_stage_annotations(raw):
        events, event_id = mne.events_from_annotations(
            raw,
            event_id=SLEEP_STAGE_EVENT_ID,
            chunk_duration=EPOCH_DURATION,
            verbose=False,
        )
        present_event_codes = set(events[:, 2])
        # R&K "Sleep stage 3" and "Sleep stage 4" intentionally share one
        # deep-sleep code, but MNE requires the values of the event_id passed to
        # Epochs to be unique. Keep one description per code; downstream
        # selection is by event code, so the descriptions are not used.
        unique_event_id: dict[str, int] = {}
        for description, code in event_id.items():
            if code in present_event_codes and code not in unique_event_id.values():
                unique_event_id[description] = code
        return mne.Epochs(
            raw,
            events,
            event_id=unique_event_id,
            tmin=0,
            tmax=EPOCH_DURATION - 1 / raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False,
        )

    return mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_DURATION,
        preload=True,
        reject_by_annotation=True,
        overlap=0.0,
        verbose=False,
    )
