from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import circmean

from offline_src.config import AMP_PERCENTILE, PHASE_WINDOW, REFRACTORY_SEC, TARGET_PHASE


def circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def in_target_window(phase: float, target: float, window: float) -> bool:
    diff = circular_distance(np.array([phase]), target)[0]
    return abs(diff) <= window


def simulate_phase_locked_triggers(
    phase: np.ndarray,
    amplitude: np.ndarray,
    sfreq: float,
    target_phase: float = TARGET_PHASE,
    phase_window: float = PHASE_WINDOW,
    amp_percentile: int = AMP_PERCENTILE,
    refractory_sec: float = REFRACTORY_SEC,
) -> Dict[str, object]:
    """
    Simulate phase-locked triggers using zero-crossing detection.
    Trigger when phase crosses target_phase from negative to positive,
    amplitude exceeds threshold, and refractory period has passed.
    """
    amp_threshold = np.percentile(amplitude, amp_percentile)
    refractory_samples = int(refractory_sec * sfreq)

    trigger_indices: List[int] = []
    trigger_phases: List[float] = []
    trigger_amplitudes: List[float] = []

    last_trigger = -refractory_samples
    
    # Shift target phase to [-π, π] range for comparison
    target_shifted = target_phase
    
    for idx in range(1, len(phase)):
        # Check if enough time has passed since last trigger
        if idx - last_trigger < refractory_samples:
            continue
        
        # Get current and previous phase, normalized to [-π, π]
        prev_phase = phase[idx - 1]
        curr_phase = phase[idx]
        
        # Detect zero-crossing: transition from negative to positive around target
        # Create a window around target: [target - π/4, target + π/4]
        window_low = target_shifted - phase_window
        window_high = target_shifted + phase_window
        
        # Check if phase crosses zero within or near target window
        # For simplicity, check if phase enters the window and amplitude is high
        phase_diff = circular_distance(np.array([curr_phase]), target_shifted)[0]
        prev_phase_diff = circular_distance(np.array([prev_phase]), target_shifted)[0]
        
        # Trigger if:
        # 1. Phase enters target window (was outside, now inside)
        # 2. Amplitude exceeds threshold
        # 3. Refractory period has passed
        # 4. Phase is crossing from negative to positive (approaching target)
        
        was_outside = abs(prev_phase_diff) > phase_window
        is_inside = abs(phase_diff) <= phase_window
        is_crossing_positive = (prev_phase_diff < 0) and (phase_diff >= 0)
        
        if is_crossing_positive and amplitude[idx] >= amp_threshold:
            trigger_indices.append(idx)
            trigger_phases.append(curr_phase)
            trigger_amplitudes.append(amplitude[idx])
            last_trigger = idx

    trigger_phases = np.array(trigger_phases)
    phase_errors = circular_distance(trigger_phases, target_phase)
    
    if len(trigger_phases) > 0:
        phase_targeting_accuracy = float(np.mean(np.abs(phase_errors) <= phase_window))
        mean_phase_error = float(circmean(phase_errors, high=np.pi, low=-np.pi))
        resultant_length = float(np.abs(np.mean(np.exp(1j * phase_errors))))
    else:
        phase_targeting_accuracy = 0.0
        mean_phase_error = 0.0
        resultant_length = 0.0

    return {
        "trigger_indices": np.array(trigger_indices, dtype=int),
        "trigger_phases": trigger_phases,
        "trigger_amplitudes": np.array(trigger_amplitudes),
        "phase_errors": phase_errors,
        "phase_targeting_accuracy": phase_targeting_accuracy,
        "mean_phase_error": mean_phase_error,
        "resultant_length": resultant_length,
        "n_triggers": len(trigger_indices),
    }
