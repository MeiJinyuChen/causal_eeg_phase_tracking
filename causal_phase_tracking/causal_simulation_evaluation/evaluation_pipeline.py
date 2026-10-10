"""Wire the causal 250 Hz stream through the cecHT core and pseudo-trigger."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_phase_tracking.causal_cecHT.pseudo_trigger import (
    CausalCecHTTrigger,
    PseudoTriggerConfig,
    PseudoTriggerOutputBuffers,
    PseudoTriggerStatus,
)
from causal_phase_tracking.causal_cecHT.streaming_core import (
    CausalCecHTCore,
    CecHTBlockContext,
    CecHTCoreConfig,
    CecHTOutputBuffers,
    CecHTProcessStatus,
    CecHTVerifiedField,
)


@dataclass(slots=True)
class EvaluationPipelineResult:
    phase_rad: np.ndarray
    valid: np.ndarray
    ready: np.ndarray
    reason_code: np.ndarray
    output_index: np.ndarray
    event: np.ndarray
    crossing_time_s: np.ndarray
    decision_time_s: np.ndarray


def run_causal_pipeline(
    stream_values: np.ndarray,
    stream_valid: np.ndarray,
    stream_index: np.ndarray,
    core_config: CecHTCoreConfig,
    trigger_config: PseudoTriggerConfig,
    chunk_sizes: list[int],
) -> EvaluationPipelineResult:
    core = CausalCecHTCore(core_config)
    trigger = CausalCecHTTrigger(trigger_config)

    phase_parts = []
    valid_parts = []
    ready_parts = []
    reason_parts = []
    index_parts = []
    event_parts = []
    crossing_parts = []
    decision_parts = []

    cursor = 0
    pattern_index = 0
    total = stream_values.size
    while cursor < total:
        width = min(chunk_sizes[pattern_index % len(chunk_sizes)], total - cursor)
        block = stream_values[cursor : cursor + width]
        core_outputs = CecHTOutputBuffers.allocate(width)
        context = CecHTBlockContext(
            stream_epoch=1,
            first_relative_index=int(stream_index[cursor]),
            declared_sampling_rate_hz=250.0,
            unit_code=1,
            channel_code=7,
            polarity_code=1,
            verified_fields_mask=CecHTVerifiedField.ALL,
        )
        status = core.process_block_into(block, context, core_outputs)
        if status is CecHTProcessStatus.STRUCTURAL_REJECT:
            raise RuntimeError("unexpected core structural reject in pipeline")
        if status is CecHTProcessStatus.STREAM_FAULT:
            raise RuntimeError("unexpected core stream fault in pipeline")

        trigger_outputs = PseudoTriggerOutputBuffers.allocate(width)
        trigger_status = trigger.process_block_into(
            core_outputs.relative_sample_index,
            core_outputs.phase_rad,
            core_outputs.valid,
            core_outputs.ready,
            trigger_outputs,
        )
        if trigger_status is PseudoTriggerStatus.STRUCTURAL_REJECT:
            raise RuntimeError("unexpected trigger structural reject in pipeline")

        phase_parts.append(core_outputs.phase_rad)
        valid_parts.append(core_outputs.valid)
        ready_parts.append(core_outputs.ready)
        reason_parts.append(core_outputs.reason_code)
        index_parts.append(core_outputs.relative_sample_index)
        event_parts.append(trigger_outputs.event)
        crossing_parts.append(trigger_outputs.crossing_time_s)
        decision_parts.append(trigger_outputs.decision_time_s)
        cursor += width
        pattern_index += 1

    return EvaluationPipelineResult(
        phase_rad=np.concatenate(phase_parts),
        valid=np.concatenate(valid_parts),
        ready=np.concatenate(ready_parts),
        reason_code=np.concatenate(reason_parts),
        output_index=np.concatenate(index_parts),
        event=np.concatenate(event_parts),
        crossing_time_s=np.concatenate(crossing_parts),
        decision_time_s=np.concatenate(decision_parts),
    )
