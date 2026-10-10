from __future__ import annotations

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.streaming_core import (
    CecHTProcessStatus,
    CecHTReasonCode,
)
from causal_phase_tracking.causal_cecHT.pseudo_trigger import (
    PseudoTriggerReasonCode,
    PseudoTriggerStatus,
)


SCOPED_CONTRACT_PATH = (
    ROOT
    / "causal_phase_tracking"
    / "causal_cecHT"
    / "contracts"
    / "causal_cecht_phase_contract.json"
)
GLOBAL_CONTRACT_PATH = (
    ROOT / "realtime_deployment" / "contracts" / "phase_tracking.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scoped_contract_exists_and_is_valid_json() -> None:
    assert SCOPED_CONTRACT_PATH.is_file()
    contract = load_json(SCOPED_CONTRACT_PATH)
    assert contract["contract_id"] == "clas-causal-cecht-phase-v1"
    assert contract["status"] == (
        "development_selection_produced_and_host_plugin_deployed_acceptance_pending"
    )
    assert contract["previous_status"] == "edf replay and offline reference implemented"


def test_input_and_phase_conventions_are_frozen() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    assert contract["input"] == {
        "sampling_rate_hz": 250.0,
        "sample_interval_s": 0.004,
        "signal": "Fpz-Cz",
        "edf_channel": "EEG Fpz-Cz",
        "numeric_unit": "uV",
        "polarity": "Fpz_minus_Cz",
        "sample_dtype": "floating",
    }
    assert contract["phase_convention"] == {
        "storage_unit": "radian",
        "normalization_interval": "[-pi,pi)",
        "wrap_equation": "((angle + pi) % (2*pi)) - pi",
        "positive_pi_maps_to_rad": -math.pi,
        "zero_phase_definition": "positive_peak_of_Fpz_minus_Cz",
        "positive_direction": "increasing_analytic_phase",
        "reference_definition": "angle(x + i*hilbert_transform(x))",
    }


def test_estimator_and_trigger_errors_have_distinct_sampling_contracts() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    estimator = contract["evaluation_errors"]["estimator_error"]
    trigger = contract["evaluation_errors"]["trigger_error"]

    assert estimator == {
        "equation": "wrap(phi_estimate - phi_reference)",
        "positive_error_meaning": "estimate_leads_reference",
        "sampling": "every_valid_ready_endpoint_at_250hz",
    }
    assert trigger["equation"] == "wrap(phi_reference_at_event - theta_target)"
    assert trigger["positive_error_meaning"] == "reference_has_passed_target"
    assert trigger["sampling"] == "pseudo_events_only"
    assert trigger["event_time_views"] == [
        "estimated_crossing_time",
        "decision_time",
    ]


def test_endpoint_and_event_times_cannot_be_mislabeled_as_acoustic_onset() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    time = contract["time_semantics"]
    assert time["estimate_time_position"] == "trailing_window_last_input_sample"
    assert time["estimate_timestamp_source"] == "endpoint_acquisition_timestamp_s"
    assert time["crossing_interpolation"] == "linear_between_adjacent_valid_endpoints"
    assert time["decision_time_position"] == "current_endpoint_callback"
    assert time["acoustic_onset_available"] is False
    assert time["event_times_may_be_labeled_acoustic_onset"] is False
    assert time["latency_compensation_s"] == {
        "estimator_phase_offset": 0.0,
        "scheduler_prediction_horizon": 0.0,
        "acoustic_output_delay": 0.0,
    }


def test_target_and_crossing_policy_match_approved_decisions() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    target = contract["target_phase"]
    assert target["default_rad"] == 0.0
    assert target["runtime_configurable"] is True
    assert target["evaluation_grid_rad"] == [
        0.0,
        math.pi / 4,
        math.pi / 2,
        3 * math.pi / 4,
        -math.pi,
        -3 * math.pi / 4,
        -math.pi / 2,
        -math.pi / 4,
    ]
    assert target["update_boundary"] == "declared_sample_or_block_boundary"
    assert target["update_is_atomic"] is True
    assert target["clear_crossing_comparator_on_update"] is True
    assert target["cancel_pending_event_on_update"] is True

    crossing = contract["pseudo_trigger"]
    assert crossing["implemented"] is True
    assert crossing["crossing_direction"] == "positive_only"
    assert crossing["phase_unwrap_required"] is True
    assert crossing["crossing_interpolation"] == (
        "linear_between_adjacent_valid_endpoints"
    )
    assert crossing["record_reverse_crossing"] is False
    assert crossing["amplitude_eligibility_enabled"] is False
    assert crossing["refractory_s_dev_placeholder"] == 0.5
    assert crossing["refractory_tuning_status"] == "deferred_to_real_stimulation_handover"
    assert crossing["real_stimulation_refractory_selected"] is False
    assert crossing["target_update_clears_comparator_and_cancels_pending"] is True
    assert crossing["real_actuation_enabled"] is False
    assert crossing["output"] == "timestamped_pseudo_event_only"
    assert crossing["output_buffer_dtypes"] == {
        "event": "bool",
        "crossing_time_s": "float64",
        "decision_time_s": "float64",
    }
    assert crossing["latency_compensation_s"] == {
        "estimator_phase_offset": 0.0,
        "scheduler_prediction_horizon": 0.0,
        "acoustic_output_delay": 0.0,
    }


def test_cecht_updates_only_at_every_250hz_sample() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    update = contract["update_grid"]
    assert update == {
        "mode": "every_input_sample",
        "sampling_rate_hz": 250.0,
        "interval_samples": 1,
        "interval_s": 0.004,
        "alternative_or_sensitivity_grids": [],
    }


def test_invalid_estimates_are_null_and_cannot_trigger() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    validity = contract["validity"]
    assert validity["old_value_carry_forward"] is False
    assert validity["invalid_numeric_fields"] == {
        "phase_rad": None,
        "amplitude": None,
        "frequency_hz": None,
    }
    assert validity["invalid_or_not_ready_may_trigger"] is False
    expected_fault_inputs = {
        "segment_change",
        "sample_gap",
        "out_of_order_sample",
        "duplicate_sample",
        "NaN",
        "Inf",
        "sampling_rate_unknown",
        "sampling_rate_mismatch",
        "unit_unknown",
        "unit_mismatch",
        "channel_unknown",
        "channel_mismatch",
        "polarity_unknown",
        "polarity_mismatch",
        "explicit_reset",
    }
    assert set(validity["fault_inputs"]) == expected_fault_inputs
    assert (
        set(global_contract["streaming_contract"]["fault_inputs"])
        == expected_fault_inputs
    )


def test_scoped_contract_is_registered_and_synchronized_globally() -> None:
    scoped = load_json(SCOPED_CONTRACT_PATH)
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    registration = global_contract["scoped_contracts"]["causal_cecht"]

    assert registration["authority"] == "standalone_scoped_machine_contract"
    assert registration["contract_id"] == scoped["contract_id"]
    assert registration["path"] == (
        "../../causal_phase_tracking/causal_cecHT/contracts/"
        "causal_cecht_phase_contract.json"
    )
    assert registration["conflict_policy"] == "fail_and_require_owner_decision"
    assert registration["update_grid_hz"] == 250.0
    assert registration["reverse_crossing_recorded"] is False
    assert registration["amplitude_eligibility_enabled"] is False

    assert registration["method_scoped_sampling_rate_hz"] == scoped["input"][
        "sampling_rate_hz"
    ]
    assert registration["overrides_global_common_analysis_rate"] is True
    assert global_contract["input"]["analysis_sampling_rate_hz"] == 100.0
    assert scoped["global_sync"]["required_method_scoped_equal_fields"] == [
        "input.sampling_rate_hz"
    ]
    assert "input.sampling_rate_hz" not in scoped["global_sync"][
        "required_equal_fields"
    ]
    assert scoped["input"]["signal"] == global_contract["input"]["signal"]
    assert scoped["input"]["numeric_unit"] == global_contract["input"][
        "numeric_unit"
    ]
    assert scoped["input"]["polarity"] == global_contract["input"]["polarity"]
    assert (
        scoped["phase_convention"]["normalization_interval"]
        == global_contract["phase_convention"]["normalization_interval"]
    )
    assert (
        scoped["target_phase"]["default_rad"]
        == global_contract["phase_convention"]["theta_target_default_rad"]
    )
    assert (
        scoped["target_phase"]["evaluation_grid_rad"]
        == global_contract["phase_convention"]["target_grid_rad"]
    )


def test_contract_decisions_are_recorded() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    decisions = contract["contract_decisions"]

    assert len(decisions) == 10
    assert all(set(entry) == {"decision", "choice"} for entry in decisions)
    by_decision = {entry["decision"]: entry["choice"] for entry in decisions}
    assert by_decision["contract authority"] == (
        "a standalone machine contract scoped to the causal cecHT method"
    )
    assert by_decision["default target phase and evaluation grid"] == (
        "0 rad by default, evaluated on the eight point grid around the circle"
    )
    assert by_decision["input unit"] == "microvolt"
    assert by_decision["reverse crossing diagnostic"] == "disabled"
    assert by_decision["may this method contract change the project level contract"] == "yes"
    assert by_decision["automatic commits"] == "not used"
    assert "owner_decisions" not in contract


def test_250hz_rebaseline_is_recorded_as_history() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    rebaseline = contract["sampling_rate_rebaseline"]

    assert rebaseline["approved_date"] == "2026-08-07"
    assert rebaseline["owner_decision"] == "exactly 250 Hz, every sample, 6 s window"
    assert rebaseline["supersedes"] == {
        "earlier update grid decision": "every sample at 100 Hz",
        "earlier sampling rate decision": "exactly 100 Hz with no resampler",
    }
    assert rebaseline["effective_sampling_rate_hz"] == 250.0
    assert rebaseline["sample_interval_s"] == 0.004
    assert rebaseline["window_duration_s"] == 6.0
    assert rebaseline["primary_profile"] == {
        "window_samples": 1500,
        "n_fft": 1500,
        "fft_mode": "exact",
    }
    assert rebaseline["secondary_profile"]["expected_resolved_n_fft"] == 1512
    assert rebaseline["resampler_implemented_in_evaluation_path"] is True
    assert rebaseline["resampler_implemented_in_host_plugin"] is False
    assert rebaseline["non_250_input_policy_core"] == "fail_closed_sampling_rate_mismatch"
    assert rebaseline["non_250_input_policy_host_plugin"] == (
        "degrade_to_same_shape_passthrough_and_record_degrade_reason"
    )


def test_window_rebaseline_supersedes_the_six_second_profile() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)

    assert contract["window_rebaseline"] == {
        "approved_date": "2026-09-15",
        "owner_decision": "exactly 250 Hz, every sample, 8 s window",
        "supersedes": {
            "window_duration_s": 6.0,
            "primary_profile": {"window_samples": 1500, "n_fft": 1500},
            "secondary_profile": {
                "window_samples": 1501,
                "n_fft": None,
                "expected_resolved_n_fft": 1512,
            },
            "f0_hz": 0.875,
        },
        "effective_sampling_rate_hz": 250.0,
        "sample_interval_s": 0.004,
        "window_duration_s": 8.0,
        "primary_profile": {
            "window_samples": 2000,
            "n_fft": 2000,
            "fft_mode": "exact",
        },
        "secondary_profile": {
            "window_samples": 2001,
            "n_fft": None,
            "fft_mode": "fast",
            "expected_resolved_n_fft": 2016,
        },
        "calibration": {"calibrate": True, "f0_hz": 0.8},
        "deployed_profile": "primary_profile",
        "rationale": (
            "The canonical cecHT development baseline and the deployed host plugin "
            "both use an 8 s window (2000 samples at 250 Hz) with n_fft = N and f0 = 0.8 Hz; "
            "the window and f0 were selected on development data."
        ),
        "evidence": [
            "../../../experiment_evidence/cecht/cecHT_HANDOFF.md: canonical baseline 8 s / f0 = 0.8 Hz, "
            "pooled MAE 1.041248 rad",
            "realtime_deployment/deploy/PhaseTracking_cecHT/PhaseTracking_cecHT_wrapper.py: "
            "DEPLOY_WINDOW_S = 8.0, DEPLOY_WINDOW_SAMPLES = 2000, DEPLOY_N_FFT = 2000, "
            "DEPLOY_F0_HZ = 0.8",
        ],
        "regression_assets_still_on_superseded_profile": [
            "causal_simulation_evaluation/goldens/cecht_synthetic_goldens.json",
            "causal_simulation_evaluation/manifests/cecht_synthetic_cases.json",
            "causal_cecHT/tests/test_endpoint_kernel.py",
            "causal_simulation_evaluation/tests/test_cecht_synthetic_math.py",
            "causal_simulation_evaluation/run_smoke_evaluation.py",
        ],
        "migration_status": "pending_regression_refreeze",
        "non_250_input_policy": "fail_closed_sampling_rate_mismatch",
    }

    # The 2026-08-07 record stays untouched as history.
    assert contract["sampling_rate_rebaseline"]["window_duration_s"] == 6.0
    assert (
        contract["sampling_rate_rebaseline"]["primary_profile"]["window_samples"] == 1500
    )


def test_streaming_core_decisions_and_implementation_status_are_recorded() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    decisions = contract["streaming_core_decisions"]

    assert len(decisions) == 9
    by_decision = {entry["decision"]: entry["choice"] for entry in decisions}
    assert by_decision["where the heavy maths runs"].startswith("once in the constructor")
    assert by_decision["hot path interface"] == (
        "caller owned output buffers with a scalar status code"
    )
    assert by_decision["timestamp field name"] == "acquisition_timestamp_s"
    assert by_decision["file scope"] == "core, tests and contracts only"
    assert "step_4_owner_decisions" not in contract

    status = contract["implementation_status"]
    assert status["contract_definition"].startswith("written first")
    assert status["fault_safe_250hz_host_independent_streaming_core"] == "complete"
    # The target-host ingress/reset probe evidence is withheld from this
    # distribution, so no such status key is recorded.
    assert "host_ingress_reset_and_lifecycle_evidence" not in status
    assert status["development_only_parameter_selection"] == (
        "candidate produced, acceptance pending"
    )
    assert status["untouched_test_evaluation"] == "not started"
    assert status["real_stimulation_refractory_selection"] == (
        "deferred to the real-stimulation handover"
    )
    assert "implementation_scope" not in contract


def test_scoped_contract_records_development_selection_and_plugin_release() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    selection = contract["development_selection"]
    plugin = contract["host_plugin"]

    assert selection["status"] == "selection_produced_acceptance_pending"
    assert selection["selected_candidate"] == {
        "calibration": "calibrated",
        "window_s": 8.0,
        "window_samples": 2000,
        "f0_hz": 0.8,
    }
    assert selection["validation_subject_mean_mae_rad"] == 1.0351
    assert selection["validation_bootstrap_ci95_rad"] == [0.9568, 1.1134]
    assert selection["acceptance_threshold_preregistered"] is False
    assert selection["untouched_test_manifest_frozen"] is False

    assert plugin["implemented"] is True
    assert plugin["plugin_file_name"] == "PhaseTracking_cecHT.py"
    assert "deployed_plugin_location" not in plugin
    assert plugin["canonical_source"].endswith("PhaseTracking_cecHT_wrapper.py")
    assert plugin["verify_status"].startswith("passed")
    assert plugin["real_actuation_enabled"] is False
    assert plugin["return_contract"] == "input_block_returned_unchanged_same_shape"
    assert plugin["warmup_s"] == 8.0
    assert contract["streaming_core"]["host_wrapper_implemented"] is True


def test_pseudo_trigger_decisions_are_recorded() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    decisions = contract["pseudo_trigger_decisions"]

    assert len(decisions) == 7
    by_decision = {entry["decision"]: entry["choice"] for entry in decisions}
    assert by_decision["trigger scope"] == (
        "an independent trigger that consumes only valid and ready phase samples"
    )
    assert by_decision["default target phase"] == "0 rad, configurable at runtime"
    assert by_decision["crossing detection"].startswith("positive crossings only")
    assert by_decision["amplitude gate"] == "disabled"
    assert "step_6_owner_decisions" not in contract
    trigger = contract["pseudo_trigger"]
    assert trigger["refractory_tuning_status"] == "deferred_to_real_stimulation_handover"
    assert trigger["real_stimulation_refractory_selected"] is False
    assert "refractory_tuned_in_step" not in trigger
    assert trigger["module"] == "causal_phase_tracking.causal_cecHT.pseudo_trigger"
    assert trigger["api"] == (
        "CausalCecHTTrigger.process_block_into(relative_sample_index, phase_rad, "
        "valid, ready, outputs) -> PseudoTriggerStatus"
    )
    assert {member.name.lower(): int(member) for member in PseudoTriggerStatus} == {
        "ok": 0,
        "empty": 1,
        "structural_reject": 2,
    }
    assert {member.name.lower(): int(member) for member in PseudoTriggerReasonCode} == {
        "none": 0,
        "explicit_reset": 1,
        "target_updated": 2,
        "not_ready": 3,
        "nonfinite_phase": 4,
        "structural_input_type": 10,
        "structural_input_ndim": 11,
        "structural_input_dtype": 12,
        "structural_input_contiguity": 13,
        "structural_input_length_mismatch": 14,
        "structural_output_schema": 15,
        "structural_alias": 16,
        "structural_capacity": 17,
    }


def test_evaluation_decisions_and_smoke_artifacts_are_recorded() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    decisions = contract["evaluation_decisions"]

    assert len(decisions) == 12
    by_decision = {entry["decision"]: entry["choice"] for entry in decisions}
    assert by_decision["adapter engine"] == "a stateful rational polyphase FIR resampler"
    assert by_decision["EDF loading"] == "read the full channel first and slice afterwards"
    assert by_decision["performance scope"] == "workstation only"
    assert "step_7_owner_decisions" not in contract

    implementation = contract["evaluation_implementation"]
    adapter = implementation["rate_adapter"]
    assert adapter["upsample"] == 5
    assert adapter["downsample"] == 2
    assert adapter["group_delay_output_samples"] == 8
    assert adapter["samples_not_shifted_internally"] is True
    assert implementation["offline_reference"]["online_use_forbidden"] is True


def test_hot_buffer_contract_matches_the_implemented_api_and_enums() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    hot = contract["streaming_core"]["hot_path"]

    assert hot["module"] == "causal_phase_tracking.causal_cecHT.streaming_core"
    assert hot["api"] == (
        "CausalCecHTCore.process_block_into(samples, block_context, "
        "output_buffers) -> CecHTProcessStatus"
    )
    assert hot["input"] == {
        "ndim": 1,
        "real_floating": True,
        "supported_dtypes": ["float32", "float64"],
        "dtype": "exact_config_match",
        "c_contiguous": True,
        "input_mutated": False,
        "implicit_conversion": False,
    }
    assert hot["pinned_endpoint_atol_by_input_dtype"] == {
        "float32": 2e-7,
        "float64": 2e-12,
    }
    assert hot["output_buffer_dtypes"] == {
        "phase_rad": "float64",
        "amplitude": "float64",
        "valid": "bool",
        "ready": "bool",
        "reason_code": "uint8",
        "relative_sample_index": "int64",
    }
    assert hot["output_buffer_requirements"] == {
        "ndim": 1,
        "c_contiguous": True,
        "writeable": True,
        "capacity": "at_least_len_samples",
        "input_output_aliasing": False,
        "semantic_output_output_aliasing": False,
    }
    assert hot["process_status_codes"] == {
        member.name.lower(): int(member) for member in CecHTProcessStatus
    }
    assert hot["reason_codes"] == {
        member.name.lower(): int(member) for member in CecHTReasonCode
    }
    assert hot["well_formed_slots"] == "len(samples)"
    assert hot["structural_reject_slots"] == 0
    assert hot["invalid_numeric_sentinel"] == 0.0
    assert hot["consumer_must_obey_valid_ready_masks"] is True


def test_core_contract_freezes_endpoint_fault_and_host_boundaries() -> None:
    contract = load_json(SCOPED_CONTRACT_PATH)
    core = contract["streaming_core"]

    assert core["sampling_rate_hz"] == 250.0
    assert core["resampler_inside_core"] is False
    assert core["owner_model"] == "single_owner_non_reentrant"
    assert core["eight_channels_require_independent_instances"] is True
    assert core["endpoint_engine"] == {
        "construction": "pinned_ECHT_fit_then_endpoint_kernel_derivation",
        "steady_state": "one_or_two_np_dot_calls_into_preallocated_0d_scratch",
        "pinned_transform_or_fft_in_hot_path": False,
        "kernel_read_only": True,
        "n_fft_must_be_at_least_window_samples": True,
    }
    assert core["fault_semantics"] == {
        "epoch_change": "clear_dynamic_state_then_ingest_new_block",
        "stream_fault": "write_len_samples_invalid_slots_clear_state_no_prefix",
        "structural_reject": "write_zero_slots_clear_state",
        "empty": "write_zero_slots_preserve_dynamic_state",
        "old_phase_carry_forward": False,
    }
    assert core["host_wrapper_implemented"] is True
    assert core["target_host_deadline_claimed"] is False
    assert core["thread_safe_or_reentrant_claimed"] is False


def test_hot_relative_context_and_cold_formal_records_are_separate() -> None:
    scoped = load_json(SCOPED_CONTRACT_PATH)
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    metadata = scoped["streaming_core"]["metadata_layering"]

    assert metadata["hot_context_fields"] == [
        "stream_epoch",
        "first_relative_index",
        "declared_sampling_rate_hz",
        "unit_code",
        "channel_code",
        "polarity_code",
        "verified_fields_mask",
    ]
    assert metadata["hot_context_has_device_time_semantics"] is False
    assert metadata["cold_record_fields"] == scoped["output_fields"]
    assert "acquisition_timestamp_s" in metadata["cold_record_fields"]
    assert "acquisition_time_s" not in metadata["cold_record_fields"]
    assert "acquisition_timestamp_s" in global_contract["common_output"][
        "required_fields"
    ]
    assert "acquisition_time_s" not in global_contract["common_output"][
        "required_fields"
    ]


def test_global_registration_reports_core_complete_and_host_wrapper_delivered() -> None:
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    registration = global_contract["scoped_contracts"]["causal_cecht"]

    assert registration["status"] == "development_selection_produced_acceptance_pending"
    assert registration["streaming_core_implemented"] is True
    assert registration["pseudo_trigger_implemented"] is True
    assert registration["rate_adapter_implemented"] is True
    assert registration["edf_replay_implemented"] is True
    assert registration["offline_reference_implemented"] is True
    assert registration["refractory_s_dev_placeholder"] == 0.5
    assert registration["host_wrapper_implemented"] is True
    assert registration["host_plugin_implemented"] is True
    assert registration["development_selection_status"] == "selection_produced_acceptance_pending"
    assert registration["acceptance_threshold_preregistered"] is False
    assert registration["resampler_inside_estimator"] is False
    assert registration["owner_model"] == "single_owner_non_reentrant"
    assert registration["hot_output"] == "caller_owned_fixed_numeric_buffers"
    assert registration["cold_output"] == "formal_14_field_records_when_metadata_exists"
    assert registration["supported_input_dtypes"] == ["float32", "float64"]
    # The host-side probe evidence is withheld from this distribution.


def test_global_plugin_release_records_the_deployed_cecht_plugin() -> None:
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    release = global_contract["host_plugin_release"]
    plugin = release["plugins"][0]

    assert release["deployed_plugin_count"] == 1
    assert plugin["estimator_id"] == "cecHT"
    assert plugin["plugin_file_name"] == "PhaseTracking_cecHT.py"
    assert plugin["gui_label"] == "PhaseTracking_cecHT"
    assert plugin["real_actuation_enabled"] is False
    assert plugin["verify_status"].startswith("passed")
    assert "host_build" not in plugin
    assert "deployed_plugin_location" not in plugin
    assert "deployed_copy_path" not in plugin
    assert (
        plugin["checked_in_build_artifact_sha256"]
        == "4e5e2db668d57342ea3ba60928279a623b6633888eaf2ca3ce86d05d5f51ef58"
    )
    assert plugin["canonical_source_sha256"] == (
        "0cb5b04836a46708aa4792749c9ab55ea5aa2b419c9b09d955c2c3ef6fc0427c"
    )
    assert plugin["rebuild_preserves_the_gui_label"] is True


def test_global_contract_records_rate_mismatch_policy_without_host_native_keys() -> None:
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    contract_input = global_contract["input"]

    # The live host-probe evidence is withheld from this distribution.
    assert "host_probe" not in global_contract
    assert contract_input["rate_mismatch_policy"] == (
        "fail_closed_then_explicit_upstream_rate_adapter_gate"
    )
    assert "host_native_100hz_status" not in contract_input
    assert "if_host_native_rate_is_not_100hz" not in contract_input


def test_global_provenance_and_known_defects_no_longer_describe_cecht_as_offline_only() -> None:
    global_contract = load_json(GLOBAL_CONTRACT_PATH)
    method = global_contract["authority_and_provenance"]["methods"]["cecHT"]

    assert method["canonical_source"] == (
        "../../causal_phase_tracking/causal_cecHT/streaming_core.py"
    )
    assert method["upstream_source"] == (
        "../../causal_phase_tracking/causal_cecHT/vendor/"
        "eikeosmers_cecht/phase.py"
    )
    assert method["upstream_commit"] == "e079c3115f202a4272cf64143bf33ddddc247778"
    assert method["upstream_source_sha256"] == (
        "49b85cfd75f382aa9b12e00fca4cbfd903bf4daad0101ccba42b162f2af27bea"
    )
    assert method["license"] == "BSD-3-Clause"
    assert method["release_status"] == (
        "streaming_core_delivered_real_actuation_disabled"
    )

    defects = global_contract["known_open_defects"]
    assert (
        "cecHT_and_AR_are_offline_trailing_window_simulators_not_persistent_streaming_adapters"
        not in defects
    )
    assert "ARHT_remains_an_offline_trailing_window_simulator" in defects
    # The host output-semantics defect item is withheld from this distribution.
