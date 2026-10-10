# PhaseTracking Overview

---

## Chapter 1 Closed-Loop Neuromodulation Architecture and Product Requirements

### 1.1 Closed-Loop Neuromodulation

#### 1.1.1 Closed-Loop Neuromodulation Architecture

A typical closed-loop neuromodulation system can be understood as a continuously cycling architecture of sensing, estimation, decision, stimulation, and feedback.

The system first acquires EEG or other physiological signals through **sensing**, forming a continuous data stream with an explicit sampling rate, unit, polarity, and channel identity. This stage is the most fundamental, but it is also the one most prone to inconsistent sampling rates, channel-mapping errors, or inverted signal polarity; the input definition must therefore be fixed before the signal enters the algorithm.

The signal then enters **causal preprocessing**. The key requirement here is that all processing may use only the current and past samples and must never access future data. Typical operations include real-time filtering, resampling, and the artifact preprocessing required. Particular attention must be paid to the filter group delay and the system delay, because if these delays are not measured and compensated correctly, the subsequent phase estimate acquires a systematic bias.

The preprocessed signal then enters **phase/state estimation**, which estimates in real time the phase or frequency of the target oscillation, or some brain state, and simultaneously reports status information such as `ready` and `valid`. The algorithm must complete a sufficient warm-up before its output is considered valid, and it must also guard against instantaneous phase jumps, loss of lock, or unstable estimates.

At the **decision** stage, the system does not stimulate directly; instead it determines, from the current phase, state, and signal quality, whether stimulation should occur now. Its output is a logical stimulation intent, for example "the target phase is satisfied and the current state permits stimulation." This stage usually also incorporates eligibility conditions such as amplitude, artifact, sleep stage, and refractory period, in order to avoid triggering at unreliable or unsuitable moments.

Only after the decision stage confirms that the conditions are met does the **actuation** stage actually deliver stimulation or sham. Here the delay between the algorithm's decision and the hardware's actual output must be compensated; otherwise the target phase may already have shifted by the time the stimulation actually arrives. Stimulation artifact must also be handled, so that the stimulus does not contaminate the next round of phase estimation.

Finally, the system analyzes the physiological response before and after stimulation through **response assessment**, for example oscillation enhancement, phase locking, power change, or behavioral effect. This part usually does not participate directly in the individual millisecond-scale stimulation decision; it serves instead to evaluate whether stimulation is effective, to compare different parameters, and to guide subsequent strategy adjustment on a slower timescale.

The entire closed-loop architecture can therefore be summarized as:

sensing → causal preprocessing → phase tracking (estimation) → decision → actuation → response assessment

The first five modules constitute the real-time control loop, whereas response assessment evaluates the effect of stimulation from a higher level and provides feedback for subsequent parameter or strategy optimization.

> References: [R1] Esfahani MJ, Farboud S, Ngo HVV, Schneider J, Weber FD, Talamini LM, Dresler M. *Closed-loop auditory stimulation of sleep slow oscillations: Basic principles and best practices*. Neuroscience & Biobehavioral Reviews 153, 105379 (2023). DOI: https://doi.org/10.1016/j.neubiorev.2023.105379 ; [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468

#### 1.1.2 Closed-Loop Acoustic Stimulation for Sleep Regulation as an Example

- sensing: single-channel Fpz-Cz EEG, 250 Hz.
- causal preprocessing: band-pass filtering that uses past samples only.
- phase estimation: online estimation of slow-oscillation (0-4 Hz) phase.
- decision: trigger at the up-phase (`theta_target = 0 rad`); at most one stimulation per natural SO.
- actuation: short acoustic pulses, for example a 50 ms sine-Gaussian pulse train.
- response assessment: compare the delta-power change under stimulation versus sham (the literature reports +15.3%).
- An isomorphic non-sleep instance: phase-locked disruption of a pathological oscillation (tremor).

> References: [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468 ; [R4] Schreglmann SR et al. *Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence*. Nature Communications 12, 363 (2021). DOI: https://doi.org/10.1038/s41467-020-20581-7

### 1.2 Positioning of This Algorithm and Completed Components

#### 1.2.1 Positioning

- This algorithm is a phase-tracking component integrated into a designated host. Once the algorithms for the other closed-loop stages (eligibility, actuation, assessment) are loaded, it serves as one component within a complete closed-loop experimental chain.

#### 1.2.2 Completed Components

- Form: a single-file host-side external algorithm (`PhaseTracking_cecHT.py`); the algorithm proper is a strictly causal cecHT; the host interface and the output semantics are withheld from this distribution for intellectual-property reasons.
- Frozen parameters and the current profile: see 3.5 and 3.7.
- On-device validation: completed; see 4.2.2 for the operating facts and the items still not closed.

#### 1.2.3 Correspondence with 1.1.1

| Loop stage | Coverage |
|---|---|
| sensing | Provided by the host (single-channel EEG, 250 Hz) |
| causal preprocessing | Covered (causal band-pass within the algorithm) |
| phase estimation | Covered (the core of this product) |
| decision | Partially covered (record-only determination of target-phase crossing) |
| actuation | Not covered |
| response assessment | Not covered |

### 1.3 Algorithm Requirements
#### 1.3.1 Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | The user can select an arbitrary tracking band |
| FR-2 | The user can select an arbitrary target phase |
| FR-3 | Functionality remains stable after a parameter change (no failure and no marked degradation) |
| FR-4 | Support for extension to 8 channels |
| FR-5 | Phase and events are observable, for the GUI and for acceptance testing to read |

#### 1.3.2 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Sampling rate: expected to remain 250 Hz; if the user selects another sampling rate, resampling is performed (the causality constraint must be preserved) |
| NFR-2 | Integrated into the host, following the host's external algorithm contract |
| NFR-3 | Accuracy: to reach the level of performance commonly reported in the literature for comparable deployable real-time phase-tracking systems |
| NFR-4 | Resources and latency: the duration of a single callback is far below the host callback interval, with no deadline miss |
| NFR-5 | Reproducibility: fixed parameters + version identifier + startup self-check |
| NFR-6 | The host GUI must subsequently be adapted to present phase, parameters, and status |

**Literature reference for NFR-3** (real-time closed-loop stimulation settings):

| Source | Measure | Reported value |
|---|---|---|
| Ferster 2022 [R3] | Proportion of stimulation triggers falling in the SW up-phase; 324 recording segments (healthy older adults + PD patients) | **> 70%** for all three algorithms; a decline in performance under low-amplitude and non-constant-frequency conditions is common to all algorithms |
| Santostasi 2016 [R2] | Proportion of stimuli falling within ±30° of the target phase | **> 95%** when the amplitude exceeds 90 µV; without an amplitude threshold, about **80%** fall within the selected phase interval and **79%** fall in the up-state (180-360°); rapid phase jumps (in < 5% of cases) produce mismatches |
| Santostasi 2016 [R2] | Engineering latency | Total latency **70 ± 5 ms** (of which about 40 ms comes from the sound card) |

Concluding statement: the common level of performance for systems of this kind is that "most triggers fall within the target phase window" (about 70%-95%, strongly dependent on signal amplitude and frequency stability), accompanied by a total latency on the order of tens of milliseconds that is measurable and compensable. Santostasi 2016 states explicitly that adding a simple amplitude threshold markedly improves phase accuracy [R2]; this point is used again in the discussion of amplitude eligibility in later chapters.

> References: [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468

---

## Chapter 2 Comparison of Offline Candidate Methods: Benchmark Measures and Results

### 2.1 Scope of This Chapter

This chapter addresses the following: which candidate phase-tracking methods were compared before committing to a streaming implementation, with what reference measure and metrics they were evaluated, what results were obtained on this project's data, and the limits of their validity; it does not cover the real-time implementation or on-device validation (see Chapters 3-4).

### 2.2 The Three Methods Compared

#### 2.2.1 pll

pll is the canonical classic phase-locked-loop estimator. Its update is recursive and sample-by-sample, depends only on the current sample and its internal state, and therefore has constant per-sample cost and no analysis window: the method is strictly causal.

The loop structure and its tuning values are not disclosed here: the input band, filter type and order, center frequency, KVCO, loop coefficients, phase offset, normalization percentile, amplitude percentile and refractory appear as empty (`None`) placeholders in the configuration schema. Two qualitative facts are retained, because they belong to the evaluation protocol rather than to the loop design: input normalization and amplitude gating are computed over the whole record, so both are evaluation-time settings that an online implementation cannot reproduce.

pll.py is subject to intellectual-property constraints and cannot be disclosed publicly in this distribution.

> References: [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007
#### 2.2.2 cecHT

cecHT (endpoint-corrected Hilbert transform) takes, at each time point, a sliding window ending at the current sample, applies band-pass filtering and the Hilbert analytic transform to the signal within the window, and applies endpoint correction at the end of the window in order to remove the endpoint error caused by truncation of the finite window; the angle of the corrected analytic value is the current phase. An optional calibration derives a complex gain analytically and further compensates the phase and amplitude bias introduced by the finite window.

In the offline benchmark, cecHT recomputes the FFT point by point over the sliding window; it is a simulator rather than a stateful streaming object. Because the window covers only the current and past samples, it is endpoint-causal. The parameters used in this project are a 0.5-1.25 Hz band-pass, a first-order Butterworth filter, a 6 s window length, 80th-percentile amplitude gating, and a 1.2 s refractory; at the input, a zero-phase 0.3-35 Hz preprocessing step is used in addition.

Open-source provenance: eikeosmers/cecHT; its ECHT implementation derives from MEEGkit, the original method is due to Schreglmann et al., and the optional calibration follows the analytical result of Osmers and Kolossa.

> References: [R4] Schreglmann SR et al. *Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence*. Nature Communications 12, 363 (2021). DOI: https://doi.org/10.1038/s41467-020-20581-7 ; [R5] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). DOI: https://doi.org/10.48550/arXiv.2601.13962 ; [R6] Osmers E. `eikeosmers/cecHT`, pinned commit `e079c3115f202a4272cf64143bf33ddddc247778`, BSD-3-Clause. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778 ; [R7] MEEGkit ECHT implementation, `meegkit/phase.py`, BSD-3-Clause. https://github.com/nbara/python-meegkit/blob/master/meegkit/phase.py

#### 2.2.3 ARHT

ARHT (Autoregressive Forward-Prediction Hilbert Transform) first fits an autoregressive model to the signal over a rolling past window, then uses that model to predict a short segment of the signal forward, applies the Hilbert transform to the predicted segment, and takes the phase at the end of the predicted segment as the current phase. The motivation is that the Hilbert transform requires a relatively complete oscillation cycle to produce a stable phase; taking the phase directly from a causally filtered signal produces conspicuous error at phase turning points, and forward prediction can compensate for this delay.

The parameters used in this project are a 6.0 s past window, AR order 12, a 200 ms prediction horizon, 500 ms Hilbert padding, a 50 ms update step, and 80th-percentile amplitude gating. An earlier, independent AR forward-prediction implementation also exists; its results serve only as corroborating evidence.

Open-source provenance: OpitzLab/CL-phase; that implementation was originally designed for alpha-band closed-loop EEG-TMS, and this project recalibrated it to 0.5-1.25 Hz slow waves.

> References: [R8] `OpitzLab/CL-phase`: real-time phase detection for closed-loop EEG-TMS (MATLAB). https://github.com/OpitzLab/CL-phase ; [R9] Shirinpour S et al. *Experimental Evaluation of Methods for Real-Time EEG Phase-Specific Transcranial Magnetic Stimulation*. Journal of Neural Engineering (2020). DOI: https://doi.org/10.1088/1741-2552/ab9dba

### 2.3 Evaluation Measure: Offline Hilbert Phase as Ground Truth

#### 2.3.1 Offline Reference Phase (Golden Standard)

The instantaneous phase of an oscillation has no independently measurable physical ground truth; the standard practice when evaluating a real-time phase estimator is therefore to construct a reference phase as the comparison baseline: apply zero-phase (non-causal) band-pass filtering to the entire recording, then take its analytic phase via the Hilbert transform. This reference phase uses future data and can serve for evaluation only; comparing the output of each estimator against it point by point is the prevailing measure in real-time phase-tracking benchmarks [R1][R3].

This project implements the reference phase under the same measure: a 4th-order Butterworth band-pass (0.5-1.25 Hz) + zero-phase filtfilt + Hilbert transform, outputting phase, analytic amplitude, and instantaneous frequency; the implementation resides in `offline_src/hilbert_phase.py`, and the filtering and Hilbert transform are provided by SciPy [R10].

> References: [R1] Esfahani MJ, Farboud S, Ngo HVV, Schneider J, Weber FD, Talamini LM, Dresler M. *Closed-loop auditory stimulation of sleep slow oscillations: Basic principles and best practices*. Neuroscience & Biobehavioral Reviews 153, 105379 (2023). DOI: https://doi.org/10.1016/j.neubiorev.2023.105379 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468 ; [R10] SciPy signal processing API: Butterworth/SOS, `filtfilt`, Hilbert transform. https://docs.scipy.org/doc/scipy/reference/signal.html

#### 2.3.2 Data and Segmentation

Data are taken from the Fpz-Cz channel of Sleep-EDF Expanded [R11]. Different runs use different scales: two single-subject versions with 19 epochs and 100 epochs, and an ARHT preview with 20 subjects × 3 epochs. A margin of 3 s is kept before and after each epoch (`evaluation_edge_trim_sec`), and candidate trigger points are drawn at a fixed step.

> References: [R11] PhysioNet Sleep-EDF Expanded. https://physionet.org/content/sleep-edfx/1.0.0/ ; Kemp B et al. IEEE Transactions on Biomedical Engineering 47(9), 1185-1194 (2000). DOI: https://doi.org/10.1109/10.867928

#### 2.3.3 Trigger Definition and Gating

Each method produces a trigger when the estimated phase crosses the target phase in the positive direction, and a refractory suppresses repeated triggering from jitter (1.2 s for cecHT, 0.8 s for pll and ARHT), while amplitude gating filters out low-amplitude events. Two aspects are evaluation-time settings: the amplitude-gating threshold is taken as that method's percentile (80th) over the entire valid epoch, and the input normalization is likewise based on the whole record; both use future information and cannot be replicated on the online path. Moreover, the number of triggers and the gating conditions differ across methods, so their hit rates are not a paired comparison over the same set of physiological opportunities.

#### 2.3.4 Metric Definitions

The core metric is the phase error at the trigger instant, defined as `wrap(φ_ref@trigger − θ_target)`, that is, the difference between the reference phase at the trigger instant and the target phase. On this basis, the following are reported: the proportions falling within π/6, π/4, and π/3; the mean and the median of the error; the circular mean of the error; the number of triggers and the trigger rate; the per-epoch runtime and the failure rate. Two aspects of the measure must be stated: `circular_resultant_length` in the code is in fact computed as the resultant length of the reference phase at the trigger instants rather than of the error, so it characterizes the concentration of the reference phase; and when an epoch has no trigger it is recorded as a zero-valued error, which manifests on different metrics as "looking perfect" or "looking worst," respectively. The statistical definitions of the circular mean and the resultant length are given in the References.

> References: [R12] Berens P. *CircStat: A MATLAB Toolbox for Circular Statistics*. Journal of Statistical Software 31(10) (2009). DOI: https://doi.org/10.18637/jss.v031.i10

### 2.4 Measured Results for This Project

#### 2.4.1 Single Subject, 100 Epochs (Main Result)

Data source: `outputs/phase_method_benchmark_local/phase_method_results.csv`.

| Method | Triggers | Trigger rate (/min) | π/6 | π/4 | π/3 | mean_abs | median_abs | circular mean | R | runtime (ms/epoch) | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cecHT | 332 | 6.64 | 0.7169 | 0.8795 | 0.9277 | 0.4179 | 0.3284 | 0.1664 | 0.8753 | 19.54 | 0 |
| pll | 311 | 6.22 | 0.5884 | 0.8071 | 0.9068 | 0.5065 | 0.4239 | −0.0719 | 0.8171 | 12.86 | 0 |

The two methods produced similar numbers of triggers (332 and 311), so this set of results is relatively comparable in terms of triggering opportunities; cecHT is superior to pll on every hit-rate metric and every error metric, and its circular mean is closer to zero.

#### 2.4.2 Small-Sample Run (Preliminary)

The small-sample results are given below; their main purpose is pipeline validation and preliminary screening.

Single subject, 20 epochs (`CLAS_initial_benchmark/outputs/phase_method_benchmark_single_subject/phase_method_results.csv`): cecHT 76 triggers, π/4 = 0.8947; pll 73 triggers, π/4 = 0.8767.

#### 2.4.3 ARHT Preview Across 20 Subjects

Data source: `CLAS_initial_benchmark/outputs/reports/ar_forward_preview_20_subjects_summary.csv`. This run covers 20 subjects and 60 epochs, with 44 triggers in total (1.47/min), π/4 = 0.6591, mean_abs = 0.6694, median_abs = 0.6112, circular mean = 0.4468, R = 0.7860, and a prediction RMSE of about 23.9 µV. Its trigger rate is markedly lower than that of the previous two sets, indicating that ARHT yields fewer triggering opportunities under this parameter set and gating; because its data scale and gating conditions differ from those of the other methods, this result cannot be compared directly with 2.4.1.

### 2.5 Chapter Summary

This chapter describes an offline benchmark: it measures the agreement of each estimator with a non-causal reference phase, and serves to compare algorithm performance at a basic level, to detect implementation errors, and to screen methods. It differs fundamentally from a streaming implementation—the reference phase, whole-record normalization, and whole-record gating cannot enter the online path—so the results in this chapter do not constitute a performance conclusion for a deployable streaming system.

---

## Chapter 3 Real-Time Implementation: Layered Architecture, Streaming Semantics, and Frozen Parameters

### 3.1 Scope of This Chapter

This chapter addresses what form the currently deployed real-time cecHT takes, how it is layered, where it differs from the offline version, and which parameters and semantics are frozen; it does not cover performance figures or error diagnosis (see Chapter 4), nor subsequent optimization and productization (see Chapters 5-6).

### 3.2 Layered Architecture

The real-time cecHT consists of four layers, whose responsibilities become successively tighter from the bottom up.

The lowest layer is the **mathematical layer**: a fixed-version upstream ECHT implementation that defines the band-pass response, the endpoint correction, and the optional complex-gain calibration. It answers only "how the phase should be computed" and involves no streaming state; this project freezes this layer at a pinned commit and license [R4][R5][R6].

Above it is the **streaming core layer**: at construction time it compiles the mathematical layer once into a fixed kernel corresponding to "the analytic endpoint of the current sample"; thereafter each sample requires only one or two dot products between the most recent N samples in the ring buffer and this kernel to obtain the phase of the current endpoint. The hot path therefore calls neither the FFT nor the upper layer's `transform()` interface, and in steady state the computational cost is constant and no dynamic allocation occurs. The kernel is uniquely determined by the construction parameters and is read-only; any change in sampling rate, window length, frequency band, filter order, or `f0` requires reconstruction [R10].

The third layer is the **trigger layer**: an independent pseudo-trigger that consumes only the `valid` and `ready` phases output by the core, detects the instant of positive crossing of the target phase, performs linear interpolation, and emits timestamped events. It produces no real stimulation and does not depend on the core's internal implementation.

The topmost layer is the **host adaptation layer**: a single-file host plugin. It moves the data blocks passed in by the host callback into the contiguous buffer required by the core, maintains the per-instance absolute sample count and state, sends the phase into the in-memory ring buffer and the sidecar log, and returns the raw waveform to the host unchanged.

> References: [R4] Schreglmann SR et al. *Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence*. Nature Communications 12, 363 (2021). DOI: https://doi.org/10.1038/s41467-020-20581-7 ; [R5] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). DOI: https://doi.org/10.48550/arXiv.2601.13962 ; [R6] Osmers E. `eikeosmers/cecHT`, pinned commit `e079c3115f202a4272cf64143bf33ddddc247778`, BSD-3-Clause. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778 ; [R10] SciPy signal processing API: Butterworth/SOS, `filtfilt`, Hilbert transform. https://docs.scipy.org/doc/scipy/reference/signal.html

### 3.3 Key Differences from the Offline Benchmark Version

| Dimension | Offline benchmark version | Real-time version |
|---|---|---|
| Phase computation | Point-by-point FFT recomputation over a sliding window | A fixed kernel derived at construction time; the hot path uses only one or two dot products |
| State | Whole-record arrays, re-initialized on each call | Fixed-capacity ring buffer + explicit sample index and fault state |
| Causality | Endpoint-causal, but with zero-phase preprocessing at the input | Strictly causal, with the band-pass contained within the kernel |
| Sampling rate | 100 Hz EDF processed directly | 250 Hz grid; anything other than 250 Hz is rejected outright |
| Parameters | Window 6 s, order 1, whole-record amplitude gating | Window 8 s, order 2, `n_fft = N`, calibrate `f0 = 0.8 Hz`, no amplitude gating |
| Output | Phase arrays and a trigger simulation | Per-sample `phase/valid/ready/reason` and events; the host side returns a waveform of the same shape |
| Triggering | Whole-record percentile gating | Refractory only; the target can be updated atomically at runtime |

The substance of the difference lies in two points. The first is **causality**: the offline version recomputes the FFT over a sliding window and applies zero-phase filtering upstream, whereas the real-time version folds all of these steps into a single strictly causal kernel. The second is **state**: the offline version processes whole-record arrays in a single pass, whereas the real-time version must compress the state into a fixed-capacity ring buffer with an explicit index, and must produce identical output under any legal blocking scheme.

### 3.4 Streaming Semantics

**warm-up and ready**: only after the ring buffer has been filled (N samples accumulated) is the endpoint marked `ready`; before that, the output is `valid = false`, does not enter the phase-error statistics, and is not permitted to trigger.

**Fault and failure semantics**: the core distinguishes three kinds of events. A structural rejection means that the specification of the input or output buffer is invalid; in that case no output slot is written and the dynamic state is cleared. A stream fault means that the data stream itself is invalid (non-finite input, a mismatch of sampling rate/unit/channel/polarity, a break in the sample sequence, duplication, or out-of-order samples); in that case the block is filled with invalid slots, the state is cleared, and a machine-readable reason code is recorded. An epoch change clears the dynamic state and re-ingests. In no case is an old phase carried forward, and an invalid or not-ready phase must not trigger.

**reset**: clears the dynamic ring buffer, readiness, index, and fault state, while retaining the immutable configuration, kernel, and calibration; both a change of sampling rate and algorithm unloading follow this path.

**target and crossing**: the default is `theta_target = 0 rad` (that is, the Fpz-Cz positive peak), and it can be updated at runtime; only positive crossings are detected, and an event is registered at the linearly interpolated instant, while reverse crossings are not recorded as diagnostic events. A target update takes effect atomically at the declared block or sample boundary; it simultaneously clears the crossing comparator and cancels any pending event not yet processed, without replaying old crossings.

**Gating**: the initial implementation contains no amplitude eligibility.

**Time semantics**: the time position of a phase estimate is the last input sample of the sliding window; an event records both `estimated_crossing_time` and `decision_time`, which differ in meaning and must not be conflated. The three compensations—phase offset, scheduling prediction horizon, and acoustic output latency—are all currently zero, and the contract requires that each be compensated exactly once.

### 3.5 Frozen Parameters

The currently effective parameters are given in `host_plugin_release` of the project-level contract `realtime_deployment/contracts/phase_tracking.json`: the sampling rate, window length, FFT grid, frequency band and filter, calibration, target, refractory, per-block limit, and phase ring buffer are all defined there and are not repeated in this document.

The algorithm performs no internal resampling; the rationale for the input sampling rate and the resampling strategy is given in 6.3.2.

### 3.6 Deployment Form and Host Interface

The deployment artifact is a single-file plugin generated by the build script from the frozen source, carrying a version identifier and a verification hash; on load it performs a one-time self-check against a built-in kernel golden, in order to confirm that the numerics of the host environment match those of the development machine.

The host interface follows the existing contract: `__init__(fs)`, `process(x, stim_label)`, `reset()`. process must return a waveform of the same shape as its input, so the phase cannot be written back through the return value; it is exposed instead through the in-memory ring buffer and the sidecar log.

The as-measured facts on the host side (the number of instances; the callback's failure to provide channel ID, segment ID, absolute sample index, or acquisition timestamp; the semantics of `reset()`; the circuit-breaker threshold and warm-up frame count; and the plugin's position after the host's filtering and resampling) are not repeated here. The project-level contract `realtime_deployment/contracts/phase_tracking.json` records the corresponding requirements rather than those measurements: the metadata the input must carry is in `input.required_metadata`, the fault and reset semantics and the warm-up requirement are in `streaming_contract`, and the release and safety requirements are in `safety_and_release`; the one-independent-instance-per-channel rule is in `streaming_core` of the scoped cecHT contract, and the circuit-breaker threshold and warm-up frame count are host-side measurements that no contract carries. Support for 8 channels is in itself a product requirement, and its channel-identity mapping and per-channel parameter management remain to be implemented later.

The time budget is a hard constraint on the hardware side: after warm-up, the host continuously monitors the plugin's execution time, and a run of consecutive over-threshold values triggers protection (the frame count and the threshold are host-side measurements, not contract fields); a single callback must be far below that threshold, and the measured data are given in Chapter 4. The 4 ms target is stricter than the circuit-breaker threshold measured for this build, and planning should follow the former.



### 3.7 Currently Effective Profile

The current deployment artifact and the canonical evaluation baseline use the same parameter set: the deployment-side values are given in `host_plugin_release` of the project-level contract, and the evaluation-side values are given in the canonical baseline of `cecHT_Experiment_Contract.md`. The contract has been updated to this profile; the sampling rate, phase convention, target semantics, endpoint definition, and fault/reset semantics are all unchanged.

---

## Chapter 4 Current Performance and Error Attribution: Comparison with the Field and Room for Optimization

### 4.1 Scope of This Chapter and Definitions of the Numbers

This chapter addresses how the current implementation performs, where the error comes from, and how far it remains from a deployable level; it does not cover the specific optimization steps to be taken (see Chapter 5) or productization and delivery (see Chapter 6). To prevent misreading, the numbers in this chapter fall into three classes that are not directly comparable with one another: per-sample estimation error (the difference between each ready endpoint and the offline reference phase), event-level trigger error (the difference between the reference phase at the pseudo-trigger instant and the target phase), and on-device engineering metrics (execution time, callback interval, and fault counts). The conversion between angle and time depends on the oscillation frequency: at 0.8 Hz, ±30° corresponds to about 104 ms and ±45° to about 156 ms; the same angular threshold placed in a wider reference band corresponds to a shorter time tolerance, a point used again when comparing against the literature in 4.4.

### 4.2 Performance of the Current Implementation

#### 4.2.1 Offline Pseudo-Streaming Evaluation

On a development set of 20 nights, 13 subjects, and 3 segments of 180 s per night, evaluated sample by sample under the frozen configuration with an 8 s window and a calibration center frequency `f0 = 0.8 Hz`, the pooled mean absolute wrapped error is 1.0412 rad, the median 0.8293 rad, q95 2.7297 rad, the circular bias −0.2945 rad, the resultant vector length 0.4332, the circular standard deviation 1.2936 rad, the ±30°/±45° hit rates 0.3336/0.4784, and the catastrophic-point (error exceeding 135°) rate 9.82%, over a total of 2,580,060 ready endpoints entering the statistics. At the event level, the default target of 0 rad yields 9,183 pseudo-triggers, whose reference phase falls within ±30°/±45° at rates of 0.2719/0.4054, with a median absolute time error of 0.1587 s; pooling the eight targets gives 73,760 events, a ±45° hit rate of 0.4415, and a median absolute time error of 0.1471 s. Note that these figures were obtained on the development set, and that set participated in the configuration selection; they therefore describe "the level of the current configuration," not a generalization estimate.

#### 4.2.2 On-Device Operating Facts

In a run on the host acquiring at 250 Hz and with no stimulation connected, the host created 8 independent instances, one per channel; each instance was called 1,593 times and processed 16,891 samples (about 67.6 s); the first ready endpoint appeared at exactly the 1999th sample, that is, the 8 s warm-up holds precisely. A single callback took 153-287 µs on average, that is, 0.4%-0.7% of the 42.4 ms callback interval, with a worst case of 3.24 ms; 157 pseudo-triggers were recorded during this period, with no stream fault, structural rejection, or internal error. On load, the plugin self-checked against the built-in kernel fingerprint, and the worst relative deviation across the 8 instances was 0.0. These results show that the engineering chain holds and that the time margin is ample, but they do not constitute evidence of accuracy. Four items remain unclosed on the device, and later chapters take this list as authoritative: the correspondence between channel instances and physical channels has not been established, the input polarity has not been confirmed, the input unit has not been confirmed, and the phase has not yet been reported to the interface.

#### 4.2.3 Apparent Conflict with the Earlier Offline Benchmark

An early offline comparison on a single subject with 100 epochs recorded a ±45° hit rate of 0.8795 for cecHT at the trigger instants, whereas the present per-sample ±45° hit rate is 0.4784. These two figures are not contradictory, and neither should they be subtracted from one another: the former is computed only over "the instants at which the algorithm itself triggered," its input further includes non-causal preprocessing (whole-record zero-phase filtering and whole-record percentile normalization), and its parameters were selected and reported on the same batch of epochs; the latter is computed over all ready endpoints, its input is strictly causal, and it covers 13 subjects. It is normal for the same algorithm to exhibit different levels under two measures; what actually needs to be answered is what the latter still lacks relative to a deployable level, which is precisely the subject of 4.3 and 4.4.

### 4.3 Sources of Error

#### 4.3.1 Implementation and Alignment Have Been Excluded

After passing a known 0.8 Hz sinusoid through exactly the same chain as the real data, the error was far lower than on the real data: there is no systematic defect in the reference-phase construction, the endpoint semantics, or the group-delay alignment, and the high error on real data can only come from the data and the parameters, not from an implementation error. The specific values for the synthetic signal and the chain details are given in §B.1 and §C of `cecHT_Experiment_Contract.md`.

#### 4.3.2 Frequency Mismatch Is the Dominant Term

It has been verified experimentally that the error is strongly correlated with the absolute value of "the instantaneous frequency of the offline reference minus the calibration center frequency," and that the direction is consistent at both the pooled and the subject level. Using the offline dominant frequency per window compresses the error to the order of 0.63 rad and improves every subject; switching to "a fixed dominant frequency per record" is instead worse, improving only 1 subject. This shows that the bottleneck is not a difference in center frequency between individuals or nights but rather frequency drift within a record—an estimator with a fixed center frequency cannot track the drift of the slow wave itself within 0.5-1.25 Hz, so a dynamic center frequency is the appropriate direction. The original analysis of ecHT places center-frequency mismatch alongside window length, bandwidth, and filter order as a design rule governing the residual endpoint bias, and notes that residual leakage sets an irreducible lower bound on the variance of the endpoint error, consistent with the direction measured in this project [R5]. The individual values, the subject-level consistency, and the comparison branches are given in §D.2 of `cecHT_Experiment_Contract.md`.

> References: [R5] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). DOI: https://doi.org/10.48550/arXiv.2601.13962

#### 4.3.3 Coupling Between Window Length and Frequency Resolution

The upstream implementation samples the band-pass response directly on an `n_fft`-point grid: `_design_bandpass` evaluates `sosfreqz` on `fftfreq(n_fft)` and then combines that response with the analytic-signal multiplier. Hence, when `n_fft` is set equal to the window length N, the window length simultaneously determines the temporal locality and the density of the frequency grid. It has been verified experimentally that, once the grid is fixed (`n_fft` fixed at 2000, with short windows zero-padded), the short window still outperforms the 8 s window, showing that the short-window advantage is real; however, in the sweep in which "window length equals grid length," a substantial part of the short-window gain came from grid coarsening, and its magnitude was markedly amplified by frequency-domain discretization [R6][R10]. The per-window values and the ordering consistency of the two sweeps are given in §D.3 and §D.4 of `cecHT_Experiment_Contract.md`.

> References: [R6] Osmers E. `eikeosmers/cecHT`, pinned commit `e079c3115f202a4272cf64143bf33ddddc247778`, BSD-3-Clause. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778 ; [R10] SciPy signal processing API: Butterworth/SOS, `filtfilt`, Hilbert transform. https://docs.scipy.org/doc/scipy/reference/signal.html

#### 4.3.4 Why the Optimal Window Length Falls on the Search Boundary

Once the grid is fixed, the best-performing candidate falls exactly on the lower search boundary (about 0.6 slow-wave cycles), and there is no interior extremum within the interval—this is a design boundary, not a physiological optimum. There are two reasons. A window of length T has a frequency resolution of about 1/T, so resolving the 0.75 Hz-wide band from 0.5 to 1.25 Hz requires a window length of at least about 1.3 s; and a slow wave is a non-stationary, non-sinusoidal oscillation whose instantaneous phase generally requires 1-2 complete cycles, that is, 1.25-2.5 s. Together these two considerations pin the reasonable interval to 1-2.5 s.

Other implementations handle this differently, and can be grouped into four kinds. The original analysis of ecHT does not treat the window length as a fixed constant but instead gives a joint design rule for window length, bandwidth, filter order, and center-frequency mismatch, and explains the residual bias in terms of the endpoint group delay; its official experiment scripts likewise treat window length, bandwidth, order and type, frequency drift, and signal-to-noise ratio as one set of sweep parameters [R5]. A pure phase-locked loop (PLL) has no window at all: it updates the phase recursively sample by sample, at the cost of a trade-off between lock time and loop bandwidth [R2]. Broadband methods based on the Hilbert transform delegate temporal resolution to the filter bandwidth rather than the window length; for example, a 0.5-4 Hz broadband reference paired with the Hilbert transform has an equivalent timescale of about the reciprocal of the bandwidth (about 0.3 s) [R3]. Newer deployment-level implementations replace a single narrowband DFT with a multiscale basis, for example a 2 s analysis window paired with 20 Morlet wavelets covering 0.6-2.0 Hz, using a single wavelet family to obtain temporal and frequency resolution simultaneously [R14].

The conclusion is that the correct next step is not to keep moving toward shorter windows but first to decouple the frequency grid from the window length, then to bracket the expected optimum within the 1-2.5 s interval from both sides, and to reconfirm the ordering under a fixed grid; the sweep interval and the boundary evidence are given in §D.4 and §E.2 of `cecHT_Experiment_Contract.md`.

> References: [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468 ; [R5] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). DOI: https://doi.org/10.48550/arXiv.2601.13962 ; [R14] `gmilab/ptas_benchmarks` (TWave), pinned commit `566dec18441d4234cf9218e4771d84c8e7fbad4c`, Zenodo v1.0.0 (2025-11-21), DOI: https://doi.org/10.5281/zenodo.17666778. License not yet clarified; architecture and parameter reference only.

#### 4.3.5 Other Causes Not Intrinsic to the Algorithm

Beyond the estimator itself, several causes related to the task, the signal, and the chain also depress the hit rate of the real-time version. Most of these have a corresponding processing stage in publicly available deployment-level implementations, whereas this project has either not yet implemented them or has an offline code implementation that has not yet been made real-time.

First, the slow wave itself is harder to track than rhythms such as alpha. It lies in the low-frequency band of 0.5-1.25 Hz, has low amplitude and large inter-individual variability, is non-sinusoidal with asymmetric upper and lower phases, and drifts in frequency continuously; published work explicitly identifies low amplitude and non-constant frequency as independent difficulties for real-time sleep phase tracking [R3]. The performance of the same estimator in different frequency bands therefore cannot be directly analogized, and the accuracy readily achieved in the alpha band cannot be expected to reproduce on slow waves.

Second, amplitude gating is absent. Slow-wave phase error is strongly correlated with signal amplitude: restricting the statistics to high-amplitude events markedly raises the hit rate—published reports show that for amplitudes above 90 µV, more than 95% of pulses fall within ±30° of the target phase, whereas the phase distribution is much wider in whole-record statistics [R2]. The current implementation counts all ready endpoints, so the denominator includes many low-amplitude, non-slow-wave, and artifact periods, which systematically depresses the hit rate.

Third, state and quality gating is absent. Public real-time systems generally gate before phase tracking: in closed-loop sleep regulation, for example, long-term NREM power and short-term delta/beta activity are first used to select stimulation windows, after which the phase algorithm operates within those windows [R3]; highly cited reviews also note that fixed thresholds are sensitive to the individual and to the montage, and that no consensus optimal online criterion exists to date [R1]. The current implementation evaluates only within the selected N3 segments, which still contain many periods that are not ongoing slow waves, and it has neither an independent artifact channel nor a signal-quality criterion. Offline code for state and quality gating in the sleep-regulation setting exists in this project, but it has not been made real-time and is therefore not packaged into the plugin.

Fourth, the system latency has not been compensated. The three compensations—phase offset, scheduling prediction horizon, and acoustic output latency—are all currently zero, while the time position of a phase estimate is the last input sample of the sliding window and the event time differs from the decision time; at 0.8 Hz, an uncompensated latency of about 100 ms amounts to a systematic phase error of about 29°. Public implementations generally compensate for latency, for example by converting the measured 70 ± 5 ms total latency into a phase lead and triggering earlier [R2], or by using a 150 ms prediction horizon [R14].

Fifth, the definition of the reference phase itself affects the numbers. The angle-to-time conversion is given in 4.1; if the reference band is wider (for example 0.5-4 Hz), the same angle corresponds to a shorter time tolerance and a steeper waveform, and the denominator and the criterion for the hit rate change accordingly [R3]. The reference definitions must therefore be aligned before hit rates are compared across papers.

Sixth, the single-channel input limits the information available. There is currently only one Fpz-Cz lead, with neither EOG/EMG nor any other channel for artifact rejection, and no spatial consistency with which to judge whether a given slow wave is reliable; public causal implementations use independent EEG/EOG/EMG thresholds in parallel to determine artifact and disengagement [R14].

Seventh, several prerequisites on the host side remain unclosed (see 4.2.2 for the list). Polarity is the most critical: if the input polarity is reversed, the phase shifts by π overall and any hit-rate statistic becomes invalid; this must be confirmed on the device first, and it is not a problem the algorithm can correct on its own.

> References: [R1] Esfahani MJ, Farboud S, Ngo HVV, Schneider J, Weber FD, Talamini LM, Dresler M. *Closed-loop auditory stimulation of sleep slow oscillations: Basic principles and best practices*. Neuroscience & Biobehavioral Reviews 153, 105379 (2023). DOI: https://doi.org/10.1016/j.neubiorev.2023.105379 ; [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468 ; [R14] `gmilab/ptas_benchmarks` (TWave), pinned commit `566dec18441d4234cf9218e4771d84c8e7fbad4c`, Zenodo v1.0.0 (2025-11-21), DOI: https://doi.org/10.5281/zenodo.17666778. License not yet clarified; architecture and parameter reference only.
### 4.4 Comparison with Implementations at a Deployable Level

#### 4.4.1 The Level Reported in Public References

The three sets of results in the public literature closest to this project's task have their measures and reported values listed in the NFR-3 reference table of 1.3.2; only details not contained in that table are added here. A wearable benchmark aimed at low-amplitude slow waves compared three methods on 324 recording segments, with the proportions of triggers falling in the slow-wave rising phase at approximately 86%, 72%, and 70%, and it identified low amplitude and slow waves above 1 Hz as the most difficult conditions [R3]. The sleep PLL work further reports that only when the statistics are restricted to slow waves with amplitude greater than 70 µV does the phase distribution converge from 240.37 ± 25.61° to 243.15 ± 3.06°, with 79% of pulses falling in the rising phase [R2]. A more recent deployment-level implementation reports the hit rate and the phase error separately, and maintains trigger quality using an amplitude range, quadrant consistency, a high-to-low frequency ratio, and a back-off strategy [R14].

#### 4.4.2 Differences in Composition

These implementations differ greatly from the current version of this project in "what is in the chain"; the role of each stage is explained in 4.3.5, and only what they actually contain is summarized here: amplitude gating, for example restricting the events entering the statistics to an amplitude range of 80-600 µV, or reporting stratified by amplitude above 90 µV [R2][R14]; sleep-state gating, first using long-term NREM power and short-term delta/beta activity to select windows [R3]; artifact and disengagement determination, using independent EOG/EMG thresholds [R14]; quadrant consistency and locking criteria, requiring the phase to agree with its quadrature component before triggering is allowed [R14]; and explicit latency compensation [R2][R14]. The current estimator and pseudo-trigger of this project deliberately exclude the first three, set the three latency compensations all to zero, and accept only a single-channel input. In the literature, these stages are not optional; they are the premises on which the accuracy figures hold.

#### 4.4.3 Differences in Test Content and Statistical Measures

The statistical treatment likewise differs. The literature reports hit rates mostly per trigger or per event, and often under conditions already filtered by gating; this project reports both per-sample estimation error and event-level trigger error, the per-sample measure applying no filtering at all. The literature presents pooled means or across-subject standard deviations and rarely gives subject-level confidence intervals; this project gives subject-level bootstrap intervals and Wilson intervals for the hit rate. The target-phase setting also differs: public work often uses a single target phase (for example 240°) or the interval "rising phase," whereas this project uses 8 equally spaced target phases and reports the hit rate of the worst target, which exposes weaknesses that are averaged away under a single target. Population and sample size also differ: nap recordings from 5 subjects [R2], 324 wearable segments from healthy older adults and Parkinson's disease patients [R3], and a cohort composed mainly of epilepsy and healthy subjects [R14], cannot be directly substituted for this project's public sleep dataset of 20 nights and 13 subjects.

#### 4.4.4 How Far These Differences Can Explain the Gap

Taken together, the gap between the current 33.4% (per-sample ±30°) and the 70% or more reported in the literature cannot be compared directly by numerical value, because the two do not rest on the same testing standard: the literature figures are founded on "confirmed deep sleep, confirmed presence of a slow wave of sufficient amplitude, compensated system latency, and triggering protected by a consistency criterion," whereas the current figures are responsible for all ready endpoints. The direction of expected improvement is clear: supplying amplitude gating will change the denominator of the hit rate, restricting the statistics to genuine slow-wave events; supplying time compensation will remove a systematic bias on the order of 20-30°; and completing the center-frequency and window-length optimization will directly lower the mean absolute error. The quantified expectations of these differences are given in 4.5, and each of them must be confirmed on data independent of the development set rather than asserted directly from the comparison in this document.

> References: [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE Transactions on Biomedical Engineering 69(9), 2916-2925 (2022). DOI: https://doi.org/10.1109/TBME.2022.3157468 ; [R14] `gmilab/ptas_benchmarks` (TWave), pinned commit `566dec18441d4234cf9218e4771d84c8e7fbad4c`, Zenodo v1.0.0 (2025-11-21), DOI: https://doi.org/10.5281/zenodo.17666778. License not yet clarified; architecture and parameter reference only.

### 4.5 Expectations After Improvements

Two benefits can be extrapolated directly, and their values have already been given in 4.3.2 and 4.3.3 and are not repeated here: one from the window length and one from the per-window center frequency. Neither can be implemented directly as a switch of the current configuration—the latter uses future information and can only serve as a target—and the two cannot simply be added together; in order, the window length should be fixed first and center-frequency tracking verified afterward, because the window length determines the frequency resolution while the center frequency determines the calibration point, and the two affect each other.

The mechanism of amplitude gating differs from the previous two: it does not change the error of the estimator itself but changes the denominator of the statistics, and the amplitude dependence is given in 4.3.5. Restricting the statistics to periods with clear slow-wave structure gives reason to expect the hit rate to converge toward the 70% or more reported in the literature, at the cost of a decline in the number of available events; coverage and event rate must therefore be reported at the same time, otherwise the improvement in the hit rate would come at the cost of an invisible loss of coverage. With respect to temporal accuracy, the current median absolute time error is 0.1587 s (see 4.2.1); after the estimator phase offset, the scheduling prediction horizon, and the acoustic output latency are each compensated once, the portion attributable to systematic latency is expected to be removed, and the magnitude of the latency handling in public implementations is given in 4.3.5.

All of the above are conditional expectations, whose validity presupposes that the window-length and center-frequency scheme is settled, that amplitude gating is implemented as pre-registered, and that latency compensation is calibrated on the device, and that they are confirmed on data independent of the development set.

### 4.6 Chapter Summary

The current implementation can already be called and runs stably with an ample time margin, but its phase accuracy remains markedly below the publicly reported deployable level. The sources of error have been clearly localized: frequency drift within a record is first, the selection bias from the coupling between window length and frequency grid is second, and the absence of amplitude, state, and artifact gating together with uncompensated system latency jointly depress the hit rate. The improvement path is equally clear: decouple the window length from the frequency grid and bracket the optimum from both sides within 1-2.5 s, introduce per-window center-frequency tracking, supply gating and latency compensation, and confirm polarity and channel mapping on the host side. The order of execution and the decision points for this work are described in later chapters.

---

## Chapter 5 Optimization Roadmap: Parameter Tuning and Supplementary Gating

### 5.1 Nature of This Chapter and Working Principles

This chapter addresses what the algorithm side will do next and in what order; it does not cover error diagnosis or comparison with the field (see Chapter 4), nor productization and delivery (see Chapter 6). All subsequent work follows three principles. First, any newly added stage may use only the current and past samples, and must not use statistics computed over a whole record, a whole night, or across records. Second, thresholds and criteria must be pre-registered before implementation and must not be adjusted after the results are seen. Third, phase estimation and eligibility determination (that is, the various forms of gating) must be implemented and evaluated separately, so as to avoid conflating phase error with data-quality problems in attribution.

### 5.2 Optimization Items with Quantified Benefits and Their Order of Execution

#### 5.2.1 Step One: Re-sweep the Window Length and Settle It in a Single Pass

The short-window advantage under a fixed grid and the cause of its boundary behavior have been recorded in 4.3.3 and 4.3.4 respectively: the short window consistently outperforms 8 s within the interval, but the optimum falls on the lower search boundary, indicating that the sweep range did not enclose the true optimum, while the frequency resolution and the slow-wave period together confine the reasonable interval to 1-2.5 s. The first step is therefore not to shorten the window further but to re-sweep, under a fixed frequency grid, an interval that brackets the expected optimum from both sides, with a suggested sweep range of 0.75-6 s; if the optimum after the re-sweep still lies close to one end, the edge must first be extended in that direction before the decision is made, and a truncated value must not be accepted. The selection criterion is applied in a pre-declared order of priority: subject-level mean absolute error, q95 and the catastrophic-point rate, the circular standard deviation and the resultant vector length, the hit rate of the worst target, and the consistency of the direction of improvement across subjects; it must not select a point solely on the mean of the pooled sample.

#### 5.2.2 Step Two: Introduce Per-Window Center-Frequency Tracking

The current configuration uses a fixed calibration center frequency, whereas the error is strongly correlated with the absolute value of "the instantaneous frequency deviating from the center frequency." Section 4.3.2 has quantified this relationship and shown that the benefit comes from tracking frequency drift within a record rather than from differences in center frequency between individuals, and it has also given the error upper bound attainable with an ideal per-window center frequency. On this basis, the second step implements a center-frequency estimator that uses past samples only, and specifies its smoothing duration, update rate, initialization behavior, and re-acquisition semantics after loss of lock. The analysis of ecHT shows that center-frequency mismatch together with window length, bandwidth, and filter order determines the residual endpoint bias, and that residual leakage sets an irreducible lower bound on the phase error; the goal of this estimator is therefore to keep the actual center frequency within a range where the mismatch is negligible, rather than to pursue an exact reproduction of the instantaneous frequency [R5].

> References: [R5] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). DOI: https://doi.org/10.48550/arXiv.2601.13962

#### 5.2.3 Step Three: Joint Review of Window Length and Center Frequency

The window length determines the frequency resolution and the center frequency determines the calibration point, and the two affect each other: a shorter window adapts better to frequency drift but has a weaker ability to resolve the band in the frequency domain, while the calibration in turn relies on a stable center frequency. After the first two steps have each been settled, a joint review must be carried out under the same grid and the same evaluation measure, to confirm that the two benefits add rather than cancel each other out, and that no reversal in direction occurs across subjects. The outcome of the review will serve as the new baseline for all subsequent metrics.

### 5.3 Gating and Quality Stages To Be Added

#### 5.3.1 Amplitude Gating

The current implementation counts all ready endpoints and applies no amplitude gating at all, so the denominator includes many low-amplitude, non-slow-wave, and artifact periods; the influence of amplitude on the error and its magnitude are given in 4.3.5. This stage therefore adds amplitude gating inside the algorithm, rather than leaving it to the host-side eligibility determination. There are three constraints in implementation: the threshold must be shared across the past and recomputable, and whole-record or whole-night percentiles are prohibited; gating and phase estimation must be evaluated separately, reporting the estimator error over all samples and the hit rate of the gated events respectively; and once gating is online, the decline in the number of available events must be reported at the same time, so that an improvement in the hit rate does not come at the cost of an invisible loss of coverage.

#### 5.3.2 State Gating

The function of state gating is to restrict phase tracking to the state of interest to the user. It is an optional stage in a research setting: if later users need to restrict phase tracking to a specific neural state, it is established; if not, it may be left unset for the time being. If enabled, it should be placed after the state criterion and before phase tracking, and it must be considered together with the warm-up strategy of the phase loop, because it will interrupt the continuous input when the gate flips.

One offline state-gating asset already exists in this project:

- A temporal gating strategy validated on 78 subjects, using a "two of the last three" hysteresis, with an entry threshold of 0.86 and an exit threshold of 0.65, balanced accuracy 0.9505, deep-sleep recall 0.911, false-positive rate 0.0099, and a mean gating delay of about 37 s (median 30 s).

#### 5.3.3 Artifact Gating

The function of artifact gating is to prevent contaminated periods from entering the phase statistics and the trigger determination. At least four forms must be covered: flat lines or near-saturation, usually corresponding to electrode detachment or amplifier clipping; abrupt large-amplitude movement and electromyographic contamination, manifested as a transient rise in broadband high-frequency energy; slow baseline drift, usually arising from electrode polarization; and sudden changes in the amplitude or derivative distribution. The criteria should be interpretable, recomputable morphological conditions rather than a statistical black box, and must likewise be past-only.

Public deployment-level implementations offer directly usable approaches: protecting trigger quality with an amplitude range, the quadrant consistency of the phase and its quadrature component, a high-to-low frequency energy ratio, and a back-off after triggering [R14]; highly cited reviews also note that fixed thresholds are sensitive to the individual and to the montage, so the threshold must be individualized or calibrated per night, and a signal-state check independent of the phase estimate is required [R1]. One epoch-level offline artifact-gating asset already exists in this project, whose criterion logic can be reused in implementation:

- Exclude epochs flagged as artifact from the phase statistics and the trigger determination, and record the reason for exclusion and the exclusion proportion in a log that is updated synchronously in the host GUI.

Implementation must also handle the interaction with the phase loop: prolonged artifact destabilizes the endpoint estimate, so explicit rules for re-acquisition and warm-up are required between gate closure and reopening, in order to avoid emitting phase before the loop has recovered.

> References: [R1] Esfahani MJ, Farboud S, Ngo HVV, Schneider J, Weber FD, Talamini LM, Dresler M. *Closed-loop auditory stimulation of sleep slow oscillations: Basic principles and best practices*. Neuroscience & Biobehavioral Reviews 153, 105379 (2023). DOI: https://doi.org/10.1016/j.neubiorev.2023.105379 ; [R14] `gmilab/ptas_benchmarks` (TWave), pinned commit `566dec18441d4234cf9218e4771d84c8e7fbad4c`, Zenodo v1.0.0 (2025-11-21), DOI: https://doi.org/10.5281/zenodo.17666778. License not yet clarified; architecture and parameter reference only.

### 5.4 Completing the Timing and Stimulation Chain

#### 5.4.1 Latency Calibration and Compensation

The time position of a phase estimate is the last input sample of the sliding window, while the event time and the decision time are two distinct quantities; at 0.8 Hz, an uncompensated latency of about 100 ms is equivalent to a systematic phase error of about 29°, and the magnitude of how public implementations handle this is given in 4.3.5. The three compensations—estimator phase offset, scheduling prediction horizon, and acoustic output latency—are all currently zero, so this stage must complete the measurement and calibration of these three latencies, and compensate each exactly once by agreement, in order to avoid double compensation.

#### 5.4.2 Settling the Refractory

The current 0.5 s refractory is a placeholder value from the development period. Its physical constraint comes from the period of the slow wave itself (about 0.8-1.5 s): there should be no repeated triggering within the same cycle, while continuous triggering across cycles is limited by the target phase and the event density. When settling it, the change in the number of triggerable events and in the hit rate must be reported together, because tightening the refractory improves the quality of an individual trigger but reduces coverage.

---

## Chapter 6 Productization: Requirements Review, Configurability, and Delivery Acceptance

### 6.1 Scope and Positioning of This Chapter

This chapter addresses what remains to be added for it to be delivered as a component of research software; it does not cover algorithm accuracy optimization (see Chapter 5) or error diagnosis (see Chapter 4). The chapter is aligned item by item against the FRs and NFRs of Chapter 1 as its skeleton, giving the current status, the gap, and the way it will be closed.

### 6.2 Item-by-Item Comparison of Product Requirements and Current Status

| ID | Requirement | Current status | Gap and action to close it |
|---|---|---|---|
| FR-1 | Arbitrary tracking band | The band participates in kernel construction and cannot be switched at runtime | A switching path and switching semantics must be defined; see 6.3.1 |
| FR-2 | Arbitrary target phase | Implemented, with atomic updates at runtime | No action needed; the interface must provide an input entry |
| FR-3 | Functionality stable after a parameter change | Stable for target changes; band changes undefined | The output semantics, warm-up duration, and failure fallback during switching must be specified |
| FR-4 | Extension to 8 channels | 8 independent instances already run; channel identity not established | Channel mapping and per-channel parameter management are needed; see 6.4 |
| FR-5 | Phase and events observable | Phase and events already exposed; presentation layer absent | See 6.5 |
| NFR-1 | 250 Hz sampling, other rates must be resampled while preserving causality | Fixed at 250 Hz; non-250 Hz handled as a failure | The upstream resampling stage and the boundary of responsibility must be specified; see 6.3.2 |
| NFR-2 | Integrated into the host and following the host contract | Implemented, and passing on-device runs and the startup self-check | Isolation, timeout, and unloading behavior must be confirmed with the software side and documented |
| NFR-3 | Accuracy reaching the publicly reported deployable level for comparable systems | Not yet reached; currently at development-set level (see 4.2.1) | Complete the gating and optimization of Chapter 5, then re-validate after pre-registering thresholds per 6.6 |
| NFR-4 | Callback duration far below the interval, with no deadline miss | Measured on the device at 0.4%-0.7% of the callback interval (see 4.2.2) | Long-run, concurrency, and memory behavior still require re-validation |
| NFR-5 | Reproducibility: fixed parameters + version identifier + startup self-check | All three are present | No action needed; after a change, the artifact must be regenerated and recorded |
| NFR-6 | GUI adapted to present phase, parameters, and status | Not implemented | Collaboration with the software-engineering side is needed; see 6.5 |

It can be seen from this table that, of the 11 requirements, 4 are met or essentially met (FR-2, NFR-2, NFR-4, NFR-5), 2 are partially met (FR-3, FR-5), and 5 are not yet met (FR-1, FR-4, NFR-1, NFR-3, NFR-6). The unmet portion is concentrated in two places: the configurability of parameters and multichannel operation, and presentation and acceptance; the first two are elaborated in 6.3 and 6.4.

### 6.3 Parameter Configurability

#### 6.3.1 Switching the Tracking Band

The reason the band cannot be changed as directly as the target phase is that it participates in the construction of the estimator. The behavior of the upstream implementation is as follows: in the fit stage, based on the band, the filter order and type, the sampling rate, and n_fft, it first evaluates the band-pass response on an n_fft-point frequency grid and then combines it with the analytic-signal multiplier to obtain the analytic-signal coefficients and their cosine term; the complex-gain calibration of the endpoint likewise depends on the band, order, window length, and center frequency [R5][R6][R10]. "The user changes the band" is therefore mathematically equivalent to "swapping in a different endpoint operator" rather than changing a writable parameter. The following four options are arranged from least to most invasive.

Option One, configuration-time switching: the band is determined when the algorithm is loaded or the session begins, and only the target phase may be modified at runtime. It is entirely consistent with the host's lifecycle of "parameters passed in at algorithm load," requires no new state machine, and introduces no additional risk; the cost is that the user cannot change the band during acquisition.

Option Two, runtime reconstruction: the switching request takes effect at a block or sample boundary, reconstructing the kernel, clearing the dynamic state, and re-warming-up, with the output invalid during the switch and ready restored once warm-up completes. It retains all the mathematics of configuration-time switching and requires only that the switching semantics and failure fallback be additionally defined; the cost is the reconstruction time and a no-output period as long as the window length, both of which require measurement.

Option Three, kernel library: at startup, kernels are pre-built and cached for a number of commonly used bands, and switching only replaces a read-only reference and clears the state. Its precedent comes from a public deployment-level implementation—covering the entire 0.6-2.0 Hz range with a fixed set of 20 wavelets rather than supporting an arbitrarily continuously tunable band [R14]. The cost is memory and startup time growing linearly with the number of bands, and the coverage being of discrete bands.

Option Four, replacing the estimator itself: if "an arbitrary band, continuously tunable at runtime" becomes a hard requirement, then an estimator family whose parameters can adapt at runtime should be used instead, for example a phase-locked loop that works recursively with an online-trackable center frequency, or an AR forward-prediction method that re-estimates its parameters per window [R2][R8]. The cost of this route is abandoning consistency with the currently frozen cecHT; the endpoint semantics, calibration, fault semantics, and all performance conclusions must be re-established, and it should be established as a new algorithm project rather than as a switch of the existing component.

The recommended main path is Option One plus Option Two: configuration-time switching as the default capability, and runtime reconstruction as an optional capability with its switching semantics made explicit; Option Three as a performance optimization to reduce waiting time; and Option Four initiated only when the requirement is upgraded to a hard one. The academic constraint that must be emphasized is that changing the band changes both the endpoint operator and the calibration gain, so every supported band must be validated independently, and conclusions at 0.5-1.25 Hz cannot be extrapolated to other bands; the support list and the validation matrix should be released as a pair.

> References: [R2] Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007 ; [R5] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). DOI: https://doi.org/10.48550/arXiv.2601.13962 ; [R6] Osmers E. `eikeosmers/cecHT`, pinned commit `e079c3115f202a4272cf64143bf33ddddc247778`, BSD-3-Clause. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778 ; [R8] `OpitzLab/CL-phase`: real-time phase detection for closed-loop EEG-TMS (MATLAB). https://github.com/OpitzLab/CL-phase ; [R10] SciPy signal processing API: Butterworth/SOS, `filtfilt`, Hilbert transform. https://docs.scipy.org/doc/scipy/reference/signal.html ; [R14] `gmilab/ptas_benchmarks` (TWave), pinned commit `566dec18441d4234cf9218e4771d84c8e7fbad4c`, Zenodo v1.0.0 (2025-11-21), DOI: https://doi.org/10.5281/zenodo.17666778. License not yet clarified; architecture and parameter reference only.

#### 6.3.2 Sampling-Rate Strategy

The main product path acquires at 250 Hz, which is the only rate that has been validated. The band, window length, frequency grid, and update grid are mutually bound at construction time, so the algorithm performs no internal resampling; if existing data at other sampling rates must be connected in the future, the upstream side should first resample to 250 Hz before passing it to the algorithm, while preserving causality. If support for other sampling rates natively is genuinely needed later, it is equivalent to adding a supported configuration and must be re-validated by the same standard as in 6.3.1.

### 6.4 Multichannel Extension

The current host behavior is to create an independent instance per channel; the 8 instances each hold complete state and output phase independently (for the facts see 3.6, and for the on-device measurements see 4.2.2). Ideally, robust phase tracking should be achieved on all 8 instances:

Option One, keeping an independent instance per channel. It preserves fault isolation, so an anomaly in one channel does not affect the others, and it has the lowest implementation complexity and validation cost, requiring only that the same parameters be configured for each instance on extension; the cost is that the channel-identity problem must be solved on the host side.

Option Two, a multichannel core sharing one kernel: packing several channels into a matrix and obtaining the endpoints of all channels in a single matrix multiplication. It can reduce interpreter-layer overhead, but it trades channel-level isolation for batch-level isolation, so an invalid input in any one channel affects the output of the whole batch; given that a single callback currently occupies less than 1% of the callback budget, the benefit of this optimization is limited, and it is recommended for evaluation only if performance becomes a bottleneck.

Option Three, running on a subset of channels: running phase estimation only on the channels of interest to the user, with the remaining channels passing through unchanged per the host contract. It facilitates a phased rollout and also reduces total overhead, at the cost of introducing a channel-selection configuration and requiring the output semantics of disabled channels to be specified.

Option Four, exploiting cross-channel information: using multichannel consistency for artifact determination, or using bipolar and re-referenced montages to improve the signal-to-noise ratio. This is a realistic direction for improving signal quality, but it changes the input definition and the meaning of the phase, and it must re-validate the endpoint semantics and accuracy, with channel mapping and time-base synchronization as prerequisites. The public implementations in the reference set are mostly single-channel and emphasize that thresholds are highly correlated with the montage, so a multichannel gain in this project is a hypothesis to be validated rather than a feature that can be enabled by default [R1][R14].

The recommended path is Option One plus Option Three: first complete the channel-identity mapping and per-channel parameter management, so that the user can specify a subset of channels; keep Option Two as a performance fallback, and establish Option Four as a separate experimental project. Two engineering details also require attention in implementation: channels with identical parameters can share the same read-only kernel to save memory and startup time; and the per-channel target, refractory, and gating parameters must each be managed independently and must not share mutable state.

> References: [R1] Esfahani MJ, Farboud S, Ngo HVV, Schneider J, Weber FD, Talamini LM, Dresler M. *Closed-loop auditory stimulation of sleep slow oscillations: Basic principles and best practices*. Neuroscience & Biobehavioral Reviews 153, 105379 (2023). DOI: https://doi.org/10.1016/j.neubiorev.2023.105379 ; [R14] `gmilab/ptas_benchmarks` (TWave), pinned commit `566dec18441d4234cf9218e4771d84c8e7fbad4c`, Zenodo v1.0.0 (2025-11-21), DOI: https://doi.org/10.5281/zenodo.17666778. License not yet clarified; architecture and parameter reference only.

### 6.5 Observability of Phase and State

The host contract requires `process` to return a waveform of the same shape as its input, so the phase cannot be exposed through the return value and can only take a side channel; the current two side-channel means (an in-memory ring buffer and a sidecar log) are described in 3.6. Both are already available; what is missing is the presentation layer.

The presentation layer requires at least six kinds of information: the real-time phase together with its timestamp, the instantaneous frequency, the valid and ready states, the event count and the time of the most recent event, the state of each gate, and the per-callback duration and fault count. As a division of labor, it is recommended that the algorithm be responsible only for exposing and recording, while the host or interface is responsible for reading and plotting, maintaining the principle that "the presentation layer is read-only"—display logic should not modify algorithm state in reverse, or both diagnosis and reproduction would lose their meaning.

### 6.6 Acceptance, Release, and Collaboration

On acceptance, what is currently missing is not metrics but thresholds: the measures of the four main metrics are already fixed (the estimator's circular statistics and mean absolute wrapped error, and the trigger hit rate and time error), but "how much counts as passing" has not been pre-registered and must be determined by the project owner, and must be written down before the results are seen. The validation data must be independent of the development set, must statistically include at least subject-level bootstrap intervals and Wilson intervals for the hit rate, and must be reported stratified by amplitude, frequency, and worst target. For every change, a record of the isolated run and a rollback point should be retained, so that a failure can be traced to a specific change.

On release, three foundations already exist: the build script generates a single-file plugin from the frozen source, the version identifier and verification hash are generated with the build, and a self-check against the built-in kernel fingerprint is performed on load. The release evidence is recommended to consist of four parts: the hash of the build artifact, the results of the startup self-check and local validation, the on-device run record (including sampling rate, channel count, execution-time distribution, and fault count), and a list of known limitations setting out the items not yet closed from 4.2.2.

On-device re-validation must be redone once all the items not yet closed listed in 4.2.2 have been confirmed, and must confirm the host sampling rate and parameters together with the reset behavior when the algorithm is disabled and when the sampling rate is switched. Fault injection should also be included in the scope of re-validation, covering thrown exceptions, wrong input shape or type, illegal values, and timeout situations, to confirm that the isolation behavior and the alerting path match host expectations.

On collaboration, the boundaries of responsibility among the three parties should be made explicit: the algorithm side is responsible for phase estimation, the gating criteria, and the acceptance report; the software-engineering side is responsible for host behavior, interface presentation, exception isolation and alerting, and the deployment process; and the signal-quality and clinical side is responsible for determining thresholds and rejection criteria and for the subject inclusion and exclusion criteria. Any change in parameters or semantics should follow the same discipline: re-validate, update the contract, update the deployment artifact—none of the three may be omitted.

### 6.7 Chapter Summary

At the product-requirement level, the internal exposure of phase and events, host integration, the time margin, and reproducibility are already in place, with the gaps concentrated in four items: band configurability, multichannel operation, the presentation layer, and accuracy acceptance. At the technical level, the feasible path for band switching is "configuration-time switching as the main route, runtime reconstruction as the auxiliary," and the feasible path for multichannel operation is "independent instances as the main route, a channel subset as the auxiliary"; neither requires rewriting the estimator itself, whereas a continuously tunable band constitutes an architectural replacement and should be established as a separate project. At the delivery level, what most needs to be added at present is the pre-registered acceptance threshold, the independent validation data, and the on-device re-validation process; these determine whether this component can be recognized as deployable rather than merely runnable.
