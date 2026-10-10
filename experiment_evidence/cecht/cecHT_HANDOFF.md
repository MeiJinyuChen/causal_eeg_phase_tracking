# cecHT HANDOFF

Cross-stage contract and decision log for the cecHT experiment program.  Each
stage is named after the question it answers (baseline, error stratification,
window length, FFT grid, oracle f0).  Results stay in each experiment directory;
this file keeps only the stable contract, the canonical baseline, the decisions
and the references that were actually used.

Last updated: 2026-09-18 (wording only; every decision below dates from 2026-09-14)

---

## A. Scope

- cecHT (endpoint-corrected Hilbert transform, causal streaming form) is the
  current **main-line** phase-tracking algorithm for the project.
- The canonical PLL baseline lives in a separate, isolated space
  (`experiment_evidence/pll/`) and must not be mixed into this file or into cecHT
  experiment directories.
- Every stage below is **experiment-only**: no main-code edits, no production
  config edits, no frozen-manifest edits, no untouched-test access, no commit
  or merge without explicit approval.
- Naming: cecHT experiment directories live under `experiment_evidence/cecht/`,
  named `cecht_p<stage>_<topic>`.

| Stage (as used in this file) | Directory |
| --- | --- |
| original baseline run | `experiment_evidence/cecht/cecht_baseline/` |
| error-stratification study | `experiment_evidence/cecht/cecht_stratification/` |
| window-length sweep | `experiment_evidence/cecht/cecht_window_sweep/` |
| fixed-grid window check | `experiment_evidence/cecht/cecht_fixed_fft_grid/` |
| oracle-f0 ablation | `experiment_evidence/cecht/cecht_oracle_f0/` |

The short `p0`-style tags survive only inside those directory names; this file
refers to every stage by what it tests.

---

## B. Frozen evaluation contract (shared by every stage)

| Item | Frozen definition |
| --- | --- |
| Data source | PhysioNet Sleep-EDF Expanded, Fpz-Cz, native 100 Hz |
| Development set | `build_manifest()` = 20 nights / 13 subjects; 3 segments per night, 180 s each; Stage 3 + Stage 4 merged as N3; pre-registered rule of at least 3 Stage 4 candidates per night |
| Test separation | Untouched nights / spare nights are never opened; parameter choice is development-only |
| Offline reference | non-causal `resample_poly(5, 2)` 100 -> 250 Hz, then zero-phase Butterworth order 2, 0.5-1.25 Hz `sosfiltfilt`, then Hilbert phase |
| Reference alignment | `ref_phase[round(seg_start_s * 250) - ADAPTER_DELAY + output_index]`, i.e. the adapter group delay (8 samples) is subtracted **exactly once** |
| Phase convention | `error = wrap(phi_estimate - phi_reference)`; bias/R/circular SD follow circular statistics |
| Ready endpoint | `output_index >= window_samples - 1` |
| Pseudo-event | positive crossing of the target by the estimated phase, linear interpolation of the crossing time, refractory 0.5 s |
| Targets | 8 targets at pi/4 spacing: `0, pi/4, pi/2, 3pi/4, pi, -3pi/4, -pi/2, -pi/4` |
| Estimator config | f0 = 0.8 Hz, `calibrate=True`, Butterworth order 2, band 0.5-1.25 Hz, `n_fft = N`, `fft_mode='exact'` |
| Aggregation | pooled + per-subject + per-night + subject-level mean |
| Uncertainty | subject-level bootstrap 95% CI (n_boot = 1000, seed = 20260914); Wilson CI for hit-45 |
| Metrics | circular bias, resultant length R, circular SD, mean / median / q95 absolute wrapped error, +-30 / +-45 deg hit, catastrophic rate (> 135 deg), median absolute timing error, ready-sample count, pseudo-event count |

### B.1 Alignment provenance (why the contract says "once")

`Causal100To250Adapter` documents that it "only reports its exact group delay,
which the offline reference and event-time layer subtract once"
(`group_delay_output_samples = (33 - 1) / (2 * 2) = 8`).  `run_evaluation.py`
therefore indexes the full-night reference as `stream_index - ADAPTER_DELAY`.

The error-stratification study, the oracle-f0 ablation and the window-length
sweep follow that rule.  The **original baseline runner did not**: it built a
segment reference starting at `base - ADAPTER_DELAY` *and* indexed it with
`output_index - ADAPTER_DELAY`, subtracting the delay twice and shifting the
offline reference by 8 samples (32 ms at 250 Hz).

---

## C. Canonical baseline

```
Canonical cecHT baseline = corrected 8 s configuration, reproduced by the
error-stratification study, the oracle-f0 ablation and the window-length sweep.
All subsequent comparisons must use this baseline.
```

| Field | Value |
| --- | --- |
| window | 8.0 s (2000 samples) |
| f0 | 0.8 Hz (calibrated) |
| Pooled MAE | **1.041248 rad** |
| Subject-level mean MAE | 1.043320 rad (bootstrap 95% CI 1.0085 - 1.0778) |
| Median absolute error | 0.829305 rad |
| q95 absolute error | 2.729671 rad |
| Circular bias | -0.294547 rad |
| Resultant length R | 0.433153 |
| Circular SD | 1.293573 rad |
| +-30 / +-45 deg hit (default target) | 0.333629 / 0.478447 |
| Catastrophic rate (> 135 deg) | 0.098248 |
| Median absolute timing error | 0.1587 s |
| Pseudo-event count (default target) | 9183 |
| Ready samples | 2 580 060 |

Reproduction evidence (three independent implementations, same protocol):

| Source | Pooled MAE (rad) | Artifact |
| --- | --- | --- |
| Error-stratification study | 1.041248 | `experiment_evidence/cecht/cecht_stratification/stratification_report.json` |
| Oracle-f0 ablation (fixed branch) | 1.041248 | `experiment_evidence/cecht/cecht_oracle_f0/oracle_report.json` |
| Window-length sweep (8 s candidate) | 1.0412479 | `experiment_evidence/cecht/cecht_window_sweep/cecht_report.json` |
| Fixed-grid window check (8 s anchor, `n_fft = 2000`) | 1.0412479 | `experiment_evidence/cecht/cecht_fixed_fft_grid/cecht_report.json` |

The fixed-grid runner re-derives the anchor in-harness and aborts if it disagrees
with 1.041248 rad by more than 5e-7, so the canonical baseline is verified on
every full fixed-grid run.

### C.1 Historical baseline artifact (provenance only)

`experiment_evidence/cecht/cecht_baseline/baseline_complete_metrics.json` reports
MAE 0.976569 rad for the same nominal 8 s configuration.  It is the double
delay subtraction described in B.1, not a better estimator.  It is kept for
provenance and **must not** be used as a baseline, and must not appear in any
delta or promotion decision.

---

## D. Diagnostic findings

### D.1 Error stratification (why the error is large)

- Frequency mismatch `|f_inst,ref - f0|` is the dominant error source.
- Low reference analytic amplitude is a secondary driver of dispersion and of
  the catastrophic tail.
- Within-window frequency variability (nonstationarity) is a smaller but real
  factor, which is what motivated the window-length sweep.
- Effects are pooled and subject-level consistent; no amplitude gate,
  artifact gate or recalibration was introduced.

### D.2 Oracle-f0 ablation (is fixed 0.8 Hz the bottleneck?)

- Recording-level oracle f0: MAE 1.303124 rad, i.e. **worse** than fixed 0.8 Hz
  (1.041248), improving only 1 of 13 subjects.
- Local (per-window) oracle f0: MAE 0.630592 rad, improving 13 of 13 subjects.
- Conclusion: the bottleneck is **within-recording frequency drift**, not
  night-to-night centre-frequency differences, so subject/night
  personalisation is not sufficient and dynamic f0 is justified.
- Both oracle branches are diagnostic only and can never enter runtime.

### D.3 Window-length sweep (is the 8 s window too long?)

Sweep: `window_sec = {1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0}` with
`n_fft = N`, everything else frozen.  Artifacts:
`experiment_evidence/cecht/cecht_window_sweep/`.

| window [s] | N | cycles at f0 | ready | pooled MAE | subj MAE | q95 | R | circ SD | hit30 | hit45 | catastr. | events | timing [s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 250 | 0.80 | 2 685 060 | 0.874088 | 0.873370 | 2.511090 | 0.579250 | 1.045008 | 0.407296 | 0.570266 | 0.064254 | 10266 | 0.1416 |
| 1.25 | 312 | 1.00 | 2 681 340 | 0.923342 | 0.923740 | 2.600891 | 0.529577 | 1.127543 | 0.387742 | 0.543003 | 0.074578 | 9239 | 0.1423 |
| 1.50 | 375 | 1.20 | 2 677 560 | 0.978581 | 0.979583 | 2.675412 | 0.480811 | 1.210191 | 0.364110 | 0.512669 | 0.085947 | 9148 | 0.1451 |
| 2.00 | 500 | 1.60 | 2 670 060 | 1.076689 | 1.078507 | 2.757381 | 0.412362 | 1.331055 | 0.318294 | 0.457242 | 0.104964 | 9426 | 0.1672 |
| 2.50 | 625 | 2.00 | 2 662 560 | 1.052721 | 1.054874 | 2.736012 | 0.425221 | 1.307782 | 0.328272 | 0.471178 | 0.100450 | 9723 | 0.1641 |
| 3.00 | 750 | 2.40 | 2 655 060 | 1.026017 | 1.028313 | 2.714590 | 0.443380 | 1.275403 | 0.340002 | 0.486273 | 0.094858 | 9455 | 0.1567 |
| 4.00 | 1000 | 3.20 | 2 640 060 | 1.040035 | 1.042084 | 2.729949 | 0.429778 | 1.299605 | 0.334584 | 0.479668 | 0.098663 | 9424 | 0.1565 |
| 6.00 | 1500 | 4.80 | 2 610 060 | 1.041973 | 1.044041 | 2.731913 | 0.432954 | 1.293926 | 0.332331 | 0.477489 | 0.098538 | 9301 | 0.1610 |
| 8.00 | 2000 | 6.40 | 2 580 060 | 1.041248 | 1.043320 | 2.729671 | 0.433153 | 1.293573 | 0.333629 | 0.478447 | 0.098248 | 9183 | 0.1587 |

Best observed candidate = 1.0 s (0.8 cycles).  Relative to the canonical 8 s
baseline it improves every headline metric: MAE -0.1672 rad, q95 -0.2186 rad,
R +0.1461, circular SD -0.2486 rad, hit30 +0.0333, hit45 +0.0471,
catastrophic -0.0340, timing -0.0171 s, and 13/13 subjects improve.

**However, the sweep design is confounded and the window selection is not
valid yet.**  Because `n_fft = N` is mandated with no zero padding, the
frequency grid used to sample the band-pass response is `fs / N`, so the
number of FFT bins inside 0.5-1.25 Hz is a deterministic, rank-collinear
function of the window length (Spearman rho = +0.966 for
`window_sec` vs `in_band_bin_count`): 1 bin at 1.0-1.5 s, 2 bins at 2.0-3.0 s,
3 at 4 s, 5 at 6 s, 6 at 8 s.

Confound tests (n = 9 candidates, descriptive):

| predictor | vs MAE | vs q95 | vs R | vs hit45 |
| --- | --- | --- | --- | --- |
| `window_sec` | +0.567 | +0.533 | -0.517 | -0.533 |
| `fft_bin_mismatch_hz` (f0-to-bin) | -0.203 | -0.203 | +0.177 | +0.034 |
| `fft_bin_spacing_hz` | -0.567 | -0.533 | +0.517 | +0.533 |
| `in_band_bin_count` | +0.638 | +0.604 | -0.587 | -0.656 |

Interpretation:

- The narrow "f0 happens to land on an FFT bin" mechanism is **rejected**: the
  best window (1.0 s) carries the *largest* mismatch (0.200 Hz), while the
  exactly aligned window (2.5 s, mismatch 0.000 Hz) is mid-pack.
- The relevant grid quantity is instead **how many bins resolve the passband**
  (`in_band_bin_count`), which correlates with worse performance as it grows
  (MAE rho +0.638, p = 0.064; R rho -0.587; hit45 rho -0.656, p = 0.055).
- That quantity is rank-collinear with the swept window length, so this design
  cannot attribute the short-window gain to temporal localisation rather than
  to coarse spectral sampling of the filter response.
- The window-performance curve is also not the hypothesised smooth
  short-optimum / long-degradation shape: there are 3 direction changes, a
  local maximum at 2.0 s, and the optimum sits on the shortest grid boundary
  (0.8 cycles), i.e. no interior optimum was found.

---

### D.4 Fixed-FFT-grid decoupling (is the short-window gain temporal?)

The window-length sweep changed `window_sec` while `n_fft = N` was mandated, so
the window length and the discretisation of the band-pass response moved
together.  The fixed-grid check pins the grid and sweeps only the endpoint
window.

Design (artifacts: `experiment_evidence/cecht/cecht_fixed_fft_grid/`):

- candidates `window_sec in {0.75, 1.0, 1.25, 1.5, 2.0, 2.5}`;
- `n_fft = 2000` (main grid, `Delta f = 0.125 Hz`) and `n_fft = 4000`
  (finer grid, `Delta f = 0.0625 Hz`) as a grid-density sensitivity check;
- `n_fft > N` is zero-padding: **zero-padding does not increase true spectral
  resolution; it is used only to isolate FFT-grid discretization from
  temporal-window length**;
- anchor `window_sec = 8.0, n_fft = 2000` recomputed in-harness and required to
  reproduce canonical MAE; it did, exactly (1.0412478545 rad).

Results at `n_fft = 2000` (subject-level MAE; `n_fft = 4000` matches to ~1e-5):

| window [s] | N | cycles | subj MAE | q95 | R | hit30 (sample) | hit45 (sample) | catastr. | events | timing [s] | d subj-MAE vs 8 s | imp/wors | sweep gain retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 188 | 0.60 | **0.8404** | **2.4849** | **0.5874** | **0.4320** | **0.5916** | **0.0613** | 10460 | **0.1223** | **-0.2029** | 13/0 | not in the sweep grid |
| 1.00 | 250 | 0.80 | 0.9341 | 2.6030 | 0.5301 | 0.3793 | 0.5339 | 0.0751 | 10182 | 0.1483 | -0.1092 | 13/0 | 64.3% |
| 1.25 | 312 | 1.00 | 1.0030 | 2.6772 | 0.4828 | 0.3453 | 0.4933 | 0.0875 | 9784 | 0.1615 | -0.0403 | 13/0 | 33.7% |
| 1.50 | 375 | 1.20 | 1.0386 | 2.7108 | 0.4546 | 0.3308 | 0.4752 | 0.0949 | 9822 | 0.1644 | -0.0047 | 7/6 | 7.4% |
| 2.00 | 500 | 1.60 | 1.0435 | 2.7256 | 0.4399 | 0.3321 | 0.4760 | 0.0977 | 9572 | 0.1620 | +0.0002 | 6/7 | 0.6% |
| 2.50 | 625 | 2.00 | 1.0376 | 2.7258 | 0.4367 | 0.3365 | 0.4812 | 0.0972 | 9528 | 0.1578 | -0.0057 | 12/1 | negative |
| 8.00 (anchor) | 2000 | 6.40 | 1.0433 | 2.7297 | 0.4332 | 0.3336 | 0.4784 | 0.0982 | 9183 | 0.1587 | 0 | - | - |

Interpretation:

1. The short-window advantage **survives** a fixed grid: 0.75 s improves 13 of
   13 subjects and cuts MAE by 0.203 rad versus canonical 8 s.  The effect is
   therefore not purely an FFT-grid artefact.
2. The sweep nevertheless **over-stated the magnitude** for the windows it
   tested: only 64% of its gain at 1.0 s survives a fixed grid, 34% at 1.25 s,
   7% at 1.5 s and ~0% at 2.0 s.  For windows >= 1.5 s the improvement was
   essentially all grid discretisation.
3. Rank order is stable: Spearman(sweep, fixed grid with `n_fft = 2000`) = +0.90, and
   Spearman(`n_fft = 2000`, `n_fft = 4000`) = +1.00, so the ordering does not
   depend on grid density.
4. The fixed-grid optimum sits on the **search boundary** (0.75 s = 0.6 cycles),
   so no final window optimum may be declared; the trend is still improving
   towards shorter windows.

Decision: `Temporal short-window effect supported = YES`;
`window selection valid = YES` with a boundary caveat;
provisional window 0.75 s; `Further refinement required = YES`.

---

## E. Current optimization status

```
Executed:            original baseline (superseded), error stratification,
                     oracle-f0 ablation, window-length sweep, FFT-grid confound
                     analysis, fixed-grid window check
window selection valid:                    YES (boundary caveat)
Temporal short-window effect supported:    YES
Provisional recommended window:            0.75 s (0.6 cycles at f0 = 0.8 Hz)
Further refinement required:               YES (extend below 0.75 s)
Next candidate:                            minimal window refinement (needs approval)
After that:                                past-only dynamic-f0 study (needs approval)
```

### E.1 Fixed-grid window check (executed)

See D.4.  The `n_fft = N` rule is relaxed only inside the fixed-grid experiment
directory; no production or main-code path changed.

### E.2 Proposed minimal refinement (proposed, not executed)

- Keep the fixed-grid setting (`n_fft = 2000`) and the frozen contract unchanged.
- Extend the window grid below the current boundary, e.g.
  `window_sec = {0.4, 0.5, 0.6, 0.75, 1.0}` (~0.32 - 0.8 cycles), to establish
  whether an interior optimum exists or the estimator keeps improving as the
  window shrinks.
- Reuse the same `cecht_fixed_fft_grid` runner with a widened grid; report
  the same metrics plus latency/warm-up, because shorter windows also shorten
  the warm-up (`ready_sample_count` grows from 2 580 060 at 8 s to 2 688 780 at
  0.75 s).
- Success criterion: a clear interior minimum in subject-level MAE with stable
  cross-subject direction; otherwise the shortest stable window is retained
  with the boundary limitation recorded.

Note: a 0.4-0.6 s window is ~0.3-0.5 cycles at 0.8 Hz; the sweep must therefore
also check that the crossing-based trigger still produces a usable event rate
and that phase estimates do not become unstable.

---

## F. References actually used

**R1. Optimal Calibration of the Endpoint-corrected Hilbert Transform**
Authors: E. Osmers, D. Kolossa. Year: 2026. arXiv:2601.13962v2.
DOI: https://doi.org/10.48550/arXiv.2601.13962 ; https://arxiv.org/abs/2601.13962
Used for: endpoint / most-recent-sample semantics, deterministic complex
calibration gain, residual leakage and frequency mismatch framing; the
`n_fft` and window-length discussion behind the FFT-grid confound.

**R2. `eikeosmers/cecHT` (pinned upstream implementation)**
Author: E. Osmers. Commit `e079c3115f202a4272cf64143bf33ddddc247778`,
BSD-3-Clause. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778
Used for: the vendored mathematical core (`phase.py`: `fit`, `transform`,
`_design_bandpass`, `_calibration`) from which the constructor-derived
endpoint kernel is built.

**R3. Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence**
Authors: S. R. Schreglmann et al. Year: 2021. Nature Communications 12, 363.
DOI: https://doi.org/10.1038/s41467-020-20581-7
Used for: the ECHT method the cecHT line descends from.

**R4. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep**
Authors: M. L. Ferster et al. Year: 2022. IEEE TBME 69(9), 2916-2925.
DOI: https://doi.org/10.1109/TBME.2022.3157468
Used for: reporting structure (phase accuracy, timing accuracy and tail
errors reported separately), hit-rate conventions and the
development-versus-test separation principle.

**R5. Phase-locked loop for precisely timed acoustic stimulation during sleep**
Authors: G. Santostasi et al. Year: 2016. J Neurosci Methods 259, 101-114.
DOI: https://doi.org/10.1016/j.jneumeth.2015.11.007
Used for: separating target phase, estimator error and offline-reference
timing error, and for the fixed-versus-dynamic target framing.

**R6. CircStat: A MATLAB Toolbox for Circular Statistics**
Author: P. Berens. Year: 2009. Journal of Statistical Software 31(10).
DOI: https://doi.org/10.18637/jss.v031.i10
Used for: circular bias, resultant length and circular SD definitions.

**R7. PhysioNet Sleep-EDF Expanded**
Kemp B. et al. Year: 2000. IEEE TBME 47(9), 1185-1194.
DOI: https://doi.org/10.1109/10.867928 ; https://physionet.org/content/sleep-edfx/1.0.0/
Used for: the Fpz-Cz development recordings and their provenance.

**R8. SciPy `scipy.signal` API (butter, sosfilt, sosfiltfilt, hilbert, sosfreqz, resample_poly) and `scipy.stats.spearmanr`**
https://docs.scipy.org/doc/scipy/reference/signal.html ;
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
Used for: reference band-pass and analytic signal, ECHT filter responses,
non-causal reference resampling, and the FFT-grid confound rank correlations.

Not listed here: MEEGkit's ECHT implementation, which is upstream of R2 and
was not consulted directly during this stage.

---

## G. Decision log

| Date | Decision | Evidence | Affected stage |
| --- | --- | --- | --- |
| 2026-09-14 | Keep cecHT as the main-line algorithm | the baseline, error-stratification and oracle-f0 diagnostics localise the error to frequency mismatch, which is addressable by calibration / dynamic f0 rather than a method change | all |
| 2026-09-14 | Run the window-length sweep before the dynamic-f0 study, even though the oracle-f0 ablation already justified dynamic f0 | the error-stratification study found a nonstationarity component; if the 8 s window is the cause, fixing the window must come before adding dynamic f0, otherwise the two effects would be confounded | window-length sweep, dynamic-f0 study |
| 2026-09-14 | Do not add amplitude / artifact / SNR gating yet | the error-stratification study flagged low amplitude as a secondary driver, and gating changes coverage; estimator performance must be fixed first so gating cannot mask it | gating work |
| 2026-09-14 | Retire the original 0.976569 rad baseline | the original baseline runner subtracted the adapter group delay twice; the error-stratification study, the oracle-f0 ablation and the window-length sweep independently reproduce 1.041248 rad for the corrected convention | all |
| 2026-09-14 | Adopt `Canonical cecHT baseline = corrected 8 s configuration` reproduced by three independent harnesses (MAE 1.041248 rad) for all future deltas | three-way reproduction above | all |
| 2026-09-14 | Keep the historical baseline artifact on disk as provenance only | the artifact documents a real earlier run, but its number is not comparable | all |
| 2026-09-14 | Record that window length and FFT frequency-grid resolution are coupled under `n_fft = N` | Spearman(`window_sec`, `in_band_bin_count`) = +0.966; best-MAE window carries the largest f0-to-bin mismatch | window-length sweep, fixed-grid window check |
| 2026-09-14 | Do not promote a window from the sweep; mark `window selection valid = NO` and require an FFT-grid sensitivity diagnostic | grid-resolution coupling is rank-collinear with the swept parameter, so the short-window gain is not attributable to temporal window length | window-length sweep |
| 2026-09-14 | Propose, but do not execute, a minimal fixed-grid check that decouples `n_fft` from `N` | needs approval because it relaxes the current `n_fft = N` rule | fixed-grid window check |
| 2026-09-14 | Keep the dynamic-f0 study after the window-length sweep closes | the oracle-f0 ablation justifies it, but its benefit must be estimated against a window choice that is not itself a grid artefact | dynamic-f0 study |
| 2026-09-14 | Execute the fixed-grid window check with a fixed FFT grid (`n_fft = 2000`, sensitivity `4000`) | the sweep's grid-resolution coupling (Spearman +0.966) made the swept window inseparable from the FFT grid | fixed-grid window check |
| 2026-09-14 | Record that a real temporal short-window effect exists, but that the sweep over-stated its size | at a fixed grid 0.75 s improves 13/13 subjects (dMAE -0.203 rad), yet only 64% of the sweep gain at 1.0 s, 34% at 1.25 s and <=7% at 1.5 s survives | window-length sweep, fixed-grid window check |
| 2026-09-14 | Mark `window selection valid = YES` with a boundary caveat; keep 0.75 s provisional | the fixed-grid rank order is stable across grids (Spearman +1.00) and consistent with the sweep (+0.90), but the optimum lies on the 0.75 s search boundary | fixed-grid window check |
| 2026-09-14 | Require a minimal refinement below 0.75 s before freezing a window or starting the dynamic-f0 study | a boundary-limited optimum would otherwise propagate an arbitrary truncation into the dynamic-f0 study | refinement, dynamic-f0 study |

---

## H. Isolation status (window-length sweep and fixed-grid stages)

```
Main code modified:          NO
Production config modified:  NO
Untouched test accessed:     NO
Commit performed:            NO
```
