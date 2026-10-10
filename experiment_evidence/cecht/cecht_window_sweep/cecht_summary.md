# cecHT P2 short-window sweep

- development nights: 20
- segments processed: 60
- window grid [s]: [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
- frozen f0: 0.8 Hz (calibrated), filter butterworth order 2, 0.5-1.25 Hz, n_fft = N
- self-checks: PASS (max |dt| = 2.84e-14 s), adapter delay = 8 samples

## Window sweep

| window [s] | N | cycles | ready | MAE subj [rad] | q95 [rad] | R | circ SD [rad] | hit30 | hit45 | catastr. | worst hit45 | events | timing [s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 250 | 0.80 | 2685060 | 0.8734 | 2.5111 | 0.5793 | 1.0450 | 0.3052 | 0.4526 | 0.0643 | 0.4526 | 10266 | 0.141570 |
| 1.25 | 312 | 1.00 | 2681340 | 0.9237 | 2.6009 | 0.5296 | 1.1275 | 0.3157 | 0.4589 | 0.0746 | 0.4281 | 9239 | 0.142312 |
| 1.50 | 375 | 1.20 | 2677560 | 0.9796 | 2.6754 | 0.4808 | 1.2102 | 0.3132 | 0.4556 | 0.0859 | 0.4230 | 9148 | 0.145079 |
| 2.00 | 500 | 1.60 | 2670060 | 1.0785 | 2.7574 | 0.4124 | 1.3311 | 0.2588 | 0.3826 | 0.1050 | 0.3826 | 9426 | 0.167184 |
| 2.50 | 625 | 2.00 | 2662560 | 1.0549 | 2.7360 | 0.4252 | 1.3078 | 0.2655 | 0.3938 | 0.1005 | 0.3938 | 9723 | 0.164077 |
| 3.00 | 750 | 2.40 | 2655060 | 1.0283 | 2.7146 | 0.4434 | 1.2754 | 0.2766 | 0.4122 | 0.0949 | 0.4092 | 9455 | 0.156671 |
| 4.00 | 1000 | 3.20 | 2640060 | 1.0421 | 2.7299 | 0.4298 | 1.2996 | 0.2753 | 0.4110 | 0.0987 | 0.4107 | 9424 | 0.156465 |
| 6.00 | 1500 | 4.80 | 2610060 | 1.0440 | 2.7319 | 0.4330 | 1.2939 | 0.2695 | 0.4059 | 0.0985 | 0.4044 | 9301 | 0.160960 |
| 8.00 | 2000 | 6.40 | 2580060 | 1.0433 | 2.7297 | 0.4332 | 1.2936 | 0.2719 | 0.4054 | 0.0982 | 0.4021 | 9183 | 0.158665 |

## Baseline (8 s) vs recommended candidate

- baseline 8 s: subject MAE 1.043320, q95 2.729671, R 0.433153, hit45 0.405423, catastrophic 0.098248
- recommended window: 1.0 s, subject MAE 0.873370, q95 2.511090, R 0.579250, hit45 0.452562, catastrophic 0.064254
- subjects improved / worsened vs 8 s: 13 / 0 (of 13)
- composite rule picks 1.0 s; MAE-only rule picks 1.0 s (agree: True)

## Answers to the P2 questions

1. optimal region within 2% of best subject MAE: {'best_window_s': 1.0, 'region_window_s': [1.0], 'contiguous': True}
2. best short (<= 3 s) window vs 8 s: {'best_short_window_s': 1.0, 'best_short_subject_mae': 0.8733699457156556, 'baseline_8s_subject_mae': 1.0433198668061874, 'improved': True}
3. simultaneous improvement: {'best_window_s': 1.0, 'per_metric': {'mae': {'baseline_8s': 1.0412478545106456, 'best_window': 0.8740877350574555, 'delta': -0.1671601194531901, 'improved': True}, 'q95': {'baseline_8s': 2.7296709081737407, 'best_window': 2.511090221579469, 'delta': -0.21858068659427188, 'improved': True}, 'resultant_length': {'baseline_8s': 0.4331526311657848, 'best_window': 0.5792503644041946, 'delta': 0.14609773323840985, 'improved': True}, 'circular_sd': {'baseline_8s': 1.2935726621828545, 'best_window': 1.0450076426258978, 'delta': -0.24856501955695665, 'improved': True}, 'hit_30_default': {'baseline_8s': 0.27191549602526405, 'best_window': 0.3051821546853692, 'delta': 0.03326665866010514, 'improved': True}, 'hit_45_default': {'baseline_8s': 0.4054230643580529, 'best_window': 0.4525618546658874, 'delta': 0.047138790307834466, 'improved': True}, 'catastrophic_rate': {'baseline_8s': 0.09824771516941466, 'best_window': 0.06425368520628962, 'delta': -0.03399402996312505, 'improved': True}}, 'all_metrics_improved': True}
4. cross-subject consistency: {'improved_counts': {'1.0': 13, '1.25': 13, '1.5': 13, '2.0': 0, '2.5': 0, '3.0': 13, '4.0': 8, '6.0': 5, '8.0': 0}, 'best_window_s': 1.0, 'subjects_improved': 13, 'subjects_worsened': 0, 'subject_count': 13}
5. U-shape evidence: {'subject_mae_interior_minimum': False, 'default_target_hit45_interior_maximum': True, 'window_series_subject_mae': {'1.0': 0.8733699457156556, '1.25': 0.9237403892010742, '1.5': 0.9795826903109239, '2.0': 1.0785072777702032, '2.5': 1.0548737138898483, '3.0': 1.0283131418167253, '4.0': 1.0420835036100302, '6.0': 1.0440414168800203, '8.0': 1.0433198668061874}, 'window_series_hit45_default': {'1.0': 0.4525618546658874, '1.25': 0.458924125987661, '1.5': 0.45561871447310887, '2.0': 0.3825588796944621, '2.5': 0.3938084953203744, '3.0': 0.41216287678477, '4.0': 0.41097198641765703, '6.0': 0.4058703365229545, '8.0': 0.4054230643580529}, 'ready_sample_count': {'1.0': 2685060, '1.25': 2681340, '1.5': 2677560, '2.0': 2670060, '2.5': 2662560, '3.0': 2655060, '4.0': 2640060, '6.0': 2610060, '8.0': 2580060}, 'cycles_axis': {'1.0': 0.8, '1.25': 1.0, '1.5': 1.2000000000000002, '2.0': 1.6, '2.5': 2.0, '3.0': 2.4000000000000004, '4.0': 3.2, '6.0': 4.800000000000001, '8.0': 6.4}}

## Canonical baseline

- window 8.0 s, f0 0.8 Hz, MAE 1.041248 rad
- provenance: reproduced independently by P1, P3 and this P2 harness (MAE = 1.041248 rad)
- all deltas in this report are relative to this canonical baseline
- historical P0 artifact: 0.976569 rad, P0 runner subtracted the adapter group delay twice (segment slice offset plus index offset), shifting the offline reference by 8 samples = 32 ms (provenance only, not an active baseline)

## Delta vs canonical 8 s baseline

| window [s] | dMAE | dsubjMAE | dq95 | dR | dSD | d hit30 | d hit45 | d catastr. | improved/worsened |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | -0.1672 | -0.1699 | -0.2186 | +0.1461 | -0.2486 | +0.0333 | +0.0471 | -0.0340 | 13/0 |
| 1.25 | -0.1179 | -0.1196 | -0.1288 | +0.0964 | -0.1660 | +0.0438 | +0.0535 | -0.0237 | 13/0 |
| 1.50 | -0.0627 | -0.0637 | -0.0543 | +0.0477 | -0.0834 | +0.0413 | +0.0502 | -0.0123 | 13/0 |
| 2.00 | +0.0354 | +0.0352 | +0.0277 | -0.0208 | +0.0375 | -0.0132 | -0.0229 | +0.0067 | 0/13 |
| 2.50 | +0.0115 | +0.0116 | +0.0063 | -0.0079 | +0.0142 | -0.0065 | -0.0116 | +0.0022 | 0/13 |
| 3.00 | -0.0152 | -0.0150 | -0.0151 | +0.0102 | -0.0182 | +0.0047 | +0.0067 | -0.0034 | 13/0 |
| 4.00 | -0.0012 | -0.0012 | +0.0003 | -0.0034 | +0.0060 | +0.0033 | +0.0055 | +0.0004 | 8/5 |
| 6.00 | +0.0007 | +0.0007 | +0.0022 | -0.0002 | +0.0004 | -0.0024 | +0.0004 | +0.0003 | 5/8 |
| 8.00 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0/0 |

## FFT-grid diagnostics

| window [s] | N | n_fft | bin spacing [Hz] | nearest bin to f0 [Hz] | mismatch [Hz] | in-band bins | cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 250 | 250 | 1.0000 | 1.0000 | 0.2000 | 1 | 0.80 |
| 1.25 | 312 | 312 | 0.8013 | 0.8013 | 0.0013 | 1 | 1.00 |
| 1.50 | 375 | 375 | 0.6667 | 0.6667 | 0.1333 | 1 | 1.20 |
| 2.00 | 500 | 500 | 0.5000 | 1.0000 | 0.2000 | 2 | 1.60 |
| 2.50 | 625 | 625 | 0.4000 | 0.8000 | 0.0000 | 2 | 2.00 |
| 3.00 | 750 | 750 | 0.3333 | 0.6667 | 0.1333 | 2 | 2.40 |
| 4.00 | 1000 | 1000 | 0.2500 | 0.7500 | 0.0500 | 3 | 3.20 |
| 6.00 | 1500 | 1500 | 0.1667 | 0.8333 | 0.0333 | 5 | 4.80 |
| 8.00 | 2000 | 2000 | 0.1250 | 0.7500 | 0.0500 | 6 | 6.40 |

Spearman (n = 9, descriptive):

| predictor | vs MAE | vs q95 | vs R | vs hit45 |
| --- | --- | --- | --- | --- |
| window_sec | +0.567 | +0.533 | -0.517 | -0.533 |
| fft_bin_mismatch_hz | -0.203 | -0.203 | +0.177 | +0.034 |
| fft_bin_spacing_hz | -0.567 | -0.533 | +0.517 | +0.533 |
| in_band_bin_count | +0.638 | +0.604 | -0.587 | -0.656 |

### FFT-grid confound questions

1. peak at f0-to-bin-aligned windows? {'best_window_s': 1.0, 'best_window_mismatch_hz': 0.19999999999999996, 'closest_mismatch_window_s': 2.5, 'closest_mismatch_window_mae_rad': 1.0527213590685949, 'supported': False, 'evidence': 'the best-MAE window carries the largest f0-to-bin mismatch, so the peak does not sit at bin-aligned windows'}
2. correlation with `fft_bin_mismatch_hz`? {'spearman': {'mean_abs_wrapped_error_rad': {'spearman_rho': -0.2025478734167333, 'p_value': 0.6012206131484142}, 'q95_abs_wrapped_error_rad': {'spearman_rho': -0.2025478734167333, 'p_value': 0.6012206131484142}, 'resultant_length': {'spearman_rho': 0.17722938923964165, 'p_value': 0.648268702562637}, 'subject_mean_mae_rad': {'spearman_rho': -0.2025478734167333, 'p_value': 0.6012206131484142}, 'hit_45': {'spearman_rho': 0.03375797890278888, 'p_value': 0.9312940863813554}}, 'strong_relationship': False}
3. window-curve shape? {'direction_changes': {'mean_abs_wrapped_error_rad': 3, 'q95_abs_wrapped_error_rad': 3, 'resultant_length': 3, 'subject_mean_mae_rad': 3, 'hit_45': 3}, 'subject_mean_mae': [0.8733699457156556, 0.9237403892010742, 0.9795826903109239, 1.0785072777702032, 1.0548737138898483, 1.0283131418167253, 1.0420835036100302, 1.0440414168800203, 1.0433198668061874], 'interior_optimum': False, 'argmin_window_s': 1.0, 'note': 'monotone degradation from 1.0 s to 2.0 s, then a plateau near 1.03-1.05 rad with a local bump at 2.0 s; the optimum sits on the shortest-window boundary of the grid, not at an interior short-optimum-long-worsening minimum'}
4. best candidate explained by the grid? {'f0_bin_alignment': False, 'grid_resolution_coupling': True, 'spearman_window_vs_in_band_bin_count': 0.9660917830792958, 'conclusion': 'the narrow f0-to-bin alignment mechanism is not supported, but the number of in-band FFT bins is a deterministic, rank-collinear function of the swept window length, so this design cannot attribute the short-window gain to temporal window length alone'}

### Decision

- decision rule: selection is allowed only if no FFT-grid diagnostic is rank-collinear with window length (|rho| >= 0.9) and no diagnostic shows a strong monotone relationship with MAE/q95/R (|rho| >= 0.5)
- f0-to-bin alignment supported: False
- grid-resolution coupling detected: True (Spearman window vs in-band bin count = +0.966)
- `P2 window selection valid`: NO
- `FFT-grid sensitivity diagnostic required`: YES
- observation for the record: the composite and MAE-only rules both point at 1.0 s, but that ranking is provisional while the grid confound is unresolved

### Proposed minimal P2b (not executed)

`P2b - FFT-grid decoupling diagnostic`

- fix the frequency grid at the canonical 8 s resolution: `n_fft = 2000` for every candidate window;
- sweep only the endpoint window: `N = {250, 312, 375, 500, 625, 750, 1000, 1500, 2000}`;
- keep f0, filter family/order/band, calibration, reference, ready semantics, targets, refractory and metrics frozen;
- add one control grid at the 1 s resolution (`n_fft = 250`) to measure the same windows under the coarse grid;
- decision: the temporal-window effect is identified only if the `n_fft`-fixed ranking still prefers short windows and the ranking is stable across the two grids;
- this is a minimal 2-condition x 9-window design (18 runs) that needs your approval because it requires relaxing the `n_fft = N` rule.

Note: decoupling `n_fft` from `N` is a genuine zero-padding change to
how the band-pass response is sampled, which is why it is proposed
here rather than executed: it is outside the frozen P2 rule [R1, R7].

## Isolation status
- Main code modified: NO
- Production config modified: NO
- Untouched test accessed: NO
- Experiment directory: experiment_evidence/cecht/cecht_window_sweep

## References
1. Osmers E, Kolossa D. Optimal Calibration of the Endpoint-corrected Hilbert Transform. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962
2. Schreglmann SR et al. Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence. Nature Communications 12, 363 (2021). https://doi.org/10.1038/s41467-020-20581-7
3. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468
4. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007
5. Berens P. CircStat: A MATLAB Toolbox for Circular Statistics. Journal of Statistical Software 31(10) (2009). https://doi.org/10.18637/jss.v031.i10
6. PhysioNet Sleep-EDF Expanded. https://physionet.org/content/sleep-edfx/1.0.0/
7. SciPy signal API (butter/sosfilt/hilbert/sosfreqz). https://docs.scipy.org/doc/scipy/reference/signal.html
