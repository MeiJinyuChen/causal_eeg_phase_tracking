# cecHT P2b fixed-FFT-grid diagnostic

- development nights: 20
- segments processed: 60
- candidate windows [s]: [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
- n_fft grid: [2000, 4000] (fixed; n_fft > N is zero padding)
- zero padding: zero-padding does not increase true spectral resolution and is used only to isolate FFT-grid discretization from temporal-window length
- canonical anchor (8 s, n_fft = 2000) reproduced MAE = 1.041248 rad
- self-checks: event parity PASS (max |dt| = 2.84e-14 s), adapter delay = 8 samples

## Candidates

| n_fft | window [s] | N | zero-pad | cycles | ready | pooled MAE | subj MAE | q95 | R | circ SD | hit30 | hit45 | catastr. | events | timing [s] | dMAE(pooled) | dMAE(subj) | imp/wors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 0.75 | 188 | 1812 | 0.60 | 2688780 | 0.8414 | 0.8404 | 2.4849 | 0.5874 | 1.0315 | 0.3604 | 0.5146 | 0.0613 | 10460 | 0.122311 | -0.1999 | -0.2029 | 13/0 |
| 2000 | 1 | 250 | 1750 | 0.80 | 2685060 | 0.9345 | 0.9341 | 2.6030 | 0.5301 | 1.1267 | 0.2919 | 0.4368 | 0.0751 | 10182 | 0.148288 | -0.1068 | -0.1092 | 13/0 |
| 2000 | 1.25 | 312 | 1688 | 1.00 | 2681340 | 1.0027 | 1.0030 | 2.6772 | 0.4828 | 1.2068 | 0.2687 | 0.4020 | 0.0875 | 9784 | 0.161533 | -0.0386 | -0.0403 | 13/0 |
| 2000 | 1.5 | 375 | 1625 | 1.20 | 2677560 | 1.0375 | 1.0386 | 2.7108 | 0.4546 | 1.2557 | 0.2604 | 0.3918 | 0.0949 | 9822 | 0.164442 | -0.0037 | -0.0047 | 7/6 |
| 2000 | 2 | 500 | 1500 | 1.60 | 2670060 | 1.0416 | 1.0435 | 2.7256 | 0.4399 | 1.2816 | 0.2670 | 0.3950 | 0.0977 | 9572 | 0.161991 | +0.0004 | +0.0002 | 6/7 |
| 2000 | 2.5 | 625 | 1375 | 2.00 | 2662560 | 1.0355 | 1.0376 | 2.7258 | 0.4367 | 1.2873 | 0.2727 | 0.4069 | 0.0972 | 9528 | 0.157848 | -0.0057 | -0.0057 | 12/1 |
| 2000 | 8 | 2000 | 0 | 6.40 | 2580060 | 1.0412 | 1.0433 | 2.7297 | 0.4332 | 1.2936 | 0.2719 | 0.4054 | 0.0982 | 9183 | 0.158665 | n/a | n/a | n/a/n/a |
| 4000 | 0.75 | 188 | 3812 | 0.60 | 2688780 | 0.8414 | 0.8404 | 2.4849 | 0.5874 | 1.0315 | 0.3604 | 0.5146 | 0.0613 | 10460 | 0.122304 | -0.1999 | -0.2029 | 13/0 |
| 4000 | 1 | 250 | 3750 | 0.80 | 2685060 | 0.9344 | 0.9340 | 2.6030 | 0.5301 | 1.1267 | 0.2919 | 0.4369 | 0.0751 | 10182 | 0.148277 | -0.1068 | -0.1093 | 13/0 |
| 4000 | 1.25 | 312 | 3688 | 1.00 | 2681340 | 1.0026 | 1.0030 | 2.6772 | 0.4828 | 1.2068 | 0.2688 | 0.4020 | 0.0875 | 9784 | 0.161527 | -0.0386 | -0.0403 | 13/0 |
| 4000 | 1.5 | 375 | 3625 | 1.20 | 2677560 | 1.0375 | 1.0386 | 2.7107 | 0.4546 | 1.2557 | 0.2605 | 0.3917 | 0.0949 | 9821 | 0.164443 | -0.0038 | -0.0048 | 7/6 |
| 4000 | 2 | 500 | 3500 | 1.60 | 2670060 | 1.0416 | 1.0435 | 2.7255 | 0.4399 | 1.2815 | 0.2671 | 0.3949 | 0.0977 | 9572 | 0.161981 | +0.0003 | +0.0002 | 6/7 |
| 4000 | 2.5 | 625 | 3375 | 2.00 | 2662560 | 1.0355 | 1.0376 | 2.7258 | 0.4367 | 1.2872 | 0.2727 | 0.4070 | 0.0972 | 9528 | 0.157835 | -0.0058 | -0.0057 | 12/1 |
| 4000 | 8 | 2000 | 2000 | 6.40 | 2580060 | 1.0391 | 1.0411 | 2.7285 | 0.4338 | 1.2924 | 0.2737 | 0.4080 | 0.0979 | 9183 | 0.157965 | n/a | n/a | n/a/n/a |

## Ranking comparison

- P2 (n_fft = N) shared-window subject MAE: [0.8733699457156556, 0.9237403892010742, 0.9795826903109239, 1.0785072777702032, 1.0548737138898483]
- P2b n_fft = 2000 subject MAE: [0.8404468184117915, 0.9340778157513311, 1.0030463389137105, 1.038610365822041, 1.04352535593664, 1.0376424695221416]
- P2b n_fft = 4000 subject MAE: [0.8404172672865844, 0.9340395414132237, 1.0030002110226948, 1.0385605866035459, 1.0434712800583754, 1.0375889815356427]
- Spearman P2 vs P2b(2000) on shared windows: 0.8999999999999998
- Spearman P2 vs P2b(4000) on shared windows: 0.8999999999999998
- Spearman P2b(2000) vs P2b(4000): 1.0
- best fixed-grid candidate: 0.75 s (nfft2000_w0.75), boundary limited: True

## Answers

1. short window still beats 8 s under a fixed grid? {'best_window_s': 0.75, 'best_n_fft_2000_subject_mae': 0.8404468184117915, 'canonical_subject_mae': 1.0433198668061874, 'delta_subject_mae': -0.2028730483943959, 'subjects_improved': 13, 'subjects_worsened': 0, 'supported': True}
2. optimum still near 1-2 cycles? {'best_window_s': 0.75, 'best_cycles': 0.6000000000000001, 'boundary_limited': True, 'window_series_subject_mae_nfft2000': {'0.75': 0.8404468184117915, '1.0': 0.9340778157513311, '1.25': 1.0030463389137105, '1.5': 1.038610365822041, '2.0': 1.04352535593664, '2.5': 1.0376424695221416}, 'window_series_subject_mae_nfft4000': {'0.75': 0.8404172672865844, '1.0': 0.9340395414132237, '1.25': 1.0030002110226948, '1.5': 1.0385605866035459, '2.0': 1.0434712800583754, '2.5': 1.0375889815356427}}
3. relative ordering of short windows? {'nfft2000_rank_by_subject_mae_best_first': {'0.75': 1, '1.0': 2, '1.25': 3, '1.5': 5, '2.0': 6, '2.5': 4}, 'nfft4000_rank_by_subject_mae_best_first': {'0.75': 1, '1.0': 2, '1.25': 3, '1.5': 5, '2.0': 6, '2.5': 4}, 'spearman_nfft2000_vs_nfft4000': 1.0}
4. how much of the P2 gain was grid discretisation? {'per_window': {'1.0': {'p2_gain_rad': 0.16994992109053175, 'fixed_grid_gain_rad': 0.10924205105485629, 'fraction_of_p2_gain_retained': 0.6427896544692329}, '1.25': {'p2_gain_rad': 0.11957947760511323, 'fixed_grid_gain_rad': 0.040273527892476846, 'fraction_of_p2_gain_retained': 0.33679297400405056}, '1.5': {'p2_gain_rad': 0.06373717649526345, 'fixed_grid_gain_rad': 0.004709500984146464, 'fraction_of_p2_gain_retained': 0.07388938831478431}, '2.0': {'p2_gain_rad': -0.035187410964015786, 'fixed_grid_gain_rad': -0.00020548913045259631, 'fraction_of_p2_gain_retained': 0.00583984796900058}, '2.5': {'p2_gain_rad': -0.011553847083660918, 'fixed_grid_gain_rad': 0.005677397284045815, 'fraction_of_p2_gain_retained': -0.4913858771832466}}, 'spearman_p2_vs_fixed_grid_shared_windows': 0.8999999999999998, 'spearman_p2_vs_nfft4000_shared_windows': 0.8999999999999998, 'p2_report_used': 'experiment_evidence/cecht/cecht_window_sweep/cecht_report.json'}
5. can a window be selected now? {'supported': True, 'ranking_stable_across_grids': True, 'ranking_consistent_with_p2': True, 'boundary_limited': True, 'selection_valid': True}

## Decision

- decision rule: supported = best fixed-grid subject MAE beats the canonical 8 s baseline with >= 10 of 13 subjects improved; selection valid = supported AND Spearman(n_fft=2000, n_fft=4000) >= 0.7; the optimum is not declared final if it sits on the 0.75 s search boundary
- `Temporal short-window effect supported`: YES
- `P2 window selection valid`: YES
- ranking stable across FFT grids: True
- ranking consistent with P2: True
- `Further refinement required`: YES

## Isolation status
- Main code modified: NO
- Production config modified: NO
- Untouched test accessed: NO
- Commit performed: NO
- Experiment directory: experiment_evidence/cecht/cecht_fixed_fft_grid

## References
1. Osmers E, Kolossa D. Optimal Calibration of the Endpoint-corrected Hilbert Transform. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962
2. eikeosmers/cecHT pinned commit e079c3115f202a4272cf64143bf33ddddc247778 (BSD-3-Clause), phase.py. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778
3. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468
4. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007
5. Berens P. CircStat: A MATLAB Toolbox for Circular Statistics. Journal of Statistical Software 31(10) (2009). https://doi.org/10.18637/jss.v031.i10
6. SciPy signal API and scipy.stats.spearmanr. https://docs.scipy.org/doc/scipy/reference/signal.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
