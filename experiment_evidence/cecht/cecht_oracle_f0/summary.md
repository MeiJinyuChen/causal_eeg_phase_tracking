# P3 oracle f0 ablation

- fixed MAE: 1.041248 rad
- recording oracle MAE: 1.303124 rad
- local oracle MAE: 0.630592 rad
- delta MAE recording: -0.261876 rad
- delta MAE local: 0.410656 rad
- recording oracle improved subjects: 1
- local oracle improved subjects: 13
- Recording-level personalization sufficient: NO
- Dynamic f0 justified: YES

## Isolation status
- Main code modified: NO
- Production config modified: NO
- Untouched test accessed: NO
- Experiment directory: experiment_evidence/cecht/cecht_oracle_f0

## References
1. Osmers E, Kolossa D. Optimal Calibration of the Endpoint-corrected Hilbert Transform. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962
2. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007
3. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468
4. scipy.signal.welch documentation for PSD peak estimation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
