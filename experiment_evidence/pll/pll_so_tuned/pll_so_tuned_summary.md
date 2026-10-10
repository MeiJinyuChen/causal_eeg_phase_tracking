# PLL P0 SO-tuned baseline

- PLL MAE: 1.633864 rad
- PLL q95: 2.962425 rad
- PLL R: 0.326319
- PLL circular SD: 1.496583 rad
- PLL ±45° hit: 0.190612
- cecHT fixed MAE: 0.976569 rad
- cecHT local-oracle MAE: 0.630592 rad
- delta MAE vs cecHT fixed: -0.657295 rad
- fixed-to-oracle gap closed: -1.8998
- subjects improved vs cecHT: 0
- internal estimated frequency mean: 0.819371 Hz
- internal frequency SD: 0.508449 Hz

## Isolation status
- Main code modified: NO
- Production config modified: NO
- Untouched test accessed: NO
- Experiment directory: experiment_evidence/pll/pll_so_tuned

## References
1. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007
2. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468
3. Berens P. CircStat: A MATLAB Toolbox for Circular Statistics. J Stat Softw 31(10) (2009). https://doi.org/10.18637/jss.v031.i10
