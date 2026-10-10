# experiment_evidence

This directory contains the experimental evidence for the phase-tracking algorithms: one self-contained subdirectory per experiment, holding both the **run script (code)** and the **output of that run (metrics, figures, conclusions)**, so that any algorithmic conclusion can be traced back to how it was reached and whether it can be reproduced.

## 1. Contents

- Included: the runners for the offline experiments, metric tables, figures, conclusion documents, and run logs.
- Not included: the algorithm implementations themselves, the host-side plugin and deployment artifacts, and the offline benchmark framework.
- This directory is the authoritative source of that evidence; each subdirectory is self-contained and holds its own runner and its own output.

## 2. Reading order

1. `cecht/cecHT_HANDOFF.md` — the cross-stage frozen contract and decision log for cecHT: dataset splits, metric definitions, bootstrap, the canonical baseline, the conclusions of each stage, and references.
2. `cecht/cecht_baseline/` — the metrics of the frozen baseline, which is the reference point for every later comparison.
3. `cecht/cecht_stratification/` — error attribution: how much error low amplitude, frequency mismatch, and frequency drift each contribute.
4. `cecht/cecht_window_sweep/` and `cecht/cecht_fixed_fft_grid/` — the window-length sweep, and the check of whether the window-length advantage is only an artifact of frequency-domain discretization. Read the two together.
5. `cecht/cecht_oracle_f0/` — the f0 upper-bound ablation: deciding whether f0 mismatch is the main fixable bottleneck.
6. `pll/` — the PLL comparison line, evaluated under the same protocol as cecHT.
7. `../docs_and_contracts/cecHT_Experiment_Contract.md` — the same source as item 1, kept in the documentation and contract package.

## 3. What each subdirectory answers

| Subdirectory | Question it answers | Runner | Main evidence |
| --- | --- | --- | --- |
| `cecht/cecht_baseline/` | What the baseline performance of the frozen configuration (8 s window, f0 = 0.8 Hz, Butterworth order 2, 0.5–1.25 Hz) is on the development set | `run_baseline.py` | `baseline_complete_metrics.json`, `baseline_subject_metrics.csv`, `baseline_target_metrics.csv`, `baseline_summary.md` |
| `cecht/cecht_stratification/` | Which signal properties the error correlates with: amplitude, frequency mismatch, or within-window non-stationarity | `run_stratification.py` | `*_strata.csv`, `interaction_matrices.json`, `phase_binned_residuals.csv`, `fig1`–`fig8` |
| `cecht/cecht_window_sweep/` | Whether the 8 s window accumulates too much frequency non-stationarity, and whether short windows of about 1–3 cycles are better | `run_window_sweep.py` | `cecht_window_metrics.csv`, `cecht_fft_grid_diagnostics.csv`, `cecht_report.json`, `fig_window_vs_*` |
| `cecht/cecht_fixed_fft_grid/` | Whether the short-window advantage still holds once the FFT grid is fixed, that is, separating a genuine time-domain effect from a frequency-domain discretization artifact | `run_fixed_fft_grid.py` | `cecht_ranking_comparison.csv`, `cecht_report.json`, `fig_ranking_comparison.png`, `fig_nfft2000_vs_4000.png` |
| `cecht/cecht_oracle_f0/` | How far the error can fall if f0 is known (per record or per window), and whether a dynamic f0 is worth implementing | `run_oracle_f0.py` | `condition_metrics.csv`, `oracle_report.json`, `summary.md` |
| `pll/pll_baseline/` | The first baseline performance of the PLL under the same evaluation protocol | `run_pll_baseline.py` | `pll_complete_metrics.json`, `pll_subject_metrics.csv`, `pll_summary.md` |
| `pll/pll_so_tuned/` | Whether the PLL improves when the parameters tuned on slow waves are used instead | `run_pll_so_tuned.py` | `pll_so_tuned_complete_metrics.json`, `pll_so_tuned_subject_metrics.csv`, `pll_so_tuned_summary.md` |

## 4. File-type conventions

| File-name pattern | Meaning |
| --- | --- |
| `run_*.py` | Experiment entry point. It only imports existing implementations; it does not modify the main code, does not modify the production configuration, and does not read the untouched test set |
| `*_metrics.csv`, `cecht_window_metrics.csv` | Pooled-level metrics, aggregated across subjects and across nights |
| `*_subject_metrics.csv`, `*_night_metrics.csv`, `*_target_metrics.csv` | Per-subject, per-night, and per-target-phase metrics |
| `*_report.json` | The complete structured report, including the self-check fields (grid check, delay check, event consistency, runner hash) |
| `summary.md`, `*_summary.md` | Human-readable conclusions and recommendations |
| `fig_*.png` | Figures supporting the conclusions |
| `run_stdout.log` | The stdout record of that formal run |

The evaluation protocol (the definition of wrapped error, circular bias, resultant length R, circular SD, the ±30° and ±45° hit rates, the catastrophic rate, the pseudo-event definition, the ready endpoint, and the bootstrap and aggregation scheme) is defined in `cecHT_HANDOFF.md` and in each runner, and is kept consistent across the experiments.

## 5. What this evidence set omits

To remove duplicated evidence, the following are omitted here:

- `cecht/cecht_nested_loso/` — an abandoned branch that produced no usable conclusion and is not used as evidence.
- `smoke_artifacts/` under each experiment — repeated runs on a reduced development subset; their values differ from the formal runs because of sample count rather than configuration, which makes them duplicated evidence.
- `*_stderr.txt` — transcripts of interpreter warnings and of the commands invoked, which are not conclusive content.

When smoke evidence is needed, it can be replayed with the switch built into the runner, for example `python run_window_sweep.py --smoke-nights 2`.

## 6. Prerequisites for rerunning

- The data path is controlled by the environment variable `CLAS_SLEEP_EDF_ROOT`: `DATA_ROOT` reads that variable first and falls back to the relative placeholder directory `physionet-sleep-data` when it is unset. Before rerunning, set it to the local PhysioNet sleep-EDF root; no code change is needed.
- The interpreter is Python 3.10 or later (see `pyproject.toml`). The dependencies are declared in the repository-root `requirements.txt`: the runners use `numpy`, `scipy` and `matplotlib` directly, and `mne` to read the hypnograms while the manifest is built.
- All experiments use only the development split and do not read the untouched test set.

## 7. Known leftovers

- Several field names in the archived JSON files keep their historical naming (`p2_report_used`, `p2_window_selection_valid`, `shared_event_definition`, and others). These are the schema keys that the runners read, and renaming them would break reproduction, so only their values were corrected: the old directory names and machine absolute paths were replaced with repository-relative paths, while the key names were left unchanged.
