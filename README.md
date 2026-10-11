# PhaseTracking

Causal EEG phase tracking for closed-loop sleep neuromodulation: a strictly causal 250 Hz
streaming cecHT phase estimator, the machine contracts that pin its behaviour, an offline
benchmarking harness, and a single-file plugin that runs the estimator inside a host
application. This repository contains the early-stage development of a production-oriented algorithm, originating from an independent research project conducted during my internship at BrainCo (BrainSync) (June–September 2026).
Repository map: [OVERVIEW_INDEX.txt](OVERVIEW_INDEX.txt)

## Components

| Path | Contents |
| --- | --- |
| `causal_phase_tracking/causal_cecHT/` | the streaming core (`streaming_core.py`), the pseudo-trigger, the scoped machine contract, the unit tests, and the byte-identical upstream snapshot under `vendor/` |
| `causal_phase_tracking/causal_simulation_evaluation/` | offline replay and evaluation: EDF replay, the non-causal reference phase, metrics, frozen synthetic goldens and manifests, and their tests |
| `realtime_deployment/` | the project-level machine contract and the host plugin: wrapper, build script, generated artifact, verification script, deployment manual, host probes and run evidence |
| `offline_benchmark/` | an isolated offline comparison of phase-estimation methods; it ships the recording it uses, so it runs as delivered |
| `experiment_evidence/` | the runner and the output of every offline experiment |
| `docs_and_contracts/` | the overview, the experiment contract, and the method contract |

## Quick start

```bash
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. The results are what this repository
produces on Python 3.13.

| Command | Expected result |
| --- | --- |
| `python -m pytest -q` | `184 passed, 1 skipped` |
| `cd offline_benchmark && python phase_method_benchmark.py --single-subject` | cecHT: 76 triggers, `accuracy_within_pi_4 = 0.894737`; the `pll` row is reported as withheld |
| `python realtime_deployment/deploy/PhaseTracking_cecHT/build_PhaseTracking_cecHT.py --check` | `[ok] on-disk plugin matches a fresh build` |
| `python realtime_deployment/deploy/PhaseTracking_cecHT/verify_PhaseTracking_cecHT.py` | `17/17 checks passed` |

`offline_benchmark` downloads nothing: the Sleep-EDF record it uses is already in
`offline_benchmark/data/`.

The runners under `experiment_evidence/` are the exception. They evaluate a fixed
development set, so they need the 20 Sleep-EDF Expanded nights `SC4011E0` to `SC4102E0`
(PSG plus hypnogram, about 0.9 GB from PhysioNet), and both environment variables have to
point at the folder holding them:

```bash
export PHASE_BENCHMARK_DATA_DIR=<folder with the Sleep-EDF nights>   # builds the manifest
export CLAS_SLEEP_EDF_ROOT=<same folder>                             # loads the PSG files
python experiment_evidence/cecht/cecht_baseline/run_baseline.py
```

(On Windows use `set VAR=value` instead of `export`.) With only the shipped record these
runners find no eligible night and stop with `ValueError: need at least one array to
concatenate`; the two runners under `experiment_evidence/pll/` stop earlier and report
that the PLL module is withheld.

## Withheld content and third-party sources

- **The canonical classic PLL is not distributed.** This  module is withheld for intellectual-property reasons. It
  appears as a placeholder that raises a clear error, as empty (`None`) configuration
  fields, and as a `withheld` row in the benchmark output. Everything downstream reports
  that state explicitly instead of inventing numbers.
- **Vendored upstream.** `causal_phase_tracking/causal_cecHT/vendor/eikeosmers_cecht/` is
  a byte-identical snapshot of the upstream cecHT `phase.py` (BSD-3-Clause, commit
  `e079c311...`, sha256 recorded in `UPSTREAM_PROVENANCE.json` and re-checked by the test
  suite).
- **Data.** `offline_benchmark/data/SC4001E0-PSG.edf` and its hypnogram come from
  PhysioNet's *Sleep-EDF Database Expanded* (record `SC4001`, Open Data Commons
  Attribution License v1.0).
- **References.** The compared baselines follow Santostasi 2016 (classic PLL), Ferster
  2022 (real-time benchmarking), Opitz-style autoregressive prediction, and Berens'
  CircStat. Other references are noted in documents. 

## Engineering invariants

- The streaming core is strictly causal and per-sample. No estimator reads future samples.
- The contracts are machine-readable JSON and the test suite pins their fields, so a
  contract edit that the tests do not reflect fails the build.
- The plugin is generated. The build inlines the frozen sources,
  records their sha256 in its header, and `--check` re-derives the artifact byte for byte.
- `.gitattributes` pins the line endings of the artifacts whose hashes are recorded.
- No machine-specific absolute path is kept in the repository.


