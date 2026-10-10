# offline_benchmark — offline phase-tracking comparison experiment (isolated directory)

## What this directory answers

This directory is a minimal runnable copy of the offline phase-tracking benchmark: it cuts
slow-wave segments of Sleep-EDF recordings into epochs and compares several phase-estimation
methods against the same non-causal reference phase under **offline batch** conditions.

It answers "how much can these methods achieve offline"; it does not answer "how much can an
online streaming implementation achieve". The two differ in causality, sampling rate,
normalization, and the definition of the reference phase, so their numbers cannot be
substituted for one another.

## Hard rules

1. **No online path may import this directory.** The online/streaming chain may use only
   `causal_phase_tracking\causal_cecHT\` and
   `realtime_deployment\deploy\PhaseTracking_cecHT\`; no module in this directory may be
   referenced by online code, the plugin, or the build scripts.
2. `offline_src\hilbert_phase.py` provides the **non-causal** reference phase (whole-segment
   zero-phase band-pass plus Hilbert, which uses future samples). It is only the golden
   standard for the offline comparison and must not enter the online path.
3. The `pll` and `cecHT` in this directory are **offline batch** implementations and are not
   the same code as the strictly per-sample, past-only streaming implementations in
   `causal_phase_tracking\causal_cecHT\`: `pll` normalizes amplitude by the whole-segment
   95th percentile (which uses future data), and `cecHT` recomputes the endpoint phase from a
   fixed-length history window at every candidate instant. Their parameters and numeric
   conclusions therefore cannot be transferred to one another.
4. The repository is always authoritative for the latest state. This `README.md` and `data\`
   ship with this directory.
5. This directory covers phase tracking only; it contains no sleep-stage classification, no
   GUI, and no main pipeline.

## Directory layout

| File | Purpose |
| --- | --- |
| `data\` | The Sleep-EDF record shipped with this directory (PSG + hypnogram); the default data root of the benchmark |
| `phase_method_benchmark.py` | Entry point; forwards to `offline_src\phase_benchmark\run_benchmark.main` |
| `offline_src\config.py` | Global constants: data root, channels, epoch length, bands, target phase |
| `offline_src\data_loader.py` | Discovery and reading of local EDF/EDF+/BDF/BDF+ recordings |
| `offline_src\epoching.py` | 30 s segmentation and reading of the sleep-stage annotations |
| `offline_src\hilbert_phase.py` | **Non-causal** Hilbert reference phase (golden standard) |
| `offline_src\trigger_simulator.py` | Trigger-simulation primitives (circular distance) |
| `offline_src\phase_benchmark\config.py` | Frozen benchmark configuration: method list, window length, filtering, per-method parameters |
| `offline_src\phase_benchmark\data.py` | Benchmark epoch assembly (find recordings, cut segments, concatenate) |
| `offline_src\phase_benchmark\cecHT.py` | Offline cecHT (endpoint-corrected Hilbert) implementation |
| `offline_src\phase_benchmark\pll.py` | canonical PLL core: this module is the result of a research internship in industry, so it is not included in this distribution for intellectual-property reasons (placeholder). |
| `offline_src\phase_benchmark\ARHT.py` | ARHT: autoregressive forward prediction + Hilbert |
| `offline_src\phase_benchmark\methods.py` | Method registry and per-method runners |
| `offline_src\phase_benchmark\evaluation.py` | Per-epoch evaluation loop |
| `offline_src\phase_benchmark\reporting.py` | Aggregated CSV / JSON / Markdown reports |
| `offline_src\phase_benchmark\visualization.py` | Comparison-figure output |
| `offline_src\phase_benchmark\run_benchmark.py` | Command-line main flow |
| `offline_src\phase_benchmark\sweep_pll.py` | canonical PLL parameter sweep: not included in this distribution for intellectual-property reasons (placeholder). |
| `offline_src\phase_benchmark\__init__.py` | Package marker |

## Method naming

The method ids match the delivery documents:

| method id | Meaning |
| --- | --- |
| `offline_hilbert` | Non-causal Hilbert reference phase, the golden standard of the offline comparison; not a real-time method |
| `cecHT` | Endpoint-corrected Hilbert (offline batch implementation) |
| `pll` | canonical PLL: this module is the result of a research internship in industry, so it is not included in this distribution for intellectual-property reasons; the benchmark reports it explicitly as withheld |
| `ferster_pll` | Simplified first-order PLL baseline, used only for comparison |
| `ARHT` | Autoregressive Forward-Prediction Hilbert Transform |

Two historical namings are kept on purpose so that existing experiment artifacts stay aligned:
the configuration keys and output columns of `pll` keep the `classic_pll_` prefix, and the
parameter-sweep entry point still accepts `--classic-pll-*` command-line flags.

## Dependencies

Python 3.10 or later. The dependencies are declared at the repository root:
`requirements.txt` (for a direct install) and `pyproject.toml` (for an editable install;
`setup.py` is also present for legacy installers).

```bash
python -m pip install -r requirements.txt
```

This directory uses `numpy`, `scipy`, `pandas`, `matplotlib`, and `mne`; `mne` reads the PSG
and the hypnogram under `data\`.

## Running

Run inside this directory (once the data is in place, the recordings under `data\` are
discovered automatically):

```bash
python phase_method_benchmark.py --single-subject
```

Other entry points:

```bash
python phase_method_benchmark.py --quick                              # quick multi-subject check
python phase_method_benchmark.py --mode full-preview
python phase_method_benchmark.py --single-subject --recording <path>  # use another recording
python -m offline_src.phase_benchmark.sweep_pll                       # not shipped; fails immediately
```

Note: `--single-subject` runs cecHT and the canonical PLL together. The canonical PLL is not
part of this distribution, so that row is marked explicitly as `withheld` (`failed = 1.0`,
`withheld = true`); the reason is printed to stderr and written to
`phase_method_summary.json` and to the Markdown report. The cecHT results are unaffected.

## Data

This directory ships one complete Sleep-EDF Expanded record under `data\`, so it runs out of
the box with nothing to download:

| File | Content | Size |
| --- | --- | --- |
| `SC4001E0-PSG.edf` (shipped) | The full-night PSG of that night, 100 Hz, including the `EEG Fpz-Cz` and `EEG Pz-Oz` channels | 48,338,048 B |
| `SC4001EC-Hypnogram.edf` (shipped) | The R&K sleep-stage annotations of the same night, one frame per 30 s | 4,620 B |

The data comes from PhysioNet's *Sleep-EDF Database Expanded* (record `SC4001`, which PhysioNet
marks as Open Data Commons Attribution License v1.0). To download it again or to verify the
files, the original addresses are:

```text
https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf
https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf
```

With both files under `data\`, the following command runs out of the box:

```bash
python phase_method_benchmark.py --single-subject
```

If the data is missing, the default discovery finds no usable recording. The command then prints
one error line and exits with status 1, for example:

```text
error: No usable EDF/BDF epochs found for phase benchmark. See the README section 'Data' for how to supply a recording.
```

This is the expected behavior.

### Pairing the PSG with the hypnogram

Sleep-EDF splits one night into two files whose night suffixes differ (`SC4001E0` versus
`SC4001EC`), so the pairing key is the six-character subject prefix that both share, not the
full file name. When `offline_src\data_loader.py` reads a recording, it automatically looks
for that subject's hypnogram **in the same directory** and attaches the stage annotations,
which is also what allows the benchmark to pick deep-sleep epochs:

- a pairable hypnogram is present next to the recording -> the 30 s epochs labeled
  `Sleep stage 3` / `Sleep stage 4` are selected by stage;
- no hypnogram -> it falls back to fixed-length epochs taken from the beginning of the
  recording, which are not deep-sleep segments.

Only the PSG therefore has to be handed to the code; the hypnogram must **not** be passed in
as a second recording: the directory scan skips hypnogram files, and passing one explicitly is
also skipped with a printed notice.

### Using your own data

The default data root is the project's `data\`; override it with the
`PHASE_BENCHMARK_DATA_DIR` environment variable, or name recording files directly on the
command line with `--recording` (repeatable, for several recordings). Do not rely on a path
from the original author's machine.

```bash
set PHASE_BENCHMARK_DATA_DIR=<your-recording-folder>
python phase_method_benchmark.py --single-subject
```

Note: the directory scan looks only at the **top level** of that directory and does not recurse
into subdirectories, so a recording must sit directly in that directory to take part in the
default discovery.

References:

- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. *Analysis of a sleep-dependent
  neuronal feedback loop: the slow-wave microcontinuity of the EEG.* IEEE Transactions on
  Biomedical Engineering, 2000, 47(9): 1185-1194. DOI: 10.1109/10.867928
- Goldberger AL, Amaral LAN, Glass L, et al. *PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a New Research Resource for Complex Physiologic Signals.* Circulation, 2000,
  101(23): e215-e220. DOI: 10.1161/01.CIR.101.23.e215
- PhysioNet. *Sleep-EDF Database Expanded*, v1.0.0. https://physionet.org/content/sleep-edfx/1.0.0/

## Outputs

Results are written to `outputs\phase_method_benchmark_<mode>\`:

```text
phase_method_results.csv           one summary row per method
phase_method_epoch_metrics.csv     per-epoch metrics
phase_method_trigger_metrics.csv   per-trigger event metrics
phase_method_summary.json          summary
phase_method_report.md             human-readable report
figures\                           per-method comparison figures
```

These artifacts are regenerated on every run and are not pre-shipped with this directory.

## Frozen parameters and metric definitions

The single authority for the frozen parameters (method list, window length, filter type and
order, PLL parameters, target phase, refractory, amplitude percentile) is
`offline_src\phase_benchmark\config.py`; this README does not duplicate the values.

The canonical PLL tuning parameters are not part of this distribution: `pll.py` is the result
of a research internship in industry, so it is not included in this distribution for
intellectual-property reasons, and the corresponding fields in `config.py` are therefore kept
empty (`None`). When `--single-subject` is run, that row is marked explicitly as withheld.

The metric definitions are in `offline_src\phase_benchmark\evaluation.py` and
`offline_src\trigger_simulator.py`; for the correspondence with the online-side metrics
(circular bias, resultant length, circular SD, MAE / q95, ±30° hit, Wilson binomial CI,
bootstrap CI) see
`causal_phase_tracking\causal_simulation_evaluation\evaluation_metrics.py` and
`docs_and_contracts\`.

## Relationship to the online chain

This directory is the "how much can be achieved offline" comparison experiment, whereas
`causal_phase_tracking\causal_cecHT\` is "what actually runs online". The two share the method
concepts and the phase convention, but they share neither implementation nor parameters nor
sampling rate (this directory runs the 100 Hz Sleep-EDF data, while the online chain runs at
250 Hz).
