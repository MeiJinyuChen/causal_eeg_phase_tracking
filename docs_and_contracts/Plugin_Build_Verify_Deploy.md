# cecHT host plugin (`PhaseTracking_cecHT.py`)

Single-file host-side external algorithm that runs the frozen causal cecHT phase
estimator on the live 250 Hz stream.  It estimates phase and runs the
pseudo-trigger; it **never commands stimulation**.

## Frozen configuration

| item | value |
| --- | --- |
| sampling rate | 250 Hz (any other rate degrades to pass-through) |
| window | 8 s (2000 samples), `n_fft = N`, `fft_mode = 'exact'` |
| band / filter | 0.5-1.25 Hz, Butterworth order 2 |
| calibration | calibrated, `f0 = 0.8 Hz` |
| pseudo-trigger | target 0 rad, refractory 0.5 s, record-only |
| max block | 1024 samples (host blocks observed at 9-15, one 144) |

This table is the quick reference for what is shipped.  The authoritative
definition of every parameter, together with the deployed file hashes and the
release record, lives in the project-level contract
`realtime_deployment/contracts/phase_tracking.json` under
`host_plugin_release`; this file does not restate it.

## Files

| file | role |
| --- | --- |
| `PhaseTracking_cecHT_wrapper.py` | hand-written wrapper (the only edited source) |
| `build_PhaseTracking_cecHT.py` | generates `PhaseTracking_cecHT.py` from the frozen sources + wrapper |
| `PhaseTracking_cecHT.py` | **generated** deployable artifact (do not edit) |
| `verify_PhaseTracking_cecHT.py` | local verification harness (17 checks) |

The builder inlines, in order, `vendor/eikeosmers_cecht/phase.py` (BSD-3),
`streaming_core.py` and `pseudo_trigger.py`, strips the two relative imports and
embeds the development-machine kernel fingerprint (`GOLDEN_KERNEL`).  The build
aborts if those relative imports are not found, so a refactor cannot silently
produce a wrong plugin.

## Build / verify / deploy

```powershell
cd ...\realtime_deployment\deploy\PhaseTracking_cecHT
python build_PhaseTracking_cecHT.py            # regenerate PhaseTracking_cecHT.py
python build_PhaseTracking_cecHT.py --check    # fail if the on-disk plugin is stale
python verify_PhaseTracking_cecHT.py           # 17 checks, exit 0 on success
```

Deploy by copying `PhaseTracking_cecHT.py` into
`...\<host install root>\_internal\user_algorithms\`, then in the host:
signal algorithm -> reload external algorithms -> select the plugin -> apply,
with the host sampling rate set to 250 Hz and stimulation disabled.

## Verification status (latest real run on the development machine)

```
17/17 checks passed
per-sample phase vs frozen package : max|dphase| = 0.000e+00 rad (bit-identical)
ready flags / pseudo-events        : identical
self-test vs GOLDEN_KERNEL         : PASS (worst relative deviation 0.0)
strided block                      : bare core rejects it; plugin stages and runs
process() cost                     : mean 88.3 us, max 345.3 us per callback
                                     ~= 0.21% of the host's ~41.5 ms callback budget
```

## Implementation notes

- The host passes strided `data[:, ch]` views and the streaming core rejects
  non-contiguous input, so the plugin stages every block into a pre-allocated
  C-contiguous `float64` buffer before calling the core.  This staging step is
  the reason an unmodified core produces nothing under this host.
- Everything else about the shipped behaviour (same-shape passthrough, where the
  estimated phase is exposed, the ring-buffer length, the warm-up length) is part
  of the release record in the project-level contract, not repeated here.

## Changing the frozen configuration later

Edit the constants at the top of `PhaseTracking_cecHT_wrapper.py`, rebuild, and re-run
`verify_PhaseTracking_cecHT.py`.  The build recomputes `GOLDEN_KERNEL` from the frozen
sources, so the self-test always reflects the configuration actually shipped.
(The window is not final: P2b left 0.75 s as a provisional value pending
refinement, so treat any window change as a new, unverified configuration.)
