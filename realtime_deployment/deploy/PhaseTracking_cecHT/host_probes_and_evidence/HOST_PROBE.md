# host interface probe

## Purpose

`host_block_probe.py` is the canonical source for the first target-host
integration check.  It is a transparent diagnostic plugin, not the classifier
and not a stimulation gate.

The current host build is the authoritative target. Its external-algorithm
directory sits inside the installation root, shown here relative to that root:

```text
<host install root>\_internal\user_algorithms
```

Some host documentation also names a root-level `user_algorithms/` directory, but that
directory is absent in this build.  Do not keep two editable copies: deploy the
canonical file above byte-for-byte to the `_internal` directory the host reads.

## Manual host check

1. Keep acoustic output disabled and use live acquisition only.
2. In the current host build, set acquisition to 250 Hz and record host filter and
   resampler state, as required by the current deployment contract.
3. Confirm Fpz is channel 0, Cz is the hardware reference, and AFz is ground.
4. Click **信号算法 -> 重载外部算法**.
5. Select **CLAS host 接口探针（仅回放/调试，原样透传）** and apply it.
6. Run live acquisition, then switch the algorithm off or reload it so the host
   calls `reset()`.
7. Inspect
   `_internal/user_algorithms/probe_logs/interface_probe.jsonl`.

The log contains aggregate shapes, dtypes, block sizes, thread IDs, inter-call
intervals, and processing times.  It deliberately contains no raw EEG samples.
The constructor value `fs` is a host declaration, not independent proof of the
physical sample clock, units, channel identity, polarity, or timestamps.

The inspected host creates eight algorithm instances without passing a channel
index, and calls `process(data[:, ch], labels)` per channel.  The user plugin
therefore cannot stably identify channel 0 from this API alone.  The plugin runs
after the host unit converter, host filters, and optional host resampling, so
the manual check must set 100 Hz and disable host filtering before replay.

Execution-time protection is configured on the host side; its thresholds and
frame counts are not reproduced here.  Actual thread placement and timing still
need the probe run.

## Safety boundary

- `process()` returns the exact input object and performs no file I/O.
- The plugin does not filter, classify, track phase, open a gate, or issue a
  stimulus command.
- Missing timestamps and channel identity in the callback remain unresolved.
- The plugin return is a waveform-processing/display branch, not a documented
  gate/trigger side channel.
- The formal causal-v2 model is not imported because the bundled host
  Python currently has NumPy/SciPy but not scikit-learn/joblib.
