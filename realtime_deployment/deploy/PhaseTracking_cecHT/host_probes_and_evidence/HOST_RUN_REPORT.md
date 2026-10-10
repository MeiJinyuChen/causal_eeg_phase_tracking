# cecHT plugin - first real-host run

Evidence: `cecht_phase_host_run_20260914.jsonl`
SHA-256: `689A527E5C98EC01D3DDB2E36F7785837A3744EAB49BD3D4DF6BB94E24E895FC`
Path normalization: the log's `module_path` field was rewritten from the machine-specific
install path to `user_algorithms/cecht_phase.py` (relative to the host install root)
for handover. Only that field changed; every event, number and timestamp is untouched.
Plugin: `cecht_phase.py`, plugin sha256 `3D23837373A60EF6BED42A166DDF231EA7B53EDB41879BB40FCCB94408792718`
Log span (UTC): 2026-09-14T09:13:58 -> 2026-09-14T09:15:09

## Host environment (as reported by the plugin itself)

```
python       3.10.20 (frozen distribution)
numpy        1.26.4
scipy        1.15.3
module name  user_algorithms.cecht_phase   (registered in sys.modules)
```

## Events

```
loaded          8
reset_summary   8
```

No `stream_fault`, `structural_reject` or `internal_error` events were written.

## Per-instance result

| # | attempts | calls | samples | duration [s] | ready | events | internal err | rejects | faults | degraded | self-test | mean proc [us] | max proc [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1593 | 1593 | 16891 | 67.6 | 14892 | 32 | 0 | 0 | 0 | false | PASS | 286.6 | 1.30 |
| 2 | 1593 | 1593 | 16891 | 67.6 | 14892 | 29 | 0 | 0 | 0 | false | PASS | 166.5 | 1.32 |
| 3 | 1593 | 1593 | 16891 | 67.6 | 14892 | 5 | 0 | 0 | 0 | false | PASS | 155.0 | 1.81 |
| 4 | 1593 | 1593 | 16891 | 67.6 | 14892 | 18 | 0 | 0 | 0 | false | PASS | 153.3 | 3.24 |
| 5 | 1593 | 1593 | 16891 | 67.6 | 14892 | 8 | 0 | 0 | 0 | false | PASS | 153.5 | 1.04 |
| 6 | 1593 | 1593 | 16891 | 67.6 | 14892 | 15 | 0 | 0 | 0 | false | PASS | 154.9 | 1.22 |
| 7 | 1593 | 1593 | 16891 | 67.6 | 14892 | 4 | 0 | 0 | 0 | false | PASS | 159.5 | 1.91 |
| 8 | 1593 | 1593 | 16891 | 67.6 | 14892 | 46 | 0 | 0 | 0 | false | PASS | 154.5 | 1.79 |

## Derived facts

- Mean callback interval 42.4 ms (10.6 samples per call, 1593 calls, 67.6 s of data
  per instance) -> the host really ingresses at 250 Hz.
- `samples - ready = 1999` on every instance: the first ready endpoint appears at
  sample index 1999, i.e. exactly the designed 8 s warm-up (`ready = index >= N-1`).
- All 8 instances produced valid phase endpoints (last value 1.53 - 2.73 rad) and
  fired the record-only pseudo-trigger (4 - 46 events each, 157 total).
- `attempts == calls` on every instance: every callback reached the estimator; no
  block was dropped, rejected or bypassed.
- Per-callback cost: mean 153 - 287 us = 0.36 - 0.68 % of the 42.4 ms callback
  interval; worst single call 3.24 ms = 7.7 % of the interval.
- Kernel self-test compared the in-host kernel against the development-machine
  golden with worst relative deviation 0.0 in all 8 instances, i.e. numpy 1.26.4 /
  scipy 1.15.3 on the host reproduce the validated kernel exactly.

## Still open (unchanged by this run)

1. Channel-instance -> channel mapping is not established; all 8 instances run and
   produce phase, but which instance carries Fpz-Cz is still unknown.
2. Input polarity is not verified (a flipped polarity would shift the phase by pi).
3. Input unit is not verified.
4. The phase is not rendered in the host GUI (by design; the companion app /
   host engineer work is separate).
5. The 8 s window is the frozen canonical configuration; the P2b 0.75 s candidate
   is still provisional pending the refinement experiment.
6. No real stimulation path exists - the pseudo-trigger is record-only.
