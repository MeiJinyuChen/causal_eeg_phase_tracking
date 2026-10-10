from __future__ import annotations

import argparse
import sys

from offline_src.phase_benchmark.config import config_for_mode
from offline_src.phase_benchmark.data import load_benchmark_epochs
from offline_src.phase_benchmark.evaluation import evaluate_methods
from offline_src.phase_benchmark.reporting import write_report, write_summary
from offline_src.phase_benchmark.visualization import save_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick benchmark for SO phase targeting methods.")
    parser.add_argument("--quick", action="store_true", help="Run quick mode: 10 subjects x 5 N3/N4 epochs.")
    parser.add_argument(
        "--single-subject",
        action="store_true",
        help="Run focused mode: 1 subject x 20 N3/N4 epochs, cecHT vs PLL only.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "single-subject", "full-preview"],
        default=None,
        help="Benchmark mode. Defaults to quick.",
    )
    parser.add_argument(
        "--recording",
        "--edf",
        dest="recording",
        action="append",
        default=None,
        help="Local EDF/EDF+/BDF/BDF+ file. Can be repeated. --edf is kept as an alias.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_subject:
        mode = "single-subject"
    else:
        mode = "quick" if args.quick or args.mode is None else args.mode
    cfg = config_for_mode(mode)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Phase benchmark mode={cfg.mode}: "
        f"{cfg.n_subjects} subjects x {cfg.epochs_per_subject} N3/N4 epochs max; "
        f"methods={', '.join(cfg.method_names)}.",
        flush=True,
    )
    try:
        epochs = load_benchmark_epochs(
            cfg.n_subjects,
            cfg.epochs_per_subject,
            cfg.fieldtrip_buffer_sec,
            edf_paths=args.recording,
        )
    except RuntimeError as exc:
        # One actionable line instead of a traceback: this failure only means
        # that no usable recording was found.
        print(
            f"error: {exc} "
            "See the README section 'Data' for how to supply a recording.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None
    print(f"Loaded {len(epochs)} benchmark epochs.", flush=True)

    results, epoch_metrics, trigger_metrics = evaluate_methods(epochs, cfg)
    results.attrs["evaluation_edge_trim_sec"] = cfg.evaluation_edge_trim_sec
    results.attrs["fieldtrip_buffer_sec"] = cfg.fieldtrip_buffer_sec
    results.to_csv(cfg.output_dir / "phase_method_results.csv", index=False)
    epoch_metrics.to_csv(cfg.output_dir / "phase_method_epoch_metrics.csv", index=False)
    trigger_metrics.to_csv(cfg.output_dir / "phase_method_trigger_metrics.csv", index=False)

    save_all_figures(results, trigger_metrics, cfg.figures_dir, target_phase=cfg.target_phase)
    summary = write_summary(results, cfg.output_dir, cfg.mode)
    write_report(results, summary, cfg.output_dir)

    print("Phase method benchmark complete.")
    print(results.to_string(index=False))
    print(f"Outputs saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
