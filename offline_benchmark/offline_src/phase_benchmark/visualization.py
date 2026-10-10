from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _phase_hist_edges(bin_width: float = np.deg2rad(10.0)) -> np.ndarray:
    return np.arange(-np.pi - bin_width / 2, np.pi + bin_width, bin_width)


def _bar(results: pd.DataFrame, value: str, title: str, ylabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = results.sort_values(value, ascending=False)
    ax.bar(ordered["method"], ordered[value], color="tab:blue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_accuracy_by_method(results: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = results.sort_values("accuracy_within_pi_4", ascending=False)
    x = np.arange(len(ordered))
    width = 0.25
    ax.bar(x - width, ordered["accuracy_within_pi_6"], width=width, label="+/- pi/6")
    ax.bar(x, ordered["accuracy_within_pi_4"], width=width, label="+/- pi/4")
    ax.bar(x + width, ordered["accuracy_within_pi_3"], width=width, label="+/- pi/3")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["method"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Phase Targeting Accuracy by Method")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_phase_error_by_method(trigger_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if trigger_df.empty:
        ax.text(0.5, 0.5, "No triggers", ha="center", va="center")
    else:
        methods = sorted(trigger_df["method"].unique())
        data = [trigger_df.loc[trigger_df["method"] == method, "abs_phase_error"] for method in methods]
        positions = np.arange(1, len(methods) + 1)
        ax.boxplot(data, positions=positions, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Absolute phase error (rad)")
    ax.set_title("Phase Error by Method")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_resultant_length_by_method(results: pd.DataFrame, save_path: Path) -> None:
    _bar(results, "circular_resultant_length", "Circular Resultant Length by Method", "R", save_path)


def plot_trigger_rate_by_method(results: pd.DataFrame, save_path: Path) -> None:
    _bar(results, "trigger_rate_per_min", "Trigger Rate by Method", "Triggers/min", save_path)


def plot_polar_hist_by_method(trigger_df: pd.DataFrame, save_path: Path) -> None:
    methods = sorted(trigger_df["method"].unique()) if not trigger_df.empty else []
    n_methods = max(1, len(methods))
    fig = plt.figure(figsize=(4 * n_methods, 4))
    bins = _phase_hist_edges()
    if not methods:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No triggers", ha="center", va="center")
        ax.axis("off")
    for idx, method in enumerate(methods, start=1):
        ax = fig.add_subplot(1, n_methods, idx, projection="polar")
        phases = trigger_df.loc[trigger_df["method"] == method, "true_trigger_phase"].to_numpy()
        ax.hist(phases, bins=bins)
        ax.set_title(method)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_circular_phase_error_hist_by_method(
    trigger_df: pd.DataFrame,
    target_phase: float,
    save_path: Path,
) -> None:
    methods = sorted(trigger_df["method"].unique()) if not trigger_df.empty else []
    n_methods = max(1, len(methods))
    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(4.2 * n_methods, 4.2),
        squeeze=False,
        sharey=True,
    )
    bins = _phase_hist_edges()

    if not methods:
        ax = axes[0, 0]
        ax.text(0.5, 0.5, "No triggers", ha="center", va="center")
        ax.axis("off")

    for ax, method in zip(axes[0], methods):
        stim_phase = trigger_df.loc[trigger_df["method"] == method, "true_trigger_phase"].to_numpy(dtype=float)
        err = np.angle(np.exp(1j * (stim_phase - target_phase)))
        ax.hist(err, bins=bins, color="tab:blue", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", label="target")
        ax.axvline(-np.pi / 4, color="tab:green", linewidth=1.0, linestyle=":", label="+/- pi/4")
        ax.axvline(np.pi / 4, color="tab:green", linewidth=1.0, linestyle=":")
        ax.set_title(method)
        ax.set_xlabel("Circular phase error (rad)")
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Trigger count")
    if methods:
        axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Circular Phase Error Histogram (target_phase={target_phase:.3f} rad)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_all_figures(
    results: pd.DataFrame,
    trigger_df: pd.DataFrame,
    figures_dir: Path,
    target_phase: float = 0.0,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_by_method(results, figures_dir / "accuracy_by_method.png")
    plot_phase_error_by_method(trigger_df, figures_dir / "phase_error_by_method.png")
    plot_resultant_length_by_method(results, figures_dir / "resultant_length_by_method.png")
    plot_trigger_rate_by_method(results, figures_dir / "trigger_rate_by_method.png")
    plot_polar_hist_by_method(trigger_df, figures_dir / "polar_hist_by_method.png")
    plot_circular_phase_error_hist_by_method(
        trigger_df,
        target_phase,
        figures_dir / "circular_phase_error_hist_by_method.png",
    )
