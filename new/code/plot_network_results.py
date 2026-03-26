"""
Plotting module for the network-structure experiment.

Generates Figures 1-5 from the CSVs produced by network_experiment.py.

Usage:
    uv run python code/plot_network_results.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Consistent style across all figures ─────────────────────────────────────

NETWORK_ORDER = ["erdos_renyi", "barabasi_albert", "sbm", "facebook"]
NETWORK_LABELS = {
    "erdos_renyi":    "Erdős-Rényi",
    "barabasi_albert": "Barabási-Albert",
    "sbm":            "Stochastic Block Model",
    "facebook":       "Facebook (real)",
}
PALETTE = {
    "erdos_renyi":    "#2196F3",   # blue
    "barabasi_albert": "#F44336",  # red
    "sbm":            "#4CAF50",   # green
    "facebook":       "#FF9800",   # orange
}

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Helper ───────────────────────────────────────────────────────────────────

def _ci95(series):
    """Return (lower_bound, upper_bound) 95% CI for a Series."""
    n = len(series)
    if n < 2:
        m = float(series.mean())
        return m, m
    m = float(series.mean())
    se = float(scipy_stats.sem(series, nan_policy="omit"))
    h = se * scipy_stats.t.ppf(0.975, df=n - 1)
    return m - h, m + h


def _rolling_smooth(series, window=10):
    """Apply a rolling mean for smoothing noisy time series."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _plot_time_series(ax, df_comparison, metric, ylabel, title, smooth_window=10):
    """Plot metric over rounds for each network type with 95% CI shading.

    Args:
        ax: matplotlib Axes.
        df_comparison: Per-round comparison DataFrame.
        metric: Column name to plot.
        ylabel: Y-axis label.
        title: Subplot title.
        smooth_window: Rolling window size for smoothing the mean line.
    """
    for nt in NETWORK_ORDER:
        sub = df_comparison[df_comparison["network_type"] == nt]
        agg = sub.groupby("round")[metric].agg(["mean", _ci95])
        rounds = agg.index.values
        mean_vals = agg["mean"].values
        ci_lower = np.array([v[0] for v in agg["_ci95"]])
        ci_upper = np.array([v[1] for v in agg["_ci95"]])

        # Smooth mean line
        smoothed = _rolling_smooth(pd.Series(mean_vals), window=smooth_window).values

        color = PALETTE[nt]
        label = NETWORK_LABELS[nt]
        ax.plot(rounds, smoothed, color=color, linewidth=2, label=label)
        ax.fill_between(rounds, ci_lower, ci_upper, color=color, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)


# ── Figure 1: MCR over rounds ─────────────────────────────────────────────

def fig1_mcr_over_rounds(df_comparison, save_path):
    """MCR over rounds, one line per network type (averaged over simulations).

    Shows which topology allows misinformation to persist over time.

    Args:
        df_comparison: DataFrame from results_network_comparison.csv.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_time_series(
        ax, df_comparison,
        metric="mcr",
        ylabel="Misinformation Contamination Rate (MCR)",
        title="Figure 1 — MCR over Rounds by Network Type",
    )
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved → {save_path}")


# ── Figure 2: False Reach over rounds ────────────────────────────────────

def fig2_false_reach_over_rounds(df_comparison, save_path):
    """False Reach (absolute count) over rounds by network type.

    Shows absolute exposure to false content across different topologies.

    Args:
        df_comparison: DataFrame from results_network_comparison.csv.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_time_series(
        ax, df_comparison,
        metric="false_reach_count",
        ylabel="False Reach (# agents exposed to false message)",
        title="Figure 2 — False Reach over Rounds by Network Type",
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved → {save_path}")


# ── Figure 3: Average Reputation over rounds ─────────────────────────────

def fig3_reputation_over_rounds(df_comparison, save_path):
    """Average reputation over rounds by network type.

    Shows how reputation evolves differently under each topology,
    reflecting how effectively the reputation mechanism is exercised.

    Args:
        df_comparison: DataFrame from results_network_comparison.csv.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_time_series(
        ax, df_comparison,
        metric="average_reputation",
        ylabel="Average Agent Reputation",
        title="Figure 3 — Average Reputation over Rounds by Network Type",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved → {save_path}")


# ── Figure 4: MCR vs p_reveal ─────────────────────────────────────────────

def fig4_mcr_vs_preveal(df_sweep, save_path):
    """MCR vs p_reveal interaction plot, one line per network type.

    Shows whether fact-checking effectiveness varies by network structure.

    Args:
        df_sweep: DataFrame from results_network_x_factchecking.csv.
        save_path: Path to save the figure.
    """
    # Compute mean MCR per (network_type, p_reveal, sim_run) first,
    # then aggregate across sim_runs for CI
    sim_means = (
        df_sweep
        .groupby(["network_type", "p_reveal", "sim_run"])["mcr"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for nt in NETWORK_ORDER:
        sub = sim_means[sim_means["network_type"] == nt]
        agg = sub.groupby("p_reveal")["mcr"].agg(["mean", _ci95])
        p_values = agg.index.values
        means = agg["mean"].values
        ci_lower = np.array([v[0] for v in agg["_ci95"]])
        ci_upper = np.array([v[1] for v in agg["_ci95"]])

        color = PALETTE[nt]
        label = NETWORK_LABELS[nt]
        ax.plot(p_values, means, marker="o", color=color, linewidth=2, label=label)
        ax.fill_between(p_values, ci_lower, ci_upper, color=color, alpha=0.15)

    ax.set_xlabel("Fact-checking Probability (p_reveal)")
    ax.set_ylabel("Mean MCR (averaged over rounds & simulations)")
    ax.set_title("Figure 4 — MCR vs Fact-checking Intensity by Network Type",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks([0.0, 0.2, 0.5, 0.8, 1.0])
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved → {save_path}")


# ── Figure 5: Structural diagnostics scatter ──────────────────────────────

def fig5_structural_scatter(diag_df, df_comparison, save_path):
    """Scatter plots of structural properties vs misinformation outcomes.

    Panel A: clustering coefficient vs mean MCR.
    Panel B: average path length vs mean False Reach.

    Args:
        diag_df: DataFrame from network_diagnostics.csv.
        df_comparison: DataFrame from results_network_comparison.csv.
        save_path: Path to save the figure.
    """
    # Aggregate comparison results to per-network summary
    summary = (
        df_comparison
        .groupby("network_type")
        .agg(
            mean_mcr=("mcr", "mean"),
            mean_false_reach=("false_reach_count", "mean"),
        )
        .reset_index()
    )
    merged = diag_df.merge(summary, on="network_type")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 5 — Network Structure vs Misinformation Outcomes",
                 fontweight="bold", fontsize=13)

    labels = [NETWORK_LABELS[nt] for nt in merged["network_type"]]
    colors = [PALETTE[nt] for nt in merged["network_type"]]

    # Panel A: clustering vs mean MCR
    ax = axes[0]
    for i, row in merged.iterrows():
        ax.scatter(row["avg_clustering"], row["mean_mcr"],
                   color=PALETTE[row["network_type"]], s=150, zorder=3)
        ax.annotate(NETWORK_LABELS[row["network_type"]],
                    (row["avg_clustering"], row["mean_mcr"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Average Clustering Coefficient")
    ax.set_ylabel("Mean MCR")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Clustering vs MCR", fontweight="bold")

    # Panel B: path length vs mean False Reach
    ax = axes[1]
    for i, row in merged.iterrows():
        ax.scatter(row["avg_path_length"], row["mean_false_reach"],
                   color=PALETTE[row["network_type"]], s=150, zorder=3)
        ax.annotate(NETWORK_LABELS[row["network_type"]],
                    (row["avg_path_length"], row["mean_false_reach"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Average Shortest Path Length")
    ax.set_ylabel("Mean False Reach (agents)")
    ax.set_title("Path Length vs False Reach", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 5 saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    """Load result CSVs and generate all five figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    comparison_path = OUTPUT_DIR / "results_network_comparison.csv"
    sweep_path = OUTPUT_DIR / "results_network_x_factchecking.csv"
    diag_path = OUTPUT_DIR / "network_diagnostics.csv"

    for p in [comparison_path, sweep_path, diag_path]:
        if not p.exists():
            print(f"ERROR: missing file {p}")
            print("Run network_experiment.py first to generate results.")
            sys.exit(1)

    print("Loading results...")
    df_comparison = pd.read_csv(comparison_path)
    df_sweep = pd.read_csv(sweep_path)
    diag_df = pd.read_csv(diag_path)

    print(f"  Comparison: {len(df_comparison):,} rows")
    print(f"  Sweep:      {len(df_sweep):,} rows")

    # Generate figures
    fig1_mcr_over_rounds(
        df_comparison,
        save_path=FIGURES_DIR / "fig1_mcr_over_rounds.png",
    )
    fig2_false_reach_over_rounds(
        df_comparison,
        save_path=FIGURES_DIR / "fig2_false_reach_over_rounds.png",
    )
    fig3_reputation_over_rounds(
        df_comparison,
        save_path=FIGURES_DIR / "fig3_reputation_over_rounds.png",
    )
    fig4_mcr_vs_preveal(
        df_sweep,
        save_path=FIGURES_DIR / "fig4_mcr_vs_preveal.png",
    )
    fig5_structural_scatter(
        diag_df,
        df_comparison,
        save_path=FIGURES_DIR / "fig5_structural_scatter.png",
    )

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
