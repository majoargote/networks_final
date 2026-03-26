"""
Experiment runner: compare misinformation dynamics across network types.

Runs two sets of experiments:
  1. Network comparison  — 4 networks × 50 sims × 250 rounds at p_reveal=0.5
  2. Fact-checking sweep — 4 networks × 5 p_reveal values × 50 sims × 250 rounds

Results are saved as per-round CSVs in output/.

Usage:
    uv run python code/network_experiment.py
"""

import contextlib
import io
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure the project's code/ directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from snlearn.simulation import Simulation

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Agent configurations (identical to config_sim_*.json files) ────────────

AGENT_CONFIG = {
    "count": None,          # fill remaining slots
    "left_bias": 2,
    "right_bias": 2,
    "ave_reputation": 0.0,
    "variance_reputation": 1.0,
    "bias_strength": 0.5,
    "reputation_reward_strength": 0.5,
    "reputation_penalty_strength": 0.5,
    "forwarding_cost": 0.1,
}

INFLUENCER_CONFIG = {
    "count": 8,
    "left_bias": 0.5,
    "right_bias": 0.5,
    "ave_reputation": 1.0,
    "variance_reputation": 1.0,
    "bias_strength": 1.0,
    "reputation_reward_strength": 1.0,
    "reputation_penalty_strength": 0.2,
    "forwarding_cost": 0.1,
}

BOT_CONFIG = {
    "count": 8,
    "left_bias": 0.5,
    "right_bias": 0.5,
    "ave_reputation": 0.0,
    "variance_reputation": 1.0,
    "bias_strength": 1.0,
    "reputation_reward_strength": 0.0,
    "reputation_penalty_strength": 0.0,
    "forwarding_cost": 0.0,
}


def _make_sim_config(pickle_path, num_rounds, p_reveal):
    """Build a simulation config dict for the given network and p_reveal.

    Args:
        pickle_path: Absolute path to the network pickle file.
        num_rounds: Number of rounds to simulate.
        p_reveal: Probability that message truth is revealed each round.

    Returns:
        Dict suitable for passing to Simulation().
    """
    return {
        "network_pickle_file": pickle_path,
        "num_rounds": num_rounds,
        "num_initial_senders": 10,
        "message": {
            "left_bias": 0.5,
            "right_bias": 0.5,
            "prob_truth": 0.6,
            "truth_revelation_prob": p_reveal,
        },
    }


def _run_one_simulation(pickle_path, num_rounds, p_reveal, seed):
    """Run a single simulation and return per-round metrics as a list of dicts.

    Args:
        pickle_path: Path to network pickle file.
        num_rounds: Number of simulation rounds.
        p_reveal: Truth revelation probability.
        seed: numpy random seed for this run.

    Returns:
        List of dicts, one per round, with keys:
            round, reach, forwarding_rate, mcr, false_reach_count, average_reputation.
    """
    np.random.seed(seed)
    config_sim = _make_sim_config(pickle_path, num_rounds, p_reveal)
    # Suppress verbose Simulation print statements
    with contextlib.redirect_stdout(io.StringIO()):
        sim = Simulation(config_sim, AGENT_CONFIG, INFLUENCER_CONFIG, BOT_CONFIG)
        sim.run()

    rows = []
    for r in range(num_rounds):
        rows.append({
            "round": r + 1,
            "reach": sim.history["reach"][r],
            "forwarding_rate": sim.history["forwarding_rate"][r],
            "mcr": sim.history["misinformation_contamination"][r],
            "false_reach_count": sim.history["false_reach_count"][r],
            "average_reputation": sim.history["average_reputation"][r],
        })
    return rows


def run_network_comparison(pickle_paths, n_sim=50, rounds=250, p_reveal=0.5, base_seed=42):
    """Run the network-comparison experiment.

    For each network type, runs n_sim independent simulations and collects
    per-round metrics.

    Args:
        pickle_paths: Dict mapping network_type → pickle file path.
        n_sim: Number of independent simulation runs per network.
        rounds: Number of rounds per simulation.
        p_reveal: Truth revelation probability (baseline).
        base_seed: Base random seed; each run uses base_seed + run_id.

    Returns:
        pd.DataFrame with columns:
            network_type, sim_run, round, reach, forwarding_rate,
            mcr, false_reach_count, average_reputation.
    """
    all_rows = []
    total = len(pickle_paths) * n_sim
    with tqdm(total=total, desc="Network comparison") as pbar:
        for network_type, pkl_path in pickle_paths.items():
            for sim_run in range(n_sim):
                seed = base_seed + sim_run
                rows = _run_one_simulation(pkl_path, rounds, p_reveal, seed)
                for row in rows:
                    row["network_type"] = network_type
                    row["sim_run"] = sim_run
                all_rows.extend(rows)
                pbar.update(1)

    df = pd.DataFrame(all_rows)
    # Reorder columns for readability
    cols = ["network_type", "sim_run", "round",
            "reach", "forwarding_rate", "mcr",
            "false_reach_count", "average_reputation"]
    return df[cols]


def run_factchecking_sweep(pickle_paths, p_reveal_values,
                           n_sim=50, rounds=250, base_seed=42):
    """Run the fact-checking sweep experiment.

    For each (network_type, p_reveal) pair, runs n_sim simulations.

    Args:
        pickle_paths: Dict mapping network_type → pickle file path.
        p_reveal_values: List of p_reveal values to test.
        n_sim: Number of simulation runs per (network, p_reveal) combination.
        rounds: Number of rounds per simulation.
        base_seed: Base random seed.

    Returns:
        pd.DataFrame with columns:
            network_type, p_reveal, sim_run, round, reach, forwarding_rate,
            mcr, false_reach_count, average_reputation.
    """
    all_rows = []
    total = len(pickle_paths) * len(p_reveal_values) * n_sim
    with tqdm(total=total, desc="Fact-checking sweep") as pbar:
        for network_type, pkl_path in pickle_paths.items():
            for p_reveal in p_reveal_values:
                for sim_run in range(n_sim):
                    seed = base_seed + sim_run
                    rows = _run_one_simulation(pkl_path, rounds, p_reveal, seed)
                    for row in rows:
                        row["network_type"] = network_type
                        row["p_reveal"] = p_reveal
                        row["sim_run"] = sim_run
                    all_rows.extend(rows)
                    pbar.update(1)

    df = pd.DataFrame(all_rows)
    cols = ["network_type", "p_reveal", "sim_run", "round",
            "reach", "forwarding_rate", "mcr",
            "false_reach_count", "average_reputation"]
    return df[cols]


def main():
    """Run both experiments and save results to output/."""
    from networks import get_pickle_paths, generate_all_networks, compute_diagnostics, save_diagnostics

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: generate networks (idempotent — skips if pickle already exists) ─
    print("=" * 60)
    print("Step 1: Generating network pickles")
    print("=" * 60)
    pickle_paths = get_pickle_paths(seed=42)
    missing = [nt for nt, p in pickle_paths.items() if not os.path.exists(p)]
    if missing:
        print(f"Missing pickles for: {missing} — generating now...")
        networks = generate_all_networks(seed=42)
        diag_df = compute_diagnostics(networks)
        save_diagnostics(diag_df)
    else:
        print("All network pickles already exist.")
        # Still regenerate diagnostics if missing
        diag_path = OUTPUT_DIR / "network_diagnostics.csv"
        if not diag_path.exists():
            networks = generate_all_networks(seed=42)
            diag_df = compute_diagnostics(networks)
            save_diagnostics(diag_df)

    # ── Step 2: network comparison ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Network comparison (p_reveal=0.5)")
    print("=" * 60)
    df_comparison = run_network_comparison(
        pickle_paths=pickle_paths,
        n_sim=50,
        rounds=250,
        p_reveal=0.5,
        base_seed=42,
    )
    out_comparison = OUTPUT_DIR / "results_network_comparison.csv"
    df_comparison.to_csv(out_comparison, index=False)
    print(f"Saved → {out_comparison}  ({len(df_comparison):,} rows)")

    # ── Step 3: fact-checking sweep ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Fact-checking sweep")
    print("=" * 60)
    p_reveal_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    df_sweep = run_factchecking_sweep(
        pickle_paths=pickle_paths,
        p_reveal_values=p_reveal_values,
        n_sim=50,
        rounds=250,
        base_seed=42,
    )
    out_sweep = OUTPUT_DIR / "results_network_x_factchecking.csv"
    df_sweep.to_csv(out_sweep, index=False)
    print(f"Saved → {out_sweep}  ({len(df_sweep):,} rows)")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
