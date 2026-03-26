"""
Network generators and structural diagnostics for the network-structure experiment.

Generates four networks (ER, BA, SBM, Facebook) as SocialNetwork objects,
saves them as pickle files in data/, and computes structural diagnostics.

Usage (standalone):
    uv run python code/networks.py
"""

import pickle
import os
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

# Resolve project root relative to this file (code/networks.py → project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _build_network(network_type, seed=42):
    """Instantiate a SocialNetwork for the given network_type.

    Args:
        network_type: One of 'erdos_renyi', 'barabasi_albert', 'sbm', 'facebook'.
        seed: Random seed.

    Returns:
        SocialNetwork instance.
    """
    # Import here to avoid circular issues when used as a module
    from snlearn.socialnetwork import SocialNetwork

    if network_type == "erdos_renyi":
        return SocialNetwork(
            num_agents=100,
            seed=seed,
            erdos_renyi_params={"p": 0.12},
        )
    elif network_type == "barabasi_albert":
        return SocialNetwork(
            num_agents=100,
            seed=seed,
            barabasi_params={"m": 6},
        )
    elif network_type == "sbm":
        # 8 communities of ~12-13 nodes; p_in=0.5, p_out=0.07 → mean degree ≈ 12
        return SocialNetwork(
            num_agents=100,
            seed=seed,
            sbm_params={"p_in": 0.5, "p_out": 0.07},
        )
    elif network_type == "facebook":
        network_file = str(DATA_DIR / "facebook_combined.txt")
        return SocialNetwork(
            num_agents=100,
            seed=seed,
            facebook_params={
                "network_file": network_file,
                "sampling_method": "community",
            },
        )
    else:
        raise ValueError(f"Unknown network_type: {network_type}")


def generate_all_networks(seed=42):
    """Generate all four networks and save each as a pickle in data/.

    Pickle filenames follow the existing convention:
        <network_type>_<method>_<n>nodes_seed<seed>.pkl

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping network_type → (SocialNetwork, pickle_path).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filenames = {
        "erdos_renyi":    f"erdos_renyi_100nodes_seed{seed}.pkl",
        "barabasi_albert": f"barabasi_albert_m6_100nodes_seed{seed}.pkl",
        "sbm":            f"sbm_100nodes_seed{seed}.pkl",
        "facebook":       f"facebook_community_100nodes_seed{seed}.pkl",
    }

    networks = {}
    for network_type, filename in filenames.items():
        pkl_path = DATA_DIR / filename
        print(f"Generating {network_type} network...")
        net = _build_network(network_type, seed=seed)
        with open(pkl_path, "wb") as f:
            pickle.dump(net, f)
        print(f"  Saved → {pkl_path}  (n={net.num_agents})")
        networks[network_type] = (net, str(pkl_path))

    return networks


def _undirected_graph(network):
    """Return the undirected version of a SocialNetwork's graph."""
    return network.graph.to_undirected()


def compute_diagnostics(networks):
    """Compute structural diagnostics for each network.

    Args:
        networks: Dict mapping network_type → (SocialNetwork, pickle_path).

    Returns:
        pd.DataFrame with one row per network type.
    """
    rows = []
    for network_type, (net, _) in networks.items():
        G_un = _undirected_graph(net)

        # Mean degree (undirected)
        degrees = [d for _, d in G_un.degree()]
        mean_degree = float(np.mean(degrees))

        # Average clustering coefficient
        avg_clustering = nx.average_clustering(G_un)

        # Average shortest path length and diameter (on LCC if disconnected)
        if nx.is_connected(G_un):
            G_cc = G_un
        else:
            largest_cc = max(nx.connected_components(G_un), key=len)
            G_cc = G_un.subgraph(largest_cc).copy()

        avg_path_length = nx.average_shortest_path_length(G_cc)
        diameter = nx.diameter(G_cc)

        # Degree assortativity
        assortativity = nx.degree_assortativity_coefficient(G_un)

        rows.append({
            "network_type": network_type,
            "n_nodes": net.num_agents,
            "mean_degree": round(mean_degree, 3),
            "avg_clustering": round(avg_clustering, 4),
            "avg_path_length": round(avg_path_length, 4),
            "diameter": diameter,
            "degree_assortativity": round(assortativity, 4),
        })

    df = pd.DataFrame(rows)
    return df


def save_diagnostics(df, path=None):
    """Save structural diagnostics DataFrame to CSV.

    Args:
        df: DataFrame returned by compute_diagnostics().
        path: Output path. Defaults to output/network_diagnostics.csv.
    """
    if path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "network_diagnostics.csv"
    df.to_csv(path, index=False)
    print(f"Diagnostics saved → {path}")


def get_pickle_paths(seed=42):
    """Return a dict mapping network_type → absolute pickle path string.

    Useful for experiment runners that need to pass pickle paths to Simulation.
    """
    return {
        "erdos_renyi":    str(DATA_DIR / f"erdos_renyi_100nodes_seed{seed}.pkl"),
        "barabasi_albert": str(DATA_DIR / f"barabasi_albert_m6_100nodes_seed{seed}.pkl"),
        "sbm":            str(DATA_DIR / f"sbm_100nodes_seed{seed}.pkl"),
        "facebook":       str(DATA_DIR / f"facebook_community_100nodes_seed{seed}.pkl"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Generating all four networks")
    print("=" * 60)
    networks = generate_all_networks(seed=42)

    print("\nComputing structural diagnostics...")
    diag_df = compute_diagnostics(networks)
    save_diagnostics(diag_df)

    print("\nDiagnostics summary:")
    print(diag_df.to_string(index=False))
    print("\nDone.")
