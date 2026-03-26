# Misinformation Diffusion Simulation — Project Overview

This repository contains two self-contained parts, separated into `old/` and `new/`.

---

## `old/` — Original Paper Replication

**"Misinformation Spread on Social Media: A Game-Theoretic Approach"**

This folder contains everything built for the original paper simulation: the core `snlearn` agent-based simulation library, experiment scripts, configuration files, the Facebook ego-network data from SNAP, all batch results, figures, and the LaTeX paper source.

```
old/
├── code/
│   ├── snlearn/           ← core simulation library (agents, messages, network, simulation)
│   ├── run_simulation.py  ← single-run simulation CLI
│   ├── batch_simulation.py ← batch runner (vary parameters across 50 runs)
│   ├── analyze_results.py ← plotting and statistical analysis
│   ├── compare_experiments.py
│   ├── create_network.py  ← network generation from SNAP data
│   ├── plot_reach_only.py
│   ├── config/            ← JSON configs for agents, bots, influencers, simulation
│   ├── tests/
│   └── Network.ipynb
├── data/
│   ├── facebook_combined.txt          ← SNAP Facebook ego-network (4039 nodes)
│   ├── facebook_community_*nodes*.pkl ← subsampled Facebook networks
│   └── barabasi_default_*nodes*.pkl   ← exploratory BA networks
├── figures/               ← all original experiment figures
├── results/               ← CSV results from original experiments
├── output/                ← timestamped single-run output folders
├── paper/                 ← LaTeX source for the paper
├── simulation_round.gif
└── INSTRUCOES_FACTCHECK.md
```

### Running the original simulation

```sh
uv sync
# Generate a 100-node Facebook network
uv run python old/code/create_network.py 100 facebook \
  --sampling_method community \
  --network_file old/data/facebook_combined.txt \
  --save

# Run with influencers and bots
uv run python old/code/run_simulation.py \
  --config-sim old/code/config/config_sim_general.json \
  --config-agent old/code/config/config_sim_agent.json \
  --config-influencer old/code/config/config_sim_influencer.json \
  --config-bot old/code/config/config_sim_bot.json \
  --no-gif --seed 42
```

---

## `new/` — Network Structure Extension

**"How Network Structure Shapes Reputation-Based Filtering of Misinformation"**

This folder contains everything built for the extension project. It tests whether the structural properties of a network (degree distribution, clustering, path length) amplify or dampen the reputation-based filtering mechanism, and therefore lead to more or less misinformation spread. Four networks are compared: Erdős–Rényi, Barabási–Albert, Stochastic Block Model, and the Facebook baseline.

```
new/
├── code/
│   ├── networks.py           ← network generators + structural diagnostics
│   ├── network_experiment.py ← experiment runner (network comparison + fact-checking sweep)
│   ├── plot_network_results.py ← all 5 figures
│   └── snlearn/              ← copy of core simulation library (self-contained)
├── data/
│   ├── erdos_renyi_100nodes_seed42.pkl
│   ├── barabasi_albert_m6_100nodes_seed42.pkl
│   ├── sbm_100nodes_seed42.pkl
│   └── facebook_community_100nodes_seed42.pkl
├── output/
│   ├── network_diagnostics.csv
│   ├── results_network_comparison.csv
│   └── results_network_x_factchecking.csv
├── figures/
│   ├── fig1_mcr_over_rounds.png
│   ├── fig2_false_reach_over_rounds.png
│   ├── fig3_reputation_over_rounds.png
│   ├── fig4_mcr_vs_preveal.png
│   └── fig5_structural_scatter.png
└── RESULTS.md               ← written summary of findings and figure descriptions
```

### Running the network extension

```sh
uv sync
# Step 1: generate networks and diagnostics
uv run python new/code/networks.py

# Step 2: run experiments (network comparison + fact-checking sweep)
uv run python new/code/network_experiment.py

# Step 3: generate all 5 figures
uv run python new/code/plot_network_results.py
```

Results land in `new/output/` and figures in `new/figures/`. See `new/RESULTS.md` for a full summary of findings.

---

## Root-level files

| File | Description |
|---|---|
| `CLAUDE.md` | Project instructions for Claude Code |
| `pyproject.toml` | Python dependencies and project metadata (shared) |
| `uv.lock` | Lockfile — do not edit manually |
| `README.md` | This file |
