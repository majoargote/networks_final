# CLAUDE.md — Networks Project: How Network Structure Shapes Reputation-Based Filtering

## Your Role
You are a research assistant helping extend an existing game-theoretic misinformation simulation.
Read ALL relevant files in this project before writing any code. Understand the existing model fully before proposing or implementing changes.

---

## Project Folder Structure

```
project-root/
├── CLAUDE.md               ← this file
├── pyproject.toml          ← dependencies and project metadata
├── uv.lock                 ← lockfile (do not edit manually)
├── README.md               ← project readme
├── code/                   ← ALL source code lives here (read every .py file)
├── data/                   ← input data (e.g. Facebook ego-network files from SNAP)
├── figures/                ← save all output figures (.png) here
├── output/                 ← save all CSV results here
├── results/                ← any additional result artifacts
└── paper/                  ← paper PDF and related documents (read-only, do not modify)
```

**Before writing any code:**
1. Read every `.py` file inside `code/`
2. Read `pyproject.toml` to understand the dependency setup and how the project is run
3. Read `README.md` for any existing usage instructions
4. Check `data/` to understand what network files are already available

---

## Context: The Existing Model

This project extends the paper **"Misinformation Spread on Social Media: A Game-Theoretic Approach"**. Here is a precise summary of what already exists:

### Agents
Three types: **regular users**, **influencers**, and **bots**, differing in network position and behavioral parameters.

| Parameter | Regular | Influencer | Bot |
|---|---|---|---|
| Initial reputation µ | 0.0 | 1.0 | 0.0 |
| Bias strength ω | 0.5 | 1.0 | 1.0 |
| Reward strength γ₀ | 0.5 | 1.0 | 0.0 |
| Penalty scale δ₀ | 0.5 | 0.2 | 0.0 |
| Forwarding cost k | 0.1 | 0.1 | 0.0 |

### Messages
- `bias = 2x - 1`, where `x ~ Beta(α, β)`, so `bias ∈ [-1, 1]`
- `truth ~ Bernoulli(q)`, where `q = 0.6` baseline

### Belief Formation
When truth is unobservable, agents estimate it via:
- **Ideological proximity**: `Φ(biasᵢ, biasₘ) = 1 - |biasᵢ - biasₘ|`
- **Sender reputation**: `Γ(R) = 1 / (1 + e^{-R})` (sigmoid)
- **Estimated truth**: `truth_hat = 1 if Φ · Γ ≥ 0.5 else 0`

### Utility & Forwarding Decision
```
utility = Γ · (ω·Φ + γ·belief - δ·(1 - belief)) - k   if action = 1
        = 0                                              if action = 0
```
Agent forwards if `average_utility ≥ 0`.

### Reputation Update
```
R_next = R + γ·Γ(R)   if forwarded a TRUE message
R_next = R - δ(R)     if forwarded a FALSE message   [δ(R) = δ₀·σ(R)·(1+σ(R))]
R_next = R             if did not forward
```
Reputation is only updated when truth is revealed (with probability `p_reveal`).

### Original Network
Facebook ego-network from SNAP (4,039 nodes, 88,234 edges), subsampled to **n=100 agents** mixing high-degree, low-degree, and bridge nodes across 8 communities.

### Simulation Loop (per round)
1. A message appears at initial senders (influencers if present, else high-degree nodes)
2. Recipients form beliefs and compute utility
3. Agents forward or ignore
4. Forwarders pass message to their neighbors
5. Steps 2–4 repeat until no new forwards
6. Truth revealed with probability `p_reveal`; reputations updated

### Baseline Settings
- Agents: 100 | Rounds: 250 | Simulations: 50 | Seed: 42
- `p_reveal = 0.5` | `q = 0.6`

### Key Metrics
| Metric | Definition |
|---|---|
| Reach | Fraction of agents who received the message |
| Forwarding Rate | Fraction of recipients who forwarded |
| Misinformation Contamination Rate (MCR) | Forwarding rate conditional on message being false |
| False Reach (FRC) | Absolute count of agents exposed to false content |
| Average Reputation | Mean reputation across all agents |

---

## The New Research Question

**How does network structure shape the effectiveness of reputation-based filtering of misinformation?**

The original paper used only the Facebook ego-network. This extension tests whether the structural properties of a network — degree distribution, clustering, path length — amplify or dampen the reputation mechanism, and therefore lead to more or less misinformation spread.

---

## What You Need to Build

### Step 1 — Read existing code
Read every `.py` file inside `code/`. Understand how the simulation is implemented — class structures, function signatures, config patterns — before touching anything. Also check `data/` for what network files are already present.

### Step 2 — Add network generators
Add a new module inside `code/` (e.g. `code/networks.py`) or extend the existing network-loading code to generate the following networks, all with **n=100 nodes** and comparable average degree to the original Facebook subsample (target: mean degree ≈ 12):

1. **Random (Erdős–Rényi)**: `G = nx.erdos_renyi_graph(n=100, p=0.12)` — no structure, random connections, mean degree ≈ 11.9
2. **Scale-free (Barabási–Albert)**: `G = nx.barabasi_albert_graph(n=100, m=6)` — hub-dominated, mean degree ≈ 11.4
3. **Stochastic Block Model (SBM)**: 8 communities of ~12–13 nodes each, with higher within-community edge probability and lower between-community probability, calibrated to keep mean degree ≈ 12. This mirrors the 8-community structure of the original Facebook network and is the formal generative model for community structure. Use `nx.stochastic_block_model()`.
4. **Original Facebook subsampled network** (keep as baseline / real-world reference)

Save each generated network as a pickle file in `data/` following the existing naming convention, so `Simulation` can load them without any changes.

For each network, also compute and store these **structural diagnostics** to use in analysis:
- Mean degree
- Degree distribution (histogram)
- Average clustering coefficient
- Average shortest path length
- Diameter (if connected; else largest connected component)
- Degree assortativity

### Step 3 — Run comparative simulations
Hold the agent composition and all behavioral parameters constant at their baseline values (84 regular, 8 influencers, 8 bots). Vary only the network structure. For each network type, run the full simulation (`Nsim=50`, `rounds=250`) and collect all metrics per round.

Additionally, run a secondary sweep: for each network type, vary `p_reveal ∈ {0.0, 0.2, 0.5, 0.8, 1.0}` to see how fact-checking intensity interacts with network structure.

### Step 4 — Output results
Save results as CSV files inside `output/` with clear naming:
- `output/results_network_comparison.csv` — one row per (network_type, simulation_run, round)
- `output/results_network_x_factchecking.csv` — one row per (network_type, p_reveal, simulation_run, round)
- `output/network_diagnostics.csv` — one row per network type with structural metrics

### Step 5 — Plotting
Generate the following figures and save as `.png` inside `figures/`:

1. **Figure 1**: MCR over rounds, one line per network type (averaged over simulations) — shows which topology lets misinformation persist
2. **Figure 2**: False Reach (count) over rounds by network type
3. **Figure 3**: Average Reputation over rounds by network type
4. **Figure 4**: MCR vs `p_reveal`, one line per network type — the interaction plot
5. **Figure 5**: Scatter of structural diagnostics (e.g., clustering coefficient vs. mean MCR; mean path length vs. mean FRC) — to visualize which structural properties correlate with misinformation spread

---

## Hypotheses to Keep in Mind (do not hardcode conclusions)

These are working hypotheses the simulation should help test:

- **H1**: Scale-free networks (BA) will show higher false reach because hubs act as super-spreaders, overriding reputation-based caution.
- **H2**: Random networks (ER) will show more uniform diffusion, where reputation filtering is most effective because no single node dominates.
- **H3**: SBM networks will reproduce echo-chamber dynamics similar to the original Facebook results — misinformation stays contained within communities but reaches high saturation inside them.
- **H4**: Fact-checking (higher `p_reveal`) will be most effective in ER networks and least effective in BA networks, because hubs near low-reputation agents can sustain false diffusion even when high-reputation agents opt out.

---

## Constraints & Style

- Use the **exact same behavioral model** as the original paper — do not change agent utility functions, reputation update rules, or belief formation
- All new source files go inside `code/`; figures in `figures/`; CSVs in `output/`; do not create files outside these folders
- Keep code modular: network generation, simulation, and plotting should be in separate functions/files
- Every function should have a docstring
- Use `numpy.random.default_rng(seed)` for reproducibility
- Prefer `networkx` for graph operations and `matplotlib`/`seaborn` for plots
- The project uses `uv` for dependency management — if you need to add a new package, add it to `pyproject.toml` and note it explicitly; do not run `pip install` directly
- If the existing code uses specific class structures or conventions, follow them exactly

---

## Deliverables Checklist

- [ ] `code/networks.py` (or equivalent) — network generator with ER, BA, SBM, and Facebook baseline (4 networks total)
- [ ] All 4 networks saved as pickles in `data/` following existing naming convention
- [ ] Structural diagnostics computed and saved to `output/network_diagnostics.csv`
- [ ] Simulation runner (inside `code/`) that accepts a network as input
- [ ] `output/results_network_comparison.csv` and `output/results_network_x_factchecking.csv`
- [ ] 5 figures saved as PNG inside `figures/`
- [ ] Brief `RESULTS.md` at project root summarizing what each figure shows (2–3 sentences per figure)

---

## Final Report (Required — do this last)

After completing ALL the steps above, you MUST produce a written summary of everything you did. Do not skip this. Format it exactly as follows:

---

### ✅ What I Did — Summary

**Files created:**
List every new file you created, with its full path and a one-line description of what it does.

**Files modified:**
List every existing file you modified, with its full path, a precise description of what you changed, and why.

**Files NOT touched:**
Explicitly confirm which existing files were left completely unchanged.

**Where to find the outputs:**
- Network pickles → `data/`
- Structural diagnostics → `output/network_diagnostics.csv`
- Simulation results → `output/results_network_comparison.csv` and `output/results_network_x_factchecking.csv`
- Figures → `figures/` (list each filename)
- Results summary → `RESULTS.md`

**How to reproduce everything from scratch:**
Provide the exact commands to run, in order, so the user can regenerate all outputs from a clean state.

**Assumptions and things to be aware of:**
Flag any parameters you hardcoded, assumptions you made about existing code, or edge cases the user should know about before running or extending the project.
