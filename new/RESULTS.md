# RESULTS.md — Network Structure and Reputation-Based Filtering

## Structural Diagnostics

| Network | n | Mean Degree | Avg Clustering | Avg Path Length | Diameter | Assortativity |
|---|---|---|---|---|---|---|
| Erdős-Rényi | 100 | 11.62 | 0.114 | 2.110 | 4 | +0.087 |
| Barabási-Albert | 100 | 11.28 | 0.210 | 2.107 | 4 | −0.100 |
| Stochastic Block Model | 100 | 11.34 | 0.172 | 2.169 | 4 | −0.003 |
| Facebook (real) | 100 | 4.82 | 0.537 | 3.571 | 8 | −0.254 |

**Note:** The three synthetic networks are calibrated to mean degree ≈ 11–12. The Facebook community-sampled subgraph has substantially lower mean degree (4.82) but much higher clustering (0.54), reflecting the dense-community, sparse-bridge structure of real ego-networks.

---

## Figure Descriptions

### Figure 1 — MCR over Rounds by Network Type
`figures/fig1_mcr_over_rounds.png`

MCR (Misinformation Contamination Rate) is plotted per round, averaged over 50 simulations, with 95% confidence bands. All four networks show MCR stabilising quickly (within ~20 rounds). Among the dense synthetic networks, Barabási-Albert produces the highest sustained MCR (≈14.0%), consistent with H1: hubs act as super-spreaders that repeatedly amplify false messages regardless of sender reputation. Erdős-Rényi shows the lowest MCR (≈13.0%), consistent with H2: without dominant hubs, reputation filtering is more effective. The SBM sits between the two (≈13.2%). The Facebook network shows the highest raw MCR (≈14.9%) but this must be interpreted alongside its much lower mean degree — fewer total messages are forwarded, but a higher fraction are false.

### Figure 2 — False Reach over Rounds by Network Type
`figures/fig2_false_reach_over_rounds.png`

False Reach counts the number of agents who received a false message in each round. The three dense synthetic networks (ER, BA, SBM) show very similar false-reach profiles, each exposing ≈37 agents per false-message round on average — roughly 37% of the network. The Facebook network exposes only ≈20 agents per false-message round, directly reflecting its lower mean degree and longer path lengths: false content spreads further in a sparse but clustered network only when it finds a bridging path, and those paths are rarer and longer. This shows that raw connectivity (mean degree) is the dominant driver of false-content exposure, more so than network topology type.

### Figure 3 — Average Reputation over Rounds by Network Type
`figures/fig3_reputation_over_rounds.png`

Average agent reputation rises steadily across all network types as agents accumulate rewards for forwarding true messages (prob_truth = 0.6, so most messages are true). Barabási-Albert agents accumulate the highest reputation (≈1.87 at steady state), because hubs receive and forward many true messages and collect rewards at high rate. Facebook agents accumulate the least (≈1.48), since the sparser network means fewer forwarding opportunities and thus fewer reputation-reward events. This result shows that topology shapes not only misinformation spread but also the long-run distribution of reputation — denser networks with active hubs create more inequality in reputation accumulation.

### Figure 4 — MCR vs Fact-checking Intensity by Network Type
`figures/fig4_mcr_vs_preveal.png`

This interaction plot shows how mean MCR (averaged over all rounds and simulations) varies with fact-checking probability (p_reveal) across the four network types. Contrary to H4, higher p_reveal is associated with a slight *increase* in MCR for all networks. This is a model-specific result: more frequent truth revelation triggers more frequent reputation *rewards* (since 60% of messages are true), which raises average reputation and thereby raises Γ(R) = sigmoid(R), making agents more willing to forward subsequent messages — including false ones. The effect is small but consistent. Barabási-Albert shows the steepest increase, again reflecting hub dynamics. Erdős-Rényi shows the flattest response, suggesting reputation filtering is most stable in random networks.

### Figure 5 — Network Structure vs Misinformation Outcomes
`figures/fig5_structural_scatter.png`

Two scatter plots place each network as a point. Panel A (clustering coefficient vs mean MCR) reveals a positive relationship: the real Facebook network, with its high clustering (0.54), shows higher MCR despite low connectivity, while ER with low clustering (0.11) shows the lowest MCR among the dense networks. Panel B (average path length vs mean False Reach) shows a strong negative relationship: the Facebook network, with the longest average path (3.57), exposes far fewer agents to false content, while the three dense synthetic networks cluster together at short path lengths (≈2.1) and high false reach (≈37 agents). Together, these two panels confirm that *path length* is the primary structural correlate of false-content exposure, while *clustering* is associated with MCR through echo-chamber dynamics.

---

## Hypothesis Assessment

| Hypothesis | Supported? | Notes |
|---|---|---|
| H1: BA → highest false reach via hubs | Partially | BA has highest MCR among dense networks; false reach is similar across dense topologies |
| H2: ER → most effective reputation filtering | Supported | ER shows the lowest MCR and the flattest response to p_reveal |
| H3: SBM → echo-chamber dynamics like Facebook | Not clearly | SBM behaves similarly to ER/BA at these parameters; SBM-specific echo effects may require stronger within/between probability contrast |
| H4: Fact-checking more effective in ER than BA | Not supported | Higher p_reveal slightly *increases* MCR in all networks due to reputation-reward feedback |
