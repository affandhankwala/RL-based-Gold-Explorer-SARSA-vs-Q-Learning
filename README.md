# RL-based Gold Explorer: SARSA vs. Q-Learning

A grid-world reinforcement-learning study that pits an off-policy **Q-Learning** agent against an
on-policy **SARSA** agent on a gold-hunting task, comparing their performance and convergence with and
without exploring starts.

## Introduction

This is the third project for Johns Hopkins University's *Reinforcement Learning* course (705.741).

## Problem Statement

An agent explores an **8 × 8 grid** using horizontal and vertical moves. The grid is populated with fixed
entities, each occupying a full cell:

- **Mountains** — impassable cells that cannot be entered from any direction.
- **Mine field** — terminal states that penalize the agent with the `LOSE` reward.
- **Gold coins** — reward the agent when first entered (collectible only once).
- **Motherlode** — the winning terminal state, granting the `WIN` reward.

Following standard RL practice, every action incurs a small step penalty to encourage the agent to find
efficient paths.

## Methodology

Two control policies are compared:

- **Q-Learning** — off-policy temporal-difference control.
- **SARSA(0)** — on-policy temporal-difference control.

Each is tested both **with and without exploring starts** (varying the agent's initial position). To draw
statistically meaningful conclusions, **50 agents of each type** are trained for a tuned number of
episodes. Performance and convergence are assessed by aggregating the returns over the final 50 episodes
across all 50 agents (computing per-episode mean return and standard error).

## Repository Structure

```
RL-based-Gold-Explorer-SARSA-vs-Q-Learning/
├── README.md
├── Q_Learning_versus_SARSA_report.pdf   # Full report
├── src/
│   ├── Project3_main.py     # Entry point: trains agent populations and plots results
│   ├── Project3_env.py      # GoldExplorer grid-world environment
│   ├── Project3_agentQ.py   # Q-Learning agent
│   └── Project3_agentS.py   # SARSA(0) agent
└── plots/                   # Learning-curve plots (with/without exploring starts)
```

## Running

```bash
cd src
python Project3_main.py
```

Training hyperparameters (learning rate `alpha`, discount, episode count, agent count) are configured in
`Project3_main.py`. Generated learning-curve plots are saved under `plots/`.

### Requirements

- Python 3
- `numpy`, `matplotlib`

## Results

See [`Q_Learning_versus_SARSA_report.pdf`](Q_Learning_versus_SARSA_report.pdf) for the full comparison of
the two policies, the effect of exploring starts, and the statistical analysis of performance and
convergence. Learning curves for each configuration are available in the `plots/` directory.
