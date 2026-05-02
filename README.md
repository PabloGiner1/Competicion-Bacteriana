# PDH Bacterial Competition Model on Complex Networks

This repository contains the implementation and simulation of a three-state bacterial competition model, referred to as the **PDH model**, on complex networks. The goal is to study the interaction between prey bacteria and predator bacteria while explicitly accounting for the local contact structure defined by a network.

The model is studied using two complementary approaches:

1. a stochastic node-level microscopic simulation;
2. a Markov approximation on networks, used to study the average evolution of the system.

---

## 1. Model Description

Each node in the network represents a site that can be in one of three possible states:

- **P (Prey):** a node occupied by prey bacteria, associated with *Pseudomonas aeruginosa*.
- **D (Predator):** a node occupied by predator bacteria, such as *Bdellovibrio* or *Vampirovibrio*.
- **H (Hole):** an empty or available site.

The network defines which nodes can interact. Therefore, transitions only occur between neighboring nodes.

---

## 2. Transition Rules

The PDH dynamics are governed by three local stochastic processes.

### Predation

A prey node in contact with a predator can be attacked. The prey node becomes occupied by a predator with probability `alpha`.

```math
P + D \xrightarrow{\alpha} 2D
```

### Colonization

An empty node can be colonized by prey if it has a prey neighbor. This occurs with probability `beta`.

```math
P + H \xrightarrow{\beta} 2P
```

### Predator Mortality

A predator can die naturally, leaving the node empty with probability `mu`.

```math
D \xrightarrow{\mu} H
```

The convention used throughout the code is:

```text
alpha = predation probability
beta  = colonization probability
mu    = predator mortality probability
```

With this convention, the theoretical mean-field survival threshold for predators is:

```math
k_c = \frac{\mu}{\alpha}
```

---

## 3. Simulation Approaches

### Stochastic Microscopic Simulation

The function `simulate_pdh` assigns each node a concrete state: `P`, `D`, or `H`. At each time step, node states are updated according to the stochastic transition rules and the neighborhood structure of the network.

This approach is useful for visualizing the spatial evolution of the system and generating animations.

### Network Markov Dynamics

The function `markov_pdh_dynamics` assigns each node probabilities of being in states `P`, `D`, and `H`, rather than assigning fixed states. At each time step, predation and colonization probabilities are computed from the states of neighboring nodes.

This approach is used to study:

- mean temporal evolution of `P`, `D`, and `H`;
- stationary state as a function of the average degree `<k>`;
- dependence on the parameters `alpha`, `beta`, and `mu`;
- stationary-state heatmaps.

---

## 4. Supported Network Topologies

The project supports several network types generated with `NetworkX`:

- **Erdős-Rényi networks:** random graphs `G(N, p)`, mainly used to study the effect of average degree.
- **Barabási-Albert networks:** scale-free networks with hubs.
- **Watts-Strogatz networks:** small-world networks.

For Erdős-Rényi networks with target average degree `<k>`, the connection probability is chosen as:

```math
p = \frac{\langle k \rangle}{N-1}
```

---

## 5. Project Structure

```text
.
├── config.py
├── main.py
├── outputs/
└── src/
    ├── graphs/
    │   └── generate_graphs.py
    ├── simulation/
    │   └── simulation_model.py
    ├── utils/
    │   ├── helpers.py
    │   └── metrics.py
    └── visualization/
        └── plots.py
```

### Main Files

- `config.py`  
  Stores global parameters for simulations, networks, model dynamics, random seeds, output saving, and execution flags.

- `main.py`  
  Entry point of the project. It runs the simulations and visualizations enabled in `config.py`.

- `src/simulation/simulation_model.py`  
  Implements the stochastic simulation, the Markov dynamics, and the stationary-state computation.

- `src/graphs/generate_graphs.py`  
  Contains functions for generating Erdős-Rényi, Barabási-Albert, and Watts-Strogatz networks.

- `src/visualization/plots.py`  
  Contains all visualization routines: temporal evolution, stationary state, parameter variation, heatmaps, and animations.

- `src/utils/helpers.py`  
  Provides helper functions for state counting, random seed management, and automatic figure saving.

- `src/utils/metrics.py`  
  Provides structural network metrics, such as average degree, clustering, average path length, diameter, centralities, density, and assortativity.

---

## 6. Available Visualizations

The project can generate the following outputs.

### Original Stochastic Simulation

Temporal evolution of the number of nodes in states `P`, `D`, and `H`.

### Individual Markov Dynamics

Evolution of the mean fractions of prey, predators, and empty sites for different values of `<k>`.

### Combined Markov Dynamics

Comparison of the temporal evolution of each state for several average degrees in the same plot.

### Stationary State vs Average Degree

Stationary fractions

```math
P_\infty,\quad D_\infty,\quad H_\infty
```

as a function of `<k>`, including the theoretical prediction:

```math
k_c=\frac{\mu}{\alpha}
```

### Parameter Variation

Stationary predator density `D_inf` as a function of `<k>`, varying:

- `mu`: predator mortality;
- `beta`: prey colonization of empty sites;
- `alpha`: predation probability.

### Heatmap

Heatmap of `D_inf` in the `(<k>, mu)` plane, including the theoretical threshold line:

```math
\mu = \alpha \langle k \rangle
```

### Animation

Generation of a GIF showing:

- spatial evolution of the network;
- temporal evolution of `P`, `D`, and `H`.

---

## 7. Configuration

The main parameters are set in `config.py`.

Example:

```python
N_NODES = 400
STEPS = 150

ALPHA = 0.2   # Predation: P + D -> 2D
BETA = 0.6    # Colonization: P + H -> 2P
MU = 0.8      # Predator mortality: D -> H

P_PROPORTION = 0.2
D_PROPORTION = 0.05
H_PROPORTION = 0.75
```

Execution flags are also controlled from `config.py`:

```python
RUN_ORIGINAL = True
RUN_MARKOV_INDIVIDUAL = True
RUN_MARKOV_COMBINED = True
RUN_STATIONARY = True
RUN_STATIONARY_PARAMETER_VARIATION = True
RUN_HEATMAP = True
RUN_ANIMATION = True
```

---

## 8. Random Seeds

The project can use either a fixed seed for reproducible results or a time-based seed for new realizations at each run.

```python
USE_FIXED_SEED = False
FIXED_SEED = 42
```

If `USE_FIXED_SEED = True`, the value of `FIXED_SEED` is used.

If `USE_FIXED_SEED = False`, a seed based on the current time is generated automatically.

---

## 9. Output Saving

Figures are automatically saved if:

```python
SAVE_FIGURES = True
```

Outputs are organized in parameter-specific folders:

```text
outputs/
└── alpha_0p200_beta_0p600_mu_0p800/
    ├── original_simulation_erdos.png
    ├── markov_individual_k_2.png
    ├── markov_combined_predators.png
    ├── stationary_state.png
    ├── variation_mu.png
    ├── variation_beta.png
    ├── variation_alpha.png
    ├── heatmap_mu.png
    └── pdh_animation_k_4.gif
```

This structure allows different parameter sets to be generated without overwriting previous results.

---

## 10. Running the Project

To run the project:

```bash
python main.py
```

The simulations executed depend on the flags enabled in `config.py`.

---

## 11. Requirements

The project requires:

- Python 3.x
- NumPy
- NetworkX
- Matplotlib
- Pillow

Recommended installation:

```bash
pip install numpy networkx matplotlib pillow
```

---

## 12. Interpretation

The key condition for predator survival is determined by the balance between predation, mortality, and network connectivity.

In the mean-field approximation, predators can persist when:

```math
\alpha \langle k \rangle > \mu
```

or equivalently:

```math
\langle k \rangle > k_c = \frac{\mu}{\alpha}
```

Below this threshold, predators tend to disappear. Above it, the system reaches a coexistence regime involving prey, predators, and empty sites.
