"""
Global configuration file for the PDH bacterial competition model.

Convention:
    P: prey bacteria
    D: predator bacteria
    H: empty site

Model parameters:
    alpha: predation probability, P + D -> 2D
    beta: colonization probability, P + H -> 2P
    mu: predator mortality probability, D -> H

For the current convention, the theoretical threshold is:

    k_c = mu / alpha
"""

# =========================================================
# Simulation size
# =========================================================

N_NODES = 400
STEPS = 150


# =========================================================
# Network parameters
# =========================================================

ERDOS_P = 0.05
BARABASI_M = 3
WATTS_K = 6
WATTS_P = 0.1

GRAPH_TYPE = "watts"  # Options: "erdos", "barabasi", "watts"


# =========================================================
# PDH model parameters
# =========================================================

ALPHA = 0.2   # Predation: P + D -> 2D
BETA = 0.6    # Colonization: P + H -> 2P
MU = 0.8      # Predator mortality: D -> H

PDH_PARAMS = [ALPHA, BETA, MU]


# =========================================================
# Initial conditions
# =========================================================

P_PROPORTION = 0.2
D_PROPORTION = 0.05
H_PROPORTION = 0.75

INITIAL_PROPORTIONS = [P_PROPORTION, D_PROPORTION, H_PROPORTION]


# =========================================================
# Execution switches
# =========================================================

RUN_ORIGINAL = False
RUN_MARKOV_INDIVIDUAL = False
RUN_MARKOV_COMBINED = False
RUN_STATIONARY = False
RUN_STATIONARY_PARAMETER_VARIATION = False
RUN_HEATMAP = False
RUN_ANIMATION = False


# =========================================================
# Random seed configuration
# =========================================================

USE_FIXED_SEED = False
FIXED_SEED = 42


# =========================================================
# Output configuration
# =========================================================

SAVE_FIGURES = True
OUTPUT_DIR = "outputs"

# =========================================================
# Innovation module switches
# =========================================================

RUN_INNOVATIONS = True

RUN_HETEROGENEOUS_PARAMETERS = False
RUN_MOBILITY = False
RUN_DYNAMIC_NETWORK = False
RUN_SCALE_FREE_COMPARISON = True


# =========================================================
# Innovation parameters
# =========================================================

HETEROGENEITY_STRENGTH = 0.5

MOBILITY_RATES = [0.0, 0.01, 0.1 ,0.5, 1.0]

REWIRING_RATES = [0.0, 0.01, 0.15, 0.5, 1.0]

INNOVATION_K_VALUES = [2, 3, 4, 5, 6, 8, 10]

INNOVATION_T_MARKOV = 250
INNOVATION_N_REALIZATIONS = 4
INNOVATION_TAIL = 30