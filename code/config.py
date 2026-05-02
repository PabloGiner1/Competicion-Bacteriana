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

GRAPH_TYPE = "erdos"  # Options: "erdos", "barabasi", "watts"


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

RUN_ORIGINAL = True
RUN_MARKOV_INDIVIDUAL = True
RUN_MARKOV_COMBINED = True
RUN_STATIONARY = True
RUN_STATIONARY_PARAMETER_VARIATION = True
RUN_HEATMAP = True
RUN_ANIMATION = True


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