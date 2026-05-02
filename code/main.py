"""
Main entry point for running PDH simulations and visualizations.
"""

from config import (
    USE_FIXED_SEED,
    FIXED_SEED,
    RUN_ORIGINAL,
    RUN_MARKOV_INDIVIDUAL,
    RUN_MARKOV_COMBINED,
    RUN_STATIONARY,
    RUN_STATIONARY_PARAMETER_VARIATION,
    RUN_HEATMAP,
    RUN_ANIMATION,
)

from src.utils.helpers import set_global_seed
from src.visualization.plots import (
    plot_original_simulation,
    plot_markov_individual,
    plot_markov_combined,
    plot_stationary_state,
    plot_stationary_vs_k_parameter_variation,
    plot_heatmap,
    animate_pdh_simulation,
)


def main():
    """Run the selected simulations according to config.py."""

    if USE_FIXED_SEED:
        seed = set_global_seed(FIXED_SEED)
        print(f"Using fixed seed: {seed}")
    else:
        seed = set_global_seed()
        print(f"Using time-based random seed: {seed}")

    if RUN_ORIGINAL:
        print("Running original stochastic simulation...")
        plot_original_simulation()

    if RUN_MARKOV_INDIVIDUAL:
        print("Running individual Markov simulations...")
        plot_markov_individual()

    if RUN_MARKOV_COMBINED:
        print("Running combined Markov simulations...")
        plot_markov_combined()

    if RUN_STATIONARY:
        print("Running stationary state analysis...")
        plot_stationary_state()

    if RUN_STATIONARY_PARAMETER_VARIATION:
        print("Running parameter variation analysis...")
        plot_stationary_vs_k_parameter_variation()

    if RUN_HEATMAP:
        print("Running heatmap analysis...")
        plot_heatmap()

    if RUN_ANIMATION:
        print("Running animation...")
        animate_pdh_simulation()


if __name__ == "__main__":
    main()