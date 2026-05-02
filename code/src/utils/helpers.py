"""
General helper functions for simulations, random seeds and file output.
"""

import os
import time
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from config import (
    ALPHA,
    BETA,
    MU,
    SAVE_FIGURES,
    OUTPUT_DIR,
)


# =========================================================
# State utilities
# =========================================================

def count_states(state):
    """
    Count the number of nodes in each PDH state.

    Parameters
    ----------
    state : dict
        Dictionary of the form {node: state}.

    Returns
    -------
    dict
        Counts for states P, D and H.
    """

    counts = {"P": 0, "D": 0, "H": 0}

    for node_state in state.values():
        counts[node_state] += 1

    return counts


def max_state_counts(state):
    """
    Return the largest state count in a given configuration.
    """

    return max(count_states(state).values())


# =========================================================
# Random seed utilities
# =========================================================

def generate_time_seed():
    """
    Generate a time-based random seed.
    """

    return int(time.time() * 1000) % (2**32)


def set_global_seed(seed=None):
    """
    Set the global random seed for both Python's random module and NumPy.

    If no seed is provided, a time-based seed is generated.

    Returns
    -------
    int
        Seed used.
    """

    if seed is None:
        seed = generate_time_seed()

    random.seed(seed)
    np.random.seed(seed)

    return seed


# =========================================================
# Output utilities
# =========================================================

def format_float_for_filename(value, decimals=3):
    """
    Convert a float into a filename-safe string.

    Example
    -------
    0.8 -> 0p800
    """

    return f"{value:.{decimals}f}".replace(".", "p")


def get_parameter_tag(alpha=None, beta=None, mu=None):
    """
    Generate a folder tag from PDH parameters.
    """

    if alpha is None:
        alpha = ALPHA

    if beta is None:
        beta = BETA

    if mu is None:
        mu = MU

    alpha_str = format_float_for_filename(alpha)
    beta_str = format_float_for_filename(beta)
    mu_str = format_float_for_filename(mu)

    return f"alpha_{alpha_str}_beta_{beta_str}_mu_{mu_str}"


def get_output_folder(alpha=None, beta=None, mu=None):
    """
    Return the output folder for a given parameter set.
    """

    parameter_tag = get_parameter_tag(alpha=alpha, beta=beta, mu=mu)
    output_folder = os.path.join(OUTPUT_DIR, parameter_tag)

    os.makedirs(output_folder, exist_ok=True)

    return output_folder


def save_current_figure(fig_name, alpha=None, beta=None, mu=None, add_timestamp=False):
    """
    Save the current Matplotlib figure into the output folder.

    Figures are stored in:

        outputs/alpha_xxx_beta_xxx_mu_xxx/figure_name.png
    """

    if not SAVE_FIGURES:
        return None

    output_folder = get_output_folder(alpha=alpha, beta=beta, mu=mu)

    filename = fig_name

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename += f"_{timestamp}"

    filepath = os.path.join(output_folder, filename + ".png")

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")

    return filepath


def get_animation_path(animation_name, alpha=None, beta=None, mu=None):
    """
    Return a GIF output path inside the parameter-specific folder.
    """

    output_folder = get_output_folder(alpha=alpha, beta=beta, mu=mu)

    if not animation_name.endswith(".gif"):
        animation_name += ".gif"

    return os.path.join(output_folder, animation_name)