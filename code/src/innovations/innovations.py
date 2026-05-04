"""
Innovation module for the PDH bacterial competition model.

This module contains possible model extensions motivated by biological and
network-based considerations:

1. Node-dependent parameters.
2. Mobility / diffusion on the network.
3. Dynamic networks.
4. Structural heterogeneity through scale-free networks.

The goal is not to replace the base PDH model, but to provide controlled
extensions that can be compared against the original dynamics.
"""

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import (
    N_NODES,
    ALPHA,
    BETA,
    MU,
    INITIAL_PROPORTIONS,
    HETEROGENEITY_STRENGTH,
    MOBILITY_RATES,
    REWIRING_RATES,
    INNOVATION_K_VALUES,
    INNOVATION_T_MARKOV,
    INNOVATION_N_REALIZATIONS,
    INNOVATION_TAIL,
    RUN_HETEROGENEOUS_PARAMETERS,
    RUN_MOBILITY,
    RUN_DYNAMIC_NETWORK,
    RUN_SCALE_FREE_COMPARISON,
)

from src.graphs.generate_graphs import generate_er_graph_from_k
from src.simulation.simulation_model import markov_pdh_dynamics
from src.utils.helpers import save_current_figure
from src.utils.metrics import degree_heterogeneity_factor


# =========================================================
# General helpers
# =========================================================

def _clip_probabilities(values):
    """
    Clip an array of probabilities to the valid interval [0, 1].
    """

    return np.clip(values, 0.0, 1.0)


def _stationary_value(time_series, tail):
    """
    Estimate the stationary value as the mean over the last `tail` steps.
    """

    return np.mean(time_series[-tail:])


def _average_degree(graph):
    """
    Compute the average degree of a graph.
    """

    return np.mean([degree for _, degree in graph.degree()])


# =========================================================
# 1) Node-dependent heterogeneous parameters
# =========================================================

def generate_heterogeneous_parameters(
    n_nodes,
    alpha=ALPHA,
    beta=BETA,
    mu=MU,
    heterogeneity_strength=HETEROGENEITY_STRENGTH,
):
    """
    Generate node-dependent alpha_i, beta_i and mu_i values.

    Parameters are sampled around their base values using a uniform noise
    window controlled by `heterogeneity_strength`.

    Example:
        alpha_i in alpha * [1 - eps, 1 + eps]

    This represents spatial or biological heterogeneity:
    - some predators are more efficient than others;
    - some prey colonize faster than others;
    - some regions are harsher for predators.
    """

    eps = heterogeneity_strength

    alpha_i = alpha * (1.0 + eps * np.random.uniform(-1.0, 1.0, n_nodes))
    beta_i = beta * (1.0 + eps * np.random.uniform(-1.0, 1.0, n_nodes))
    mu_i = mu * (1.0 + eps * np.random.uniform(-1.0, 1.0, n_nodes))

    return (
        _clip_probabilities(alpha_i),
        _clip_probabilities(beta_i),
        _clip_probabilities(mu_i),
    )


def markov_pdh_dynamics_heterogeneous(
    adjacency_matrix,
    alpha_i,
    beta_i,
    mu_i,
    P0,
    D0,
    H0,
    T,
):
    """
    Markov PDH dynamics with node-dependent parameters.

    Convention:
        alpha_i controls predator efficiency at node i.
        beta_i controls prey colonization efficiency at node i.
        mu_i controls predator mortality at node i.

    Predation and colonization are computed as neighbor-driven processes:
        - a prey at i is attacked by predator neighbors j with efficiency alpha_j;
        - an empty site i is colonized by prey neighbors j with efficiency beta_j.
    """

    A = adjacency_matrix
    n_nodes = A.shape[0]

    p = np.full(n_nodes, P0, dtype=float)
    d = np.full(n_nodes, D0, dtype=float)
    h = np.full(n_nodes, H0, dtype=float)

    P_mean = [np.mean(p)]
    D_mean = [np.mean(d)]
    H_mean = [np.mean(h)]

    alpha_neighbors = alpha_i[np.newaxis, :]
    beta_neighbors = beta_i[np.newaxis, :]

    for _ in range(T):

        predation_probability = 1.0 - np.prod(
            1.0 - A * alpha_neighbors * d[np.newaxis, :],
            axis=1,
        )

        colonization_probability = 1.0 - np.prod(
            1.0 - A * beta_neighbors * p[np.newaxis, :],
            axis=1,
        )

        p_new = p * (1.0 - predation_probability) + h * colonization_probability
        d_new = d * (1.0 - mu_i) + p * predation_probability
        h_new = h * (1.0 - colonization_probability) + mu_i * d

        total = p_new + d_new + h_new
        total[total == 0] = 1.0

        p = p_new / total
        d = d_new / total
        h = h_new / total

        P_mean.append(np.mean(p))
        D_mean.append(np.mean(d))
        H_mean.append(np.mean(h))

    return np.array(P_mean), np.array(D_mean), np.array(H_mean)


def stationary_heterogeneous_vs_degree(
    k_values=INNOVATION_K_VALUES,
    n_realizations=INNOVATION_N_REALIZATIONS,
):
    """
    Compare homogeneous and heterogeneous parameter dynamics on ER networks.
    """

    homogeneous_D = []
    heterogeneous_D = []

    homogeneous_std = []
    heterogeneous_std = []

    P0, D0, H0 = INITIAL_PROPORTIONS

    for k_avg in k_values:

        hom_runs = []
        het_runs = []

        for _ in range(n_realizations):
            _, A = generate_er_graph_from_k(N_NODES, k_avg)

            _, D_hom, _ = markov_pdh_dynamics(
                adjacency_matrix=A,
                beta=BETA,
                alpha=ALPHA,
                mu=MU,
                P0=P0,
                D0=D0,
                H0=H0,
                T=INNOVATION_T_MARKOV,
            )

            alpha_i, beta_i, mu_i = generate_heterogeneous_parameters(N_NODES)

            _, D_het, _ = markov_pdh_dynamics_heterogeneous(
                adjacency_matrix=A,
                alpha_i=alpha_i,
                beta_i=beta_i,
                mu_i=mu_i,
                P0=P0,
                D0=D0,
                H0=H0,
                T=INNOVATION_T_MARKOV,
            )

            hom_runs.append(_stationary_value(D_hom, INNOVATION_TAIL))
            het_runs.append(_stationary_value(D_het, INNOVATION_TAIL))

        homogeneous_D.append(np.mean(hom_runs))
        heterogeneous_D.append(np.mean(het_runs))

        homogeneous_std.append(np.std(hom_runs))
        heterogeneous_std.append(np.std(het_runs))

    return (
        np.array(homogeneous_D),
        np.array(heterogeneous_D),
        np.array(homogeneous_std),
        np.array(heterogeneous_std),
    )


def plot_heterogeneous_parameters():
    """
    Plot the effect of node-dependent parameters on stationary predator density.
    """

    k_values = np.array(INNOVATION_K_VALUES)

    D_hom, D_het, D_hom_std, D_het_std = stationary_heterogeneous_vs_degree(
        k_values=k_values
    )

    k_crit = MU / ALPHA

    plt.figure("Innovation - heterogeneous parameters", figsize=(10, 6))

    plt.errorbar(
        k_values,
        D_hom,
        yerr=D_hom_std,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="Homogeneous parameters",
    )

    plt.errorbar(
        k_values,
        D_het,
        yerr=D_het_std,
        marker="s",
        linewidth=2.2,
        capsize=4,
        label="Node-dependent parameters",
    )

    plt.axvline(
        k_crit,
        linestyle="--",
        linewidth=2.4,
        alpha=0.8,
        label=rf"$k_c=\mu/\alpha={k_crit:.2f}$",
    )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title("Effect of heterogeneous node-dependent parameters", fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend()
    plt.tight_layout()

    save_current_figure("innovation_heterogeneous_parameters")

    plt.show()


# =========================================================
# 2) Mobility / diffusion on the network
# =========================================================

def _diffuse_state_probabilities(A, p, d, h, mobility_rate):
    """
    Apply diffusion to state probabilities.

    Diffusion is modeled as probability mass moving from each node to its
    neighbors with probability `mobility_rate`.

    The transition matrix is column-stochastic:
        T_ij = A_ij / k_j

    so that probability mass is redistributed from node j to its neighbors i.

    Note:
        This exploratory version diffuses P, D and H probabilities. If a more
        biological implementation is desired, one may diffuse only P and D and
        then reconstruct H = 1 - P - D.
    """

    if mobility_rate <= 0.0:
        return p, d, h

    degrees = A.sum(axis=0)
    degrees[degrees == 0] = 1.0

    transition_matrix = A / degrees[np.newaxis, :]

    p_diffused = (1.0 - mobility_rate) * p + mobility_rate * (transition_matrix @ p)
    d_diffused = (1.0 - mobility_rate) * d + mobility_rate * (transition_matrix @ d)
    h_diffused = (1.0 - mobility_rate) * h + mobility_rate * (transition_matrix @ h)

    total = p_diffused + d_diffused + h_diffused
    total[total == 0] = 1.0

    return (
        p_diffused / total,
        d_diffused / total,
        h_diffused / total,
    )


def markov_pdh_dynamics_with_mobility(
    adjacency_matrix,
    beta,
    alpha,
    mu,
    P0,
    D0,
    H0,
    T,
    mobility_rate,
):
    """
    Markov PDH dynamics with diffusion after each reaction step.

    The algorithm is:
        1. Apply PDH local reactions.
        2. Apply mobility/diffusion of P, D and H probabilities.
        3. Normalize probabilities at each node.
    """

    A = adjacency_matrix
    n_nodes = A.shape[0]

    p = np.full(n_nodes, P0, dtype=float)
    d = np.full(n_nodes, D0, dtype=float)
    h = np.full(n_nodes, H0, dtype=float)

    P_mean = [np.mean(p)]
    D_mean = [np.mean(d)]
    H_mean = [np.mean(h)]

    for _ in range(T):

        predation_probability = 1.0 - np.prod(
            1.0 - alpha * A * d[np.newaxis, :],
            axis=1,
        )

        colonization_probability = 1.0 - np.prod(
            1.0 - beta * A * p[np.newaxis, :],
            axis=1,
        )

        p_new = p * (1.0 - predation_probability) + h * colonization_probability
        d_new = d * (1.0 - mu) + p * predation_probability
        h_new = h * (1.0 - colonization_probability) + mu * d

        total = p_new + d_new + h_new
        total[total == 0] = 1.0

        p = p_new / total
        d = d_new / total
        h = h_new / total

        p, d, h = _diffuse_state_probabilities(
            A=A,
            p=p,
            d=d,
            h=h,
            mobility_rate=mobility_rate,
        )

        P_mean.append(np.mean(p))
        D_mean.append(np.mean(d))
        H_mean.append(np.mean(h))

    return np.array(P_mean), np.array(D_mean), np.array(H_mean)


def plot_mobility_effect():
    """
    Plot the effect of mobility on stationary predator density.
    """

    k_values = np.array(INNOVATION_K_VALUES)
    P0, D0, H0 = INITIAL_PROPORTIONS

    plt.figure("Innovation - mobility", figsize=(10, 6))

    for mobility_rate in MOBILITY_RATES:

        D_values = []
        D_std = []

        for k_avg in k_values:

            D_runs = []

            for _ in range(INNOVATION_N_REALIZATIONS):
                _, A = generate_er_graph_from_k(N_NODES, k_avg)

                _, D, _ = markov_pdh_dynamics_with_mobility(
                    adjacency_matrix=A,
                    beta=BETA,
                    alpha=ALPHA,
                    mu=MU,
                    P0=P0,
                    D0=D0,
                    H0=H0,
                    T=INNOVATION_T_MARKOV,
                    mobility_rate=mobility_rate,
                )

                D_runs.append(_stationary_value(D, INNOVATION_TAIL))

            D_values.append(np.mean(D_runs))
            D_std.append(np.std(D_runs))

        plt.errorbar(
            k_values,
            D_values,
            yerr=D_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=rf"mobility = {mobility_rate}",
        )

    k_crit = MU / ALPHA

    plt.axvline(
        k_crit,
        linestyle="--",
        color="black",
        linewidth=2.4,
        alpha=0.75,
        label=rf"$k_c=\mu/\alpha={k_crit:.2f}$",
    )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title("Effect of mobility on predator survival", fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend()
    plt.tight_layout()

    save_current_figure("innovation_mobility")

    plt.show()


# =========================================================
# 3) Dynamic networks
# =========================================================

def rewire_network_er_like(graph, rewiring_rate):
    """
    Rewire an undirected graph in an ER-like way.

    Each existing edge is removed with probability `rewiring_rate` and replaced
    by a random edge between two previously unconnected nodes.

    This keeps the number of edges approximately constant.
    """

    if rewiring_rate <= 0.0:
        return graph

    new_graph = graph.copy()
    nodes = list(new_graph.nodes())
    edges = list(new_graph.edges())

    for edge in edges:
        if random.random() < rewiring_rate:

            u, v = edge

            if new_graph.has_edge(u, v):
                new_graph.remove_edge(u, v)

            for _ in range(50):
                a, b = random.sample(nodes, 2)

                if a != b and not new_graph.has_edge(a, b):
                    new_graph.add_edge(a, b)
                    break

    return new_graph


def markov_pdh_dynamics_dynamic_network(
    graph,
    beta,
    alpha,
    mu,
    P0,
    D0,
    H0,
    T,
    rewiring_rate,
):
    """
    Markov PDH dynamics on a time-dependent network G(t).

    At each time step:
        1. Run one Markov reaction step.
        2. Rewire part of the network.
    """

    n_nodes = graph.number_of_nodes()

    p = np.full(n_nodes, P0, dtype=float)
    d = np.full(n_nodes, D0, dtype=float)
    h = np.full(n_nodes, H0, dtype=float)

    P_mean = [np.mean(p)]
    D_mean = [np.mean(d)]
    H_mean = [np.mean(h)]

    current_graph = graph.copy()

    for _ in range(T):

        A = nx.to_numpy_array(current_graph, dtype=float)

        predation_probability = 1.0 - np.prod(
            1.0 - alpha * A * d[np.newaxis, :],
            axis=1,
        )

        colonization_probability = 1.0 - np.prod(
            1.0 - beta * A * p[np.newaxis, :],
            axis=1,
        )

        p_new = p * (1.0 - predation_probability) + h * colonization_probability
        d_new = d * (1.0 - mu) + p * predation_probability
        h_new = h * (1.0 - colonization_probability) + mu * d

        total = p_new + d_new + h_new
        total[total == 0] = 1.0

        p = p_new / total
        d = d_new / total
        h = h_new / total

        current_graph = rewire_network_er_like(
            graph=current_graph,
            rewiring_rate=rewiring_rate,
        )

        P_mean.append(np.mean(p))
        D_mean.append(np.mean(d))
        H_mean.append(np.mean(h))

    return np.array(P_mean), np.array(D_mean), np.array(H_mean)


def plot_dynamic_network_effect():
    """
    Plot the effect of network rewiring on stationary predator density.
    """

    k_values = np.array(INNOVATION_K_VALUES)
    P0, D0, H0 = INITIAL_PROPORTIONS

    plt.figure("Innovation - dynamic network", figsize=(10, 6))

    for rewiring_rate in REWIRING_RATES:

        D_values = []
        D_std = []

        for k_avg in k_values:

            D_runs = []

            for _ in range(INNOVATION_N_REALIZATIONS):
                graph, _ = generate_er_graph_from_k(N_NODES, k_avg)

                _, D, _ = markov_pdh_dynamics_dynamic_network(
                    graph=graph,
                    beta=BETA,
                    alpha=ALPHA,
                    mu=MU,
                    P0=P0,
                    D0=D0,
                    H0=H0,
                    T=INNOVATION_T_MARKOV,
                    rewiring_rate=rewiring_rate,
                )

                D_runs.append(_stationary_value(D, INNOVATION_TAIL))

            D_values.append(np.mean(D_runs))
            D_std.append(np.std(D_runs))

        plt.errorbar(
            k_values,
            D_values,
            yerr=D_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=rf"rewiring = {rewiring_rate}",
        )

    k_crit = MU / ALPHA

    plt.axvline(
        k_crit,
        linestyle="--",
        color="black",
        linewidth=2.4,
        alpha=0.75,
        label=rf"$k_c=\mu/\alpha={k_crit:.2f}$",
    )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title("Effect of dynamic network rewiring", fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend()
    plt.tight_layout()

    save_current_figure("innovation_dynamic_network")

    plt.show()


# =========================================================
# 4) Structural heterogeneity: scale-free networks
# =========================================================

def generate_barabasi_graph_from_target_k(n_nodes, k_avg):
    """
    Generate a Barabási-Albert graph with an approximate target average degree.

    For BA networks:
        <k> approximately 2m

    Therefore:
        m approximately <k> / 2
    """

    m = max(1, int(round(k_avg / 2)))
    graph = nx.barabasi_albert_graph(n_nodes, m)
    adjacency_matrix = nx.to_numpy_array(graph, dtype=float)

    return graph, adjacency_matrix


def stationary_on_graph_generator(
    graph_generator,
    k_values,
    n_realizations=INNOVATION_N_REALIZATIONS,
):
    """
    Compute stationary predator density for a generic graph generator.

    This function returns the real average degree, because the target average
    degree may differ slightly from the realized one, especially for finite
    networks.
    """

    P0, D0, H0 = INITIAL_PROPORTIONS

    real_k_values = []
    D_values = []
    D_std = []

    for k_avg in k_values:

        k_runs = []
        D_runs = []

        for _ in range(n_realizations):
            graph, A = graph_generator(N_NODES, k_avg)

            _, D, _ = markov_pdh_dynamics(
                adjacency_matrix=A,
                beta=BETA,
                alpha=ALPHA,
                mu=MU,
                P0=P0,
                D0=D0,
                H0=H0,
                T=INNOVATION_T_MARKOV,
            )

            k_runs.append(_average_degree(graph))
            D_runs.append(_stationary_value(D, INNOVATION_TAIL))

        real_k_values.append(np.mean(k_runs))
        D_values.append(np.mean(D_runs))
        D_std.append(np.std(D_runs))

    return np.array(real_k_values), np.array(D_values), np.array(D_std)


def stationary_vs_degree_heterogeneity_factor(
    graph_generator,
    k_values,
    n_realizations=INNOVATION_N_REALIZATIONS,
):
    """
    Compute stationary predator density as a function of the structural
    heterogeneity factor kappa = <k^2> / <k>.

    This is useful for comparing homogeneous and heterogeneous network
    topologies beyond their average degree.
    """

    P0, D0, H0 = INITIAL_PROPORTIONS

    kappa_values = []
    D_values = []
    D_std = []

    for k_avg in k_values:

        kappa_runs = []
        D_runs = []

        for _ in range(n_realizations):
            graph, A = graph_generator(N_NODES, k_avg)

            _, D, _ = markov_pdh_dynamics(
                adjacency_matrix=A,
                beta=BETA,
                alpha=ALPHA,
                mu=MU,
                P0=P0,
                D0=D0,
                H0=H0,
                T=INNOVATION_T_MARKOV,
            )

            kappa_runs.append(degree_heterogeneity_factor(graph))
            D_runs.append(_stationary_value(D, INNOVATION_TAIL))

        kappa_values.append(np.mean(kappa_runs))
        D_values.append(np.mean(D_runs))
        D_std.append(np.std(D_runs))

    return np.array(kappa_values), np.array(D_values), np.array(D_std)


def plot_scale_free_comparison():
    """
    Compare stationary predator density in ER and Barabási-Albert networks.
    """

    k_values = np.array(INNOVATION_K_VALUES)

    k_er, D_er, D_er_std = stationary_on_graph_generator(
        graph_generator=generate_er_graph_from_k,
        k_values=k_values,
    )

    k_ba, D_ba, D_ba_std = stationary_on_graph_generator(
        graph_generator=generate_barabasi_graph_from_target_k,
        k_values=k_values,
    )

    k_crit = MU / ALPHA

    plt.figure("Innovation - ER vs scale-free", figsize=(10, 6))

    plt.errorbar(
        k_er,
        D_er,
        yerr=D_er_std,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="Erdős-Rényi",
    )

    plt.errorbar(
        k_ba,
        D_ba,
        yerr=D_ba_std,
        marker="s",
        linewidth=2.2,
        capsize=4,
        label="Barabási-Albert",
    )

    plt.axvline(
        k_crit,
        linestyle="--",
        color="black",
        linewidth=2.4,
        alpha=0.75,
        label=rf"$k_c=\mu/\alpha={k_crit:.2f}$",
    )

    plt.xlabel(r"Real average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title(
        "Effect of structural heterogeneity: ER vs scale-free networks",
        fontsize=15,
    )
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend()
    plt.tight_layout()

    save_current_figure("innovation_scale_free_comparison")

    plt.show()


def plot_scale_free_heterogeneity_factor_comparison():
    """
    Compare ER and Barabási-Albert networks using the structural heterogeneity
    factor kappa = <k^2> / <k> as the x-axis.

    This plot complements the usual D_inf vs <k> representation. While <k>
    controls the mean-field threshold, kappa captures the effect of hubs and
    degree heterogeneity.
    """

    k_values = np.array(INNOVATION_K_VALUES)

    kappa_er, D_er, D_er_std = stationary_vs_degree_heterogeneity_factor(
        graph_generator=generate_er_graph_from_k,
        k_values=k_values,
    )

    kappa_ba, D_ba, D_ba_std = stationary_vs_degree_heterogeneity_factor(
        graph_generator=generate_barabasi_graph_from_target_k,
        k_values=k_values,
    )

    plt.figure("Innovation - structural heterogeneity factor", figsize=(10, 6))

    plt.errorbar(
        kappa_er,
        D_er,
        yerr=D_er_std,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="Erdős-Rényi",
    )

    plt.errorbar(
        kappa_ba,
        D_ba,
        yerr=D_ba_std,
        marker="s",
        linewidth=2.2,
        capsize=4,
        label="Barabási-Albert",
    )

    plt.xlabel(
        r"Degree heterogeneity factor $\kappa=\langle k^2\rangle/\langle k\rangle$",
        fontsize=13,
    )
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title("Predator survival vs structural heterogeneity", fontsize=15)

    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend()
    plt.tight_layout()

    save_current_figure("innovation_structural_heterogeneity_factor")

    plt.show()


# =========================================================
# Master function
# =========================================================

def run_innovation_analysis():
    """
    Run all selected innovation analyses according to config.py.
    """

    if RUN_HETEROGENEOUS_PARAMETERS:
        print("Running innovation: heterogeneous parameters...")
        plot_heterogeneous_parameters()

    if RUN_MOBILITY:
        print("Running innovation: mobility / diffusion...")
        plot_mobility_effect()

    if RUN_DYNAMIC_NETWORK:
        print("Running innovation: dynamic network...")
        plot_dynamic_network_effect()

    if RUN_SCALE_FREE_COMPARISON:
        print("Running innovation: ER vs scale-free networks...")
        plot_scale_free_comparison()

        print("Running innovation: structural heterogeneity factor comparison...")
        plot_scale_free_heterogeneity_factor_comparison()