"""
Simulation routines for the PDH bacterial competition model.

States:
    P: prey bacteria
    D: predator bacteria
    H: empty site

Processes:
    P + D -> 2D with probability alpha
    P + H -> 2P with probability beta
    D -> H     with probability mu
"""

import random
import numpy as np

from config import *
import src.graphs.generate_graphs


def simulate_pdh(graph, pdh_parameters, initial_proportions, steps):
    """
    Run a stochastic microscopic simulation of the PDH model on a network.

    Parameters
    ----------
    graph : networkx.Graph
        Network where each node represents a possible bacterial site.
    pdh_parameters : list or tuple
        Model parameters [alpha, beta, mu].
    initial_proportions : list or tuple
        Initial fractions [P0, D0, H0].
    steps : int
        Number of time steps.

    Returns
    -------
    list[dict]
        Simulation history. Each element is a dictionary {node: state}.
    """

    p_proportion, d_proportion, _ = initial_proportions
    alpha, beta, mu = pdh_parameters

    state = {}

    for node in graph.nodes():
        rand = random.random()

        if rand < p_proportion:
            state[node] = "P"
        elif rand < p_proportion + d_proportion:
            state[node] = "D"
        else:
            state[node] = "H"

    history = []

    for _ in range(steps):
        new_state = state.copy()

        for node in graph.nodes():

            if state[node] == "H":
                for neighbor in graph.neighbors(node):
                    if state[neighbor] == "P" and random.random() < beta:
                        new_state[node] = "P"
                        break

            elif state[node] == "P":
                for neighbor in graph.neighbors(node):
                    if state[neighbor] == "D" and random.random() < alpha:
                        new_state[node] = "D"
                        break

            elif state[node] == "D":
                if random.random() < mu:
                    new_state[node] = "H"

        state = new_state
        history.append(state.copy())

    return history


def markov_pdh_dynamics(adjacency_matrix, beta, alpha, mu, P0, D0, H0, T):
    """
    Run the Markov approximation of the PDH model on a fixed network.

    Each node is assigned probabilities of being in states P, D and H.
    The dynamics are updated using neighbor-dependent transition probabilities.

    Parameters
    ----------
    adjacency_matrix : numpy.ndarray
        Network adjacency matrix.
    beta : float
        Colonization probability, P + H -> 2P.
    alpha : float
        Predation probability, P + D -> 2D.
    mu : float
        Predator mortality probability, D -> H.
    P0, D0, H0 : float
        Initial probabilities for each state.
    T : int
        Number of time steps.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Mean fractions of P, D and H over time.
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
            axis=1
        )

        colonization_probability = 1.0 - np.prod(
            1.0 - beta * A * p[np.newaxis, :],
            axis=1
        )

        p_new = p * (1.0 - predation_probability) + h * colonization_probability
        d_new = d * (1.0 - mu) + p * predation_probability
        h_new = h * (1.0 - colonization_probability) + mu * d

        total = p_new + d_new + h_new
        total[total == 0] = 1.0

        p = p_new / total
        d = d_new / total
        h = h_new / total

        P_mean.append(np.mean(p))
        D_mean.append(np.mean(d))
        H_mean.append(np.mean(h))

    return np.array(P_mean), np.array(D_mean), np.array(H_mean)


def stationary_state_vs_degree(
    generate_er_graph_func,
    n_nodes,
    k_values,
    beta,
    alpha,
    mu,
    P0,
    D0,
    H0,
    T,
    n_realizations=5,
    tail=20,
):
    """
    Compute stationary PDH fractions as a function of the mean degree.

    For each value of <k>, several ER network realizations are generated and
    averaged. The stationary state is estimated as the mean over the last
    `tail` time steps.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Mean and standard deviation of stationary P, D and H fractions.
    """

    P_inf_list = []
    D_inf_list = []
    H_inf_list = []

    P_std_list = []
    D_std_list = []
    H_std_list = []

    for k_avg in k_values:
        P_inf_runs = []
        D_inf_runs = []
        H_inf_runs = []

        for _ in range(n_realizations):
            _, adjacency_matrix = src.graphs.generate_graphs.generate_graph_combined(GRAPH_TYPE, k_avg)

            P, D, H = markov_pdh_dynamics(
                adjacency_matrix=adjacency_matrix,
                beta=beta,
                alpha=alpha,
                mu=mu,
                P0=P0,
                D0=D0,
                H0=H0,
                T=T,
            )

            P_inf_runs.append(np.mean(P[-tail:]))
            D_inf_runs.append(np.mean(D[-tail:]))
            H_inf_runs.append(np.mean(H[-tail:]))

        P_inf_list.append(np.mean(P_inf_runs))
        D_inf_list.append(np.mean(D_inf_runs))
        H_inf_list.append(np.mean(H_inf_runs))

        P_std_list.append(np.std(P_inf_runs))
        D_std_list.append(np.std(D_inf_runs))
        H_std_list.append(np.std(H_inf_runs))

    return (
        np.array(P_inf_list),
        np.array(D_inf_list),
        np.array(H_inf_list),
        np.array(P_std_list),
        np.array(D_std_list),
        np.array(H_std_list),
    )