"""
Visualization routines for the PDH bacterial competition model.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

from config import (
    N_NODES,
    STEPS,
    GRAPH_TYPE,
    ALPHA,
    BETA,
    MU,
    PDH_PARAMS,
    INITIAL_PROPORTIONS,
)

from src.simulation.simulation_model import (
    simulate_pdh,
    markov_pdh_dynamics,
    stationary_state_vs_degree,
)

from src.graphs.generate_graphs import (
    generate_graph,
    generate_er_graph_from_k,
    generate_graph_combined,
)

from src.utils.helpers import (
    count_states,
    save_current_figure,
    get_animation_path,
    format_float_for_filename,
)


# =========================================================
# General simulation helpers
# =========================================================

def run_stochastic_simulation(graph_type):
    """
    Run the stochastic PDH simulation on a selected graph type.
    """

    graph = generate_graph(graph_type)

    history = simulate_pdh(
        graph=graph,
        pdh_parameters=PDH_PARAMS,
        initial_proportions=INITIAL_PROPORTIONS,
        steps=STEPS,
    )

    P_list = []
    D_list = []
    H_list = []

    for state in history:
        counts = count_states(state)
        P_list.append(counts["P"])
        D_list.append(counts["D"])
        H_list.append(counts["H"])

    return P_list, D_list, H_list


# =========================================================
# 1) Original stochastic simulation
# =========================================================

def plot_original_simulation():
    """
    Plot the stochastic PDH simulation on the configured graph type.
    """

    P, D, H = run_stochastic_simulation(GRAPH_TYPE)

    plt.figure("Original stochastic simulation", figsize=(10, 6))
    plt.plot(P, label="Prey")
    plt.plot(D, label="Predators")
    plt.plot(H, label="Empty sites")

    plt.xlabel("Time")
    plt.ylabel("Number of nodes")
    plt.title(f"Stochastic PDH simulation on a {GRAPH_TYPE} graph")
    plt.legend()
    plt.grid(True, alpha=0.65)
    plt.tight_layout()

    save_current_figure(f"original_simulation_{GRAPH_TYPE}")

    plt.show()


# =========================================================
# 2) Individual Markov dynamics
# =========================================================

def plot_markov_individual():
    """
    Plot Markov dynamics separately for several average degrees.
    """

    k_values = [2, 4, 6, 8, 10]
    P0, D0, H0 = INITIAL_PROPORTIONS
    T_markov = 200

    for k_avg in k_values:
        _, adjacency_matrix = generate_graph_combined(GRAPH_TYPE, k_avg)

        P_markov, D_markov, H_markov = markov_pdh_dynamics(
            adjacency_matrix=adjacency_matrix,
            beta=BETA,
            alpha=ALPHA,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_markov,
        )

        plt.figure(f"Markov k={k_avg}", figsize=(10, 6))
        plt.plot(P_markov, label="Prey")
        plt.plot(D_markov, label="Predators")
        plt.plot(H_markov, label="Empty sites")

        plt.title(rf"Markov PDH dynamics for $\langle k \rangle={k_avg}$")
        plt.xlabel("Time")
        plt.ylabel("Fraction")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.65)
        plt.tight_layout()

        k_tag = format_float_for_filename(k_avg)
        save_current_figure(f"markov_individual_k_{k_tag}")

        plt.show()


# =========================================================
# 3) Combined Markov dynamics
# =========================================================

def plot_markov_combined():
    """
    Plot combined Markov dynamics for several average degrees.
    """

    k_values = [2, 4, 6, 8, 10]
    P0, D0, H0 = INITIAL_PROPORTIONS
    T_markov = 200

    results_P = {}
    results_D = {}
    results_H = {}

    for k_avg in k_values:
        _, adjacency_matrix = generate_graph_combined(GRAPH_TYPE, k_avg)

        P_markov, D_markov, H_markov = markov_pdh_dynamics(
            adjacency_matrix=adjacency_matrix,
            beta=BETA,
            alpha=ALPHA,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_markov,
        )

        results_P[k_avg] = P_markov
        results_D[k_avg] = D_markov
        results_H[k_avg] = H_markov

    plot_data = [
        ("prey", "Prey", results_P),
        ("predators", "Predators", results_D),
        ("empty_sites", "Empty sites", results_H),
    ]

    for filename_tag, state_label, results in plot_data:
        plt.figure(f"Combined Markov - {state_label}", figsize=(10, 6))

        for k_avg in k_values:
            plt.plot(
                results[k_avg],
                label=rf"$\langle k \rangle={k_avg}$",
            )

        plt.xlabel("Time")
        plt.ylabel("Fraction")
        plt.title(rf"{state_label} dynamics for different $\langle k \rangle$")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.65)
        plt.tight_layout()

        save_current_figure(f"markov_combined_{filename_tag}")

        plt.show()


# =========================================================
# 4) Stationary state
# =========================================================

def plot_stationary_state():
    """
    Plot stationary fractions as a function of the average degree.
    """

    k_values = np.array([1, 2, 3, 3.5, 4, 4.5, 5, 6, 8, 10])

    P0, D0, H0 = INITIAL_PROPORTIONS
    T_markov = 300
    n_realizations = 5
    tail = 30

    P_inf, D_inf, H_inf, P_std, D_std, H_std = stationary_state_vs_degree(
        generate_er_graph_func= generate_graph(GRAPH_TYPE),
        n_nodes=N_NODES,
        k_values=k_values,
        beta=BETA,
        alpha=ALPHA,
        mu=MU,
        P0=P0,
        D0=D0,
        H0=H0,
        T=T_markov, 
        n_realizations=n_realizations,
        tail=tail,
    )

    k_crit = MU / ALPHA

    plt.figure("Stationary state", figsize=(10, 6))

    plt.errorbar(
        k_values,
        P_inf,
        yerr=P_std,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="Prey",
    )

    plt.errorbar(
        k_values,
        D_inf,
        yerr=D_std,
        marker="s",
        linewidth=2.2,
        capsize=4,
        label="Predators",
    )

    plt.errorbar(
        k_values,
        H_inf,
        yerr=H_std,
        marker="^",
        linewidth=2.2,
        capsize=4,
        label="Empty sites",
    )

    plt.axvline(
        x=k_crit,
        linestyle="--",
        linewidth=2.5,
        alpha=0.85,
        label=rf"$k_c=\mu/\alpha={k_crit:.2f}$",
    )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel("Stationary fraction", fontsize=13)
    plt.title("Stationary state of the PDH model", fontsize=15)

    plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.65)
    plt.legend(fontsize=11)
    plt.tight_layout()

    save_current_figure("stationary_state")

    plt.show()


# =========================================================
# 5) Parameter variation
# =========================================================

def plot_stationary_vs_k_parameter_variation():
    """
    Study how stationary predator density changes under parameter variation.
    """

    k_values = np.array([
        0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4,
        4.5, 5, 6, 7, 8, 9, 10,
    ])

    P0, D0, H0 = INITIAL_PROPORTIONS
    T_markov = 300
    n_realizations = 5
    tail = 30

    _plot_mu_variation(k_values, P0, D0, H0, T_markov, n_realizations, tail)
    _plot_beta_variation(k_values, P0, D0, H0, T_markov, n_realizations, tail)
    _plot_alpha_variation(k_values, P0, D0, H0, T_markov, n_realizations, tail)


def _plot_mu_variation(k_values, P0, D0, H0, T_markov, n_realizations, tail):
    """
    Plot stationary predator density for several mortality values.
    """

    mu_values = [0.2, 0.5, 0.8]

    plt.figure("Mortality variation", figsize=(10, 6))

    for mu in mu_values:
        _, D_inf, _, _, D_std, _ = stationary_state_vs_degree(
            generate_er_graph_func=generate_er_graph_from_k,
            n_nodes=N_NODES,
            k_values=k_values,
            beta=BETA,
            alpha=ALPHA,
            mu=mu,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_markov,
            n_realizations=n_realizations,
            tail=tail,
        )

        curve = plt.errorbar(
            k_values,
            D_inf,
            yerr=D_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=rf"$\mu={mu}$",
        )

        color = curve[0].get_color()
        k_crit = mu / ALPHA

        plt.axvline(
            x=k_crit,
            linestyle="--",
            linewidth=2.3,
            color=color,
            alpha=0.55,
            label=rf"$k_c(\mu={mu})={k_crit:.2f}$",
        )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title(r"Effect of predator mortality $\mu$", fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()

    save_current_figure("variation_mu")

    plt.show()


def _plot_beta_variation(k_values, P0, D0, H0, T_markov, n_realizations, tail):
    """
    Plot stationary predator density for several colonization values.
    """

    beta_values = [0.3, 0.6, 0.9]

    plt.figure("Colonization variation", figsize=(10, 6))

    for beta in beta_values:
        _, D_inf, _, _, D_std, _ = stationary_state_vs_degree(
            generate_er_graph_func=generate_er_graph_from_k,
            n_nodes=N_NODES,
            k_values=k_values,
            beta=beta,
            alpha=ALPHA,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_markov,
            n_realizations=n_realizations,
            tail=tail,
        )

        plt.errorbar(
            k_values,
            D_inf,
            yerr=D_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=rf"$\beta={beta}$",
        )

    k_crit = MU / ALPHA

    plt.axvline(
        x=k_crit,
        linestyle="--",
        linewidth=2.6,
        color="black",
        alpha=0.75,
        label=rf"$k_c=\mu/\alpha={k_crit:.2f}$",
    )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title(r"Effect of prey colonization $\beta$", fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend(fontsize=11)
    plt.tight_layout()

    save_current_figure("variation_beta")

    plt.show()


def _plot_alpha_variation(k_values, P0, D0, H0, T_markov, n_realizations, tail):
    """
    Plot stationary predator density for several predation values.
    """

    alpha_values = [0.2, 0.5, 0.8]

    plt.figure("Predation variation", figsize=(10, 6))

    for alpha in alpha_values:
        _, D_inf, _, _, D_std, _ = stationary_state_vs_degree(
            generate_er_graph_func=generate_er_graph_from_k,
            n_nodes=N_NODES,
            k_values=k_values,
            beta=BETA,
            alpha=alpha,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_markov,
            n_realizations=n_realizations,
            tail=tail,
        )

        curve = plt.errorbar(
            k_values,
            D_inf,
            yerr=D_std,
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=rf"$\alpha={alpha}$",
        )

        color = curve[0].get_color()
        k_crit = MU / alpha

        plt.axvline(
            x=k_crit,
            linestyle="--",
            linewidth=2.3,
            color=color,
            alpha=0.55,
            label=rf"$k_c(\alpha={alpha})={k_crit:.2f}$",
        )

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"$D_\infty$", fontsize=13)
    plt.title(r"Effect of predation $\alpha$", fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.65)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()

    save_current_figure("variation_alpha")

    plt.show()


# =========================================================
# 6) Heatmap
# =========================================================

def plot_heatmap():
    """
    Plot a heatmap of stationary predator density in the (k, mu) plane.
    """

    k_values = np.linspace(1, 20, 10)
    mu_values = np.linspace(0.05, 0.95, 10)

    P0, D0, H0 = INITIAL_PROPORTIONS
    T_markov = 200

    heatmap_D = np.zeros((len(mu_values), len(k_values)))

    for i, mu in enumerate(mu_values):
        for j, k_avg in enumerate(k_values):

            _, D_inf, _, _, _, _ = stationary_state_vs_degree(
                generate_er_graph_func=generate_graph_combined(GRAPH_TYPE, k_avg),
                n_nodes=N_NODES,
                k_values=[k_avg],
                beta=BETA,
                alpha=ALPHA,
                mu=mu,
                P0=P0,
                D0=D0,
                H0=H0,
                T=T_markov,
                n_realizations=1,
                tail=20,
            )

            heatmap_D[i, j] = D_inf[0]

    plt.figure("Heatmap", figsize=(10, 6))

    image = plt.imshow(
        heatmap_D,
        origin="lower",
        aspect="auto",
        extent=[k_values[0], k_values[-1], mu_values[0], mu_values[-1]],
    )

    plt.colorbar(image, label=r"$D_\infty$")

    plt.xlabel(r"Average degree $\langle k \rangle$", fontsize=13)
    plt.ylabel(r"Mortality $\mu$", fontsize=13)
    plt.title(
        r"Stationary predator density $D_\infty$ in the $(\langle k \rangle,\mu)$ plane",
        fontsize=15,
    )

    ax = plt.gca()
    ax.axline(
        (0, 0),
        slope=ALPHA,
        linestyle="--",
        color="white",
        linewidth=2.5,
        label=rf"$\mu=\alpha \langle k \rangle$",
    )

    plt.xlim(k_values[0], k_values[-1])
    plt.ylim(mu_values[0], mu_values[-1])

    plt.legend()
    plt.tight_layout()

    save_current_figure("heatmap_mu")

    plt.show()


# =========================================================
# 7) Animation
# =========================================================

def animate_pdh_simulation():
    """
    Create and save an animation of the stochastic PDH dynamics.
    """

    n_vis = 40
    k_vis = 4
    steps_vis = 100

    graph, _ = generate_graph_combined(GRAPH_TYPE, k_vis)

    history = simulate_pdh(
        graph=graph,
        pdh_parameters=PDH_PARAMS,
        initial_proportions=INITIAL_PROPORTIONS,
        steps=steps_vis,
    )

    pos = nx.spring_layout(graph)

    fig, (ax_network, ax_curves) = plt.subplots(1, 2, figsize=(18, 8))

    P_values = []
    D_values = []
    H_values = []

    def node_colors(state):
        colors = []

        for node in graph.nodes():
            if state[node] == "P":
                colors.append("green")
            elif state[node] == "D":
                colors.append("red")
            else:
                colors.append("gray")

        return colors

    def update(frame):
        ax_network.clear()
        ax_curves.clear()

        state = history[frame]
        counts = count_states(state)

        P_values.append(counts["P"])
        D_values.append(counts["D"])
        H_values.append(counts["H"])

        nx.draw(
            graph,
            pos,
            node_color=node_colors(state),
            node_size=120,
            ax=ax_network,
        )

        ax_network.set_title(
            rf"ER network with $\langle k \rangle={k_vis}$ | Step {frame} | {GRAPH_TYPE}"
        )

        ax_curves.plot(P_values, label="Prey")
        ax_curves.plot(D_values, label="Predators")
        ax_curves.plot(H_values, label="Empty sites")

        ax_curves.set_xlim(0, steps_vis)
        ax_curves.set_ylim(0, n_vis)

        ax_curves.set_xlabel("Time")
        ax_curves.set_ylabel("Number of nodes")
        ax_curves.set_title("Temporal evolution")
        ax_curves.legend()
        ax_curves.grid(True, alpha=0.65)

        plt.tight_layout()

    simulation_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=200,
        blit=False,
        repeat=False,
    )

    k_tag = format_float_for_filename(k_vis)
    gif_path = get_animation_path(f"pdh_animation_k_{k_tag}")

    simulation_animation.save(
        gif_path,
        writer="pillow",
        fps=5,
    )

    print(f"Animation saved to: {gif_path}")

    plt.show()