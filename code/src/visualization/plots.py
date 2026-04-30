from src.simulation.simulation_model import *
from src.utils.helpers import *
from src.utils.metrics import *
from config import *

import src.graphs.generate_graphs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np


# =========================================================
# UTILIDADES
# =========================================================

def run_simulation(graph_type):
    G = src.graphs.generate_graphs.generate_graph(graph_type)

    history = simulate_pdh(G, pdh_params, initial_proportions, STEPS)

    P_list, D_list, H_list = [], [], []

    for state in history:
        counts = count_states(state)
        P_list.append(counts["P"])
        D_list.append(counts["D"])
        H_list.append(counts["H"])

    return P_list, D_list, H_list


def generate_er_graph_from_k(n_nodes, k_avg, seed=None):
    p = k_avg / (n_nodes - 1)
    G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    return G, A


# =========================================================
# 1) SIMULACIÓN ORIGINAL
# =========================================================

def plot_original_simulation():
    P, D, H = run_simulation(graph_type)

    plt.figure("Simulación original")
    plt.plot(P, label="Presas")
    plt.plot(D, label="Depredadoras")
    plt.plot(H, label="Nodos vacíos")

    plt.legend()
    plt.xlabel("Tiempo")
    plt.ylabel("Número de nodos")
    plt.title(f"Simulación PDH en red {graph_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# 2) MARKOV INDIVIDUAL PARA DISTINTOS k
# =========================================================

def plot_markov_individual():
    k_values = [2, 4, 6, 8, 10]
    P0, D0, H0 = initial_proportions
    T_MARKOV = 200

    for k_avg in k_values:
        _, A = generate_er_graph_from_k(N_NODES, k_avg, seed=42)

        P_markov, D_markov, H_markov = markov_pdh_dynamics(
            A=A,
            beta=BETHA,
            alpha=ALPHA,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_MARKOV
        )

        plt.figure(f"Markov k={k_avg}")
        plt.plot(P_markov, label="Presa")
        plt.plot(D_markov, label="Depredador")
        plt.plot(H_markov, label="Hueco")

        plt.title(f"Markov PDH (<k>={k_avg})")
        plt.xlabel("Tiempo")
        plt.ylabel("Fracción")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =========================================================
# 2.5) MARKOV COMBINADO
# =========================================================

def plot_markov_combined():
    k_values = [2, 4, 6, 8, 10]
    P0, D0, H0 = initial_proportions
    T_MARKOV = 200

    results_P = {}
    results_D = {}
    results_H = {}

    for k_avg in k_values:
        _, A = generate_er_graph_from_k(N_NODES, k_avg, seed=42)

        P_markov, D_markov, H_markov = markov_pdh_dynamics(
            A=A,
            beta=BETHA,
            alpha=ALPHA,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_MARKOV
        )

        results_P[k_avg] = P_markov
        results_D[k_avg] = D_markov
        results_H[k_avg] = H_markov

    # Presas
    plt.figure("Markov combinado - Presas")
    for k in k_values:
        plt.plot(results_P[k], label=f"<k>={k}")
    plt.title("Presas para distintos <k>")
    plt.xlabel("Tiempo")
    plt.ylabel("Fracción")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Depredadores
    plt.figure("Markov combinado - Depredadores")
    for k in k_values:
        plt.plot(results_D[k], label=f"<k>={k}")
    plt.title("Depredadores para distintos <k>")
    plt.xlabel("Tiempo")
    plt.ylabel("Fracción")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Huecos
    plt.figure("Markov combinado - Huecos")
    for k in k_values:
        plt.plot(results_H[k], label=f"<k>={k}")
    plt.title("Huecos para distintos <k>")
    plt.xlabel("Tiempo")
    plt.ylabel("Fracción")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# 3) ESTADO ESTACIONARIO
# =========================================================

def plot_stationary_state():
    k_values = [2, 4, 6, 8, 10]
    P0, D0, H0 = initial_proportions
    T_MARKOV = 200

    P_inf, D_inf, H_inf, P_std, D_std, H_std = stationary_state_vs_degree(
        generate_er_graph_func=generate_er_graph_from_k,
        n_nodes=N_NODES,
        k_values=k_values,
        beta=BETHA,
        alpha=ALPHA,
        mu=MU,
        P0=P0,
        D0=D0,
        H0=H0,
        T=T_MARKOV,
        n_realizations=5,
        tail=20
    )

    plt.figure("Estado estacionario")
    plt.errorbar(k_values, P_inf, yerr=P_std, marker='o', capsize=4, label='Presa')
    plt.errorbar(k_values, D_inf, yerr=D_std, marker='s', capsize=4, label='Depredador')
    plt.errorbar(k_values, H_inf, yerr=H_std, marker='^', capsize=4, label='Hueco')

    plt.xlabel("<k>")
    plt.ylabel("Fracción estacionaria")
    plt.title("Estado estacionario PDH")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# 4) HEATMAP
# =========================================================

def plot_heatmap():
    k_values = np.linspace(1, 20, 25)
    alpha_values = np.linspace(0.05, 0.9, 25)

    P0, D0, H0 = initial_proportions
    T_MARKOV = 200

    heatmap_D = np.zeros((len(alpha_values), len(k_values)))

    for i, alpha in enumerate(alpha_values):
        for j, k_avg in enumerate(k_values):

            _, D_inf, _, _, _, _ = stationary_state_vs_degree(
                generate_er_graph_func=generate_er_graph_from_k,
                n_nodes=N_NODES,
                k_values=[k_avg],
                beta=BETHA,
                alpha=alpha,
                mu=MU,
                P0=P0,
                D0=D0,
                H0=H0,
                T=T_MARKOV,
                n_realizations=1,
                tail=20
            )

            heatmap_D[i, j] = D_inf[0]

    plt.figure("Heatmap")
    im = plt.imshow(
        heatmap_D,
        origin='lower',
        aspect='auto',
        extent=[k_values[0], k_values[-1], alpha_values[0], alpha_values[-1]]
    )

    plt.colorbar(im, label="D∞")
    plt.xlabel("<k>")
    plt.ylabel("alpha")
    plt.title("Mapa de calor D∞")
    plt.tight_layout()
    plt.show()

# =========================================================
# 5) ANIMACIÓN
# =========================================================

def animate_pdh_simulation():
    """
    Animación del modelo PDH:
    - izquierda: evolución espacial de la red
    - derecha: evolución temporal de P, D y H

    Guarda además un GIF en outputs/pdh_simulation.gif
    """

    import os
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import networkx as nx

    # =====================================================
    # PARÁMETROS DE VISUALIZACIÓN
    # =====================================================
    n_vis = 40          # mejor pequeño para visualizar
    k_vis = 4
    steps_vis = 100

    # Crear carpeta de salida
    os.makedirs("outputs", exist_ok=True)

    # =====================================================
    # GENERAR RED Y SIMULAR
    # =====================================================
    G, _ = generate_er_graph_from_k(
        n_nodes=n_vis,
        k_avg=k_vis,
        seed=42
    )

    history = simulate_pdh(
        G,
        pdh_params,
        initial_proportions,
        steps_vis
    )

    # Layout visual (NO cambia la red, solo dibujo)
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # =====================================================
    # FIGURA
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    P_vals = []
    D_vals = []
    H_vals = []

    # =====================================================
    # COLORES
    # =====================================================
    def get_colors(state):
        colors = []

        for node in G.nodes():
            if state[node] == "P":
                colors.append("green")
            elif state[node] == "D":
                colors.append("red")
            else:
                colors.append("lightgray")

        return colors

    # =====================================================
    # UPDATE FRAME
    # =====================================================
    def update(frame):

        ax1.clear()
        ax2.clear()

        state = history[frame]
        counts = count_states(state)

        P_vals.append(counts["P"])
        D_vals.append(counts["D"])
        H_vals.append(counts["H"])

        # ---------------------------------
        # PANEL IZQUIERDO: RED
        # ---------------------------------
        colors = get_colors(state)

        nx.draw(
            G,
            pos,
            node_color=colors,
            node_size=120,
            with_labels=False,
            width=0.3,
            alpha=0.9,
            ax=ax1
        )

        ax1.set_title(
            f"Modelo PDH en red ER\n"
            f"Paso = {frame}"
        )

        # ---------------------------------
        # PANEL DERECHO: CURVAS
        # ---------------------------------
        ax2.plot(
            range(len(P_vals)),
            P_vals,
            label="Presas",
            color="green"
        )

        ax2.plot(
            range(len(D_vals)),
            D_vals,
            label="Depredadores",
            color="red"
        )

        ax2.plot(
            range(len(H_vals)),
            H_vals,
            label="Huecos",
            color="gray"
        )

        ax2.set_xlim(0, steps_vis)
        ax2.set_ylim(0, n_vis)

        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("Número de nodos")

        ax2.set_title("Evolución temporal")

        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

    # =====================================================
    # ANIMACIÓN
    # =====================================================
    print("Generando animación...")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=250,
        blit=False,
        repeat=False
    )

    plt.show()

    print("Guardando GIF...")

    ani.save(
        "outputs/pdh_simulation.gif",
        writer="pillow",
        fps=5
    )

    print("GIF guardado en outputs/pdh_simulation.gif")