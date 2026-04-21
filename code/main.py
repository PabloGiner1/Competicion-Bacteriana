from src.simulation.simulation_model import *
import src.graphs.generate_graphs
from src.utils.helpers import *
from src.utils.metrics import *
from config import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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
    """
    Genera una red Erdős-Rényi con grado medio aproximado <k>.
    """
    p = k_avg / (n_nodes - 1)
    G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    return G, A


if __name__ == "__main__":
    
    # =========================================================
    # 1) SIMULACIÓN ORIGINAL DEL REPO
    # =========================================================
    P, D, H = run_simulation(graph_type)
    
    plt.figure()
    plt.plot(P, label="Presas")
    plt.plot(D, label="Depredadoras")
    plt.plot(H, label="Nodos vacíos")
    
    plt.legend()
    plt.xlabel("Tiempo")
    plt.ylabel("Número de nodos")
    plt.title("Simulación PDH en red " + graph_type)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    # =========================================================
    # 2) DINÁMICA DE MARKOV EN REDES ER PARA DISTINTOS <k>
    # =========================================================
    k_values = [2, 4, 6, 8, 10]

    # Condiciones iniciales de la dinámica de Markov
    P0 = 0.20
    D0 = 0.05
    H0 = 0.75
    T_MARKOV = 200

    for k_avg in k_values:

        # Generar red ER con grado medio <k>
        _, A = generate_er_graph_from_k(N_NODES, k_avg, seed=42)

        # Ejecutar dinámica de Markov
        P_markov, D_markov, H_markov = markov_pdh_dynamics(
            A=A,
            beta=BETHA,   # cambia a BETA si en tu config está escrito así
            alpha=ALPHA,
            mu=MU,
            P0=P0,
            D0=D0,
            H0=H0,
            T=T_MARKOV
        )

        # Representar evolución temporal
        plt.figure()
        plt.plot(P_markov, label="Presa")
        plt.plot(D_markov, label="Depredador")
        plt.plot(H_markov, label="Hueco")
        plt.title(f"Markov PDH en red ER (<k> = {k_avg})")
        plt.xlabel("Tiempo")
        plt.ylabel("Fracción")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    # =========================================================
    # 3) ESTADO ESTACIONARIO FRENTE AL GRADO MEDIO <k>
    # =========================================================
    n_realizations = 5   # número de redes distintas para cada <k>
    tail = 20            # últimos pasos usados para estimar el estado estacionario

    P_inf, D_inf, H_inf, P_std, D_std, H_std = stationary_state_vs_degree(
        generate_er_graph_func=generate_er_graph_from_k,
        n_nodes=N_NODES,
        k_values=k_values,
        beta=BETHA,   # cambia a BETA si en tu config está escrito así
        alpha=ALPHA,
        mu=MU,
        P0=P0,
        D0=D0,
        H0=H0,
        T=T_MARKOV,
        n_realizations=n_realizations,
        tail=tail
    )

    # Gráfica del estado estacionario
    plt.figure(figsize=(8, 5))
    plt.errorbar(k_values, P_inf, yerr=P_std, marker='o', capsize=4, label='Presa')
    plt.errorbar(k_values, D_inf, yerr=D_std, marker='s', capsize=4, label='Depredador')
    plt.errorbar(k_values, H_inf, yerr=H_std, marker='^', capsize=4, label='Hueco')

    plt.xlabel("Grado medio <k>")
    plt.ylabel("Fracción estacionaria")
    plt.title("Estado estacionario del modelo PDH en redes ER")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()