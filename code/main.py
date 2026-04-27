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

#Genera una red Erdős-Rényi con grado medio aproximado <k>.
def generate_er_graph_from_k(n_nodes, k_avg, seed=None):
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
    P0, D0, H0 = initial_proportions
    T_MARKOV = 200

    for k_avg in k_values:

        # Generar red ER con grado medio <k>
        _, A = generate_er_graph_from_k(N_NODES, k_avg, seed=42)

        # Ejecutar dinámica de Markov
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
        beta=BETHA,   
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


    # =========================================================
    # 4) MAPA DE CALOR D∞ vs (alpha, <k>)
    # =========================================================

    k_values = np.linspace(1, 20, 25) #Esto funciona (k_inicial, k_final, n_puntos)
    alpha_values = np.linspace(0.05, 0.9, 25)

    # Ajuste de parámetros (zona interesante) 
    BETA_HEAT = BETHA
    MU_HEAT = MU

    heatmap_D = np.zeros((len(alpha_values), len(k_values)))

    for i, alpha in enumerate(alpha_values):
        for j, k_avg in enumerate(k_values):

            P_inf, D_inf, H_inf, _, _, _ = stationary_state_vs_degree(
                generate_er_graph_func=generate_er_graph_from_k,
                n_nodes=N_NODES,
                k_values=[k_avg],
                beta=BETA_HEAT,
                alpha=alpha,
                mu=MU_HEAT,
                P0=P0,
                D0=D0,
                H0=H0,
                T=T_MARKOV,
                n_realizations=1,   #numero de redes erdos para el analisis
                tail=20
            )

            heatmap_D[i, j] = D_inf[0]


    # =========================================================
    # VISUALIZACIÓN (MAPA NORMAL)
    # =========================================================
    plt.figure(figsize=(8,6))

    im = plt.imshow(
        heatmap_D,
        origin='lower',
        aspect='auto',
        extent=[k_values[0], k_values[-1], alpha_values[0], alpha_values[-1]]
    )

    plt.colorbar(im, label="Fracción estacionaria de depredadores")

    plt.xlabel("<k>")
    plt.ylabel("alpha")
    plt.title(f"Mapa de calor D∞ (β={BETA_HEAT}, μ={MU_HEAT})")

    plt.tight_layout()
    plt.show()