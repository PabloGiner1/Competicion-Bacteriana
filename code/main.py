from src.simulation.simulation_model import simulate_pdh
import src.graphs.generate_graphs
from src.utils.helpers import *
from src.utils.metrics import *
from config import *

import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    
    P, D, H = run_simulation(graph_type)
    
    plt.plot(P, label="Presas")
    plt.plot(D, label="Depredadoras")
    plt.plot(H, label="Nodos vacíos")
    
    plt.legend()
    plt.xlabel("Tiempo")
    plt.ylabel("Número de nodos")
    plt.title("Simulación PDH en red " + graph_type)
    
    plt.show()