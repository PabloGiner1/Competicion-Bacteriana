import networkx as nx
from config import *

def generate_graph(graph_type):
    
    if graph_type == "erdos":
        G = nx.erdos_renyi_graph(N_NODES, ERDOS_P)
    
    elif graph_type == "barabasi":
        G = nx.barabasi_albert_graph(N_NODES, BARABASI_M)
    
    elif graph_type == "watts":
        G = nx.watts_strogatz_graph(N_NODES, WATTS_K, WATTS_P)
    
    else:
        raise ValueError("Tipo de grafo no válido")
    
    return G