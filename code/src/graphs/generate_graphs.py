"""
Network generation utilities.
"""

import networkx as nx
import numpy as np

from config import (
    N_NODES,
    ERDOS_P,
    BARABASI_M,
    WATTS_K,
    WATTS_P,
)
from src.simulation.simulation_model import *
from src.utils.helpers import *
from src.utils.metrics import *
from config import *
from src.visualization.plots import *

import src.graphs.generate_graphs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np


def generate_graph(graph_type):
    """
    Generate a network according to the requested graph type.

    Parameters
    ----------
    graph_type : str
        One of: "erdos", "barabasi", "watts".

    Returns
    -------
    networkx.Graph
        Generated network.
    """

    if graph_type == "erdos":
        return generate_er_graph_from_k(N_NODES, ERDOS_P)

    if graph_type == "barabasi":
        return nx.barabasi_albert_graph(N_NODES, BARABASI_M)

    if graph_type == "watts":
        return nx.watts_strogatz_graph(N_NODES, WATTS_K, WATTS_P)

    raise ValueError(
        f"Invalid graph type '{graph_type}'. "
        "Valid options are: 'erdos', 'barabasi', 'watts'."
    )


def generate_er_graph_from_k(n_nodes, k_avg):
    """
    Generate an Erdős-Rényi graph with a target average degree.

    For an ER graph G(N, p), the expected mean degree is:

        <k> = p (N - 1)

    Therefore:

        p = <k> / (N - 1)

    Returns
    -------
    tuple[networkx.Graph, numpy.ndarray]
        Generated graph and its adjacency matrix.
    """

    p = k_avg / (n_nodes - 1)
    graph = nx.erdos_renyi_graph(n_nodes, p)
    adjacency_matrix = nx.to_numpy_array(graph, dtype=float)

    return graph, adjacency_matrix


def generate_graph_combined(graph_type, k_avg):
    
    if graph_type == "erdos":
        G, A = generate_er_graph_from_k(N_NODES, k_avg)

    elif graph_type == "barabasi":
        if k_avg < 2:
            k_avg = 2
        BARABASI_M = k_avg // 2  # Ajustar m para obtener el grado medio deseado
        G = nx.barabasi_albert_graph(N_NODES, int(BARABASI_M))
        A = nx.to_numpy_array(G, dtype=float)
    
    elif graph_type == "watts":
        k_avg = int(k_avg)  # Asegurarse de que k_avg sea un entero
        G = nx.watts_strogatz_graph(N_NODES, k_avg, WATTS_P)
        A = nx.to_numpy_array(G, dtype=float)
    
    else:
        raise ValueError("Tipo de grafo no válido")
    
    return G, A