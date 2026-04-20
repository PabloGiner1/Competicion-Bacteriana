# Contiene funciones utiles para calcular propiedades estructurales de las redes. 

import networkx as nx
import numpy as np

import networkx as nx
import numpy as np


# 1. GRADO
#Devuelvo la distribucion de grados como un diccionario {grado: frecuencia} y la media del grado.
def degree_distribution(G):

    degrees = [deg for _, deg in G.degree()]
    unique, counts = np.unique(degrees, return_counts=True)

    return dict(zip(unique, counts / len(degrees)))


def average_degree(G):
    return np.mean([deg for _, deg in G.degree()])


# 2. CLUSTERING
# Devuelvo el coeficiente de clustering medio de la red.
def clustering_coefficient(G):
    return nx.average_clustering(G)


# 3. DISTANCIAS
# Devuelvo la longitud media de los caminos más cortos y el diámetro de la red.
def average_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        return None


def diameter(G):
    if nx.is_connected(G):
        return nx.diameter(G)
    else:
        return None


# 4. CENTRALIDAD
# Devuelvo la centralidad de grado, intermediación y cercanía de cada nodo como diccionarios {nodo: centralidad}.
def degree_centrality(G):
    return nx.degree_centrality(G)


def betweenness_centrality(G):
    return nx.betweenness_centrality(G)


def closeness_centrality(G):
    return nx.closeness_centrality(G)


# 5. CORRELACIONES
# Devuelvo el coeficiente de assortatividad de la red.
def assortativity(G):
    return nx.degree_assortativity_coefficient(G)


# 6. DENSIDAD
# Devuelvo la densidad de la red.
def density(G):
    return nx.density(G)



# 7. NÚMERO DE ENLACES
# Devuelvo el número total de enlaces en la red.
def number_of_links(G):
    return G.number_of_edges()