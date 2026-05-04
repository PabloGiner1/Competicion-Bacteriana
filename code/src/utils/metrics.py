"""
Network metric utilities.
"""

import networkx as nx
import numpy as np


def degree_distribution(graph):
    """
    Return the normalized degree distribution as {degree: frequency}.
    """

    degrees = [degree for _, degree in graph.degree()]
    unique, counts = np.unique(degrees, return_counts=True)

    return dict(zip(unique, counts / len(degrees)))


def average_degree(graph):
    """
    Return the average degree of the graph.
    """

    return np.mean([degree for _, degree in graph.degree()])


def clustering_coefficient(graph):
    """
    Return the average clustering coefficient.
    """

    return nx.average_clustering(graph)


def average_path_length(graph):
    """
    Return the average shortest path length if the graph is connected.
    """

    if nx.is_connected(graph):
        return nx.average_shortest_path_length(graph)

    return None


def diameter(graph):
    """
    Return the graph diameter if the graph is connected.
    """

    if nx.is_connected(graph):
        return nx.diameter(graph)

    return None


def degree_centrality(graph):
    """
    Return degree centrality for each node.
    """

    return nx.degree_centrality(graph)


def betweenness_centrality(graph):
    """
    Return betweenness centrality for each node.
    """

    return nx.betweenness_centrality(graph)


def closeness_centrality(graph):
    """
    Return closeness centrality for each node.
    """

    return nx.closeness_centrality(graph)


def assortativity(graph):
    """
    Return degree assortativity coefficient.
    """

    return nx.degree_assortativity_coefficient(graph)


def density(graph):
    """
    Return graph density.
    """

    return nx.density(graph)


def number_of_links(graph):
    """
    Return the total number of edges.
    """

    return graph.number_of_edges()

def degree_heterogeneity_factor(graph):
    """
    Compute the structural heterogeneity factor of a network.

    The quantity

        kappa = <k^2> / <k>

    is commonly used to characterize degree heterogeneity. Networks with hubs,
    such as Barabási-Albert networks, typically have larger values of kappa
    than homogeneous Erdős-Rényi networks with similar average degree.
    """

    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)

    mean_k = np.mean(degrees)
    mean_k2 = np.mean(degrees ** 2)

    if mean_k == 0:
        return 0.0

    return mean_k2 / mean_k