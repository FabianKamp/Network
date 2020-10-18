import networkx as nx
import numpy as np
import pandas as pd

def shortest_path_length(adjacency):
    """
    Takes Functional Connectivity Matrix and returns the shortest path length between all nodes as array
    :param FC: NxN dimensional array of the functional connectivity
    :return: NxN dimensional array of shortest pathlengths
    """
    assert isinstance(adjacency, np.ndarray), "Adjacency matrix must be np.ndarray."
    adjacency = np.asarray(adjacency)
    N = adjacency.shape[0]
    G = nx.from_numpy_array(adjacency)

    # If neither the source nor target are specified nx.shortest path length returns a dictionary of dictionaries 
    # with path[source][target]=[list of nodes in path]
    shortest_path = nx.shortest_path_length(G, weight='weight')
    path_lengths_mat=np.zeros((N,N))

    for n, tup in enumerate(shortest_path):             # Shortestpath is list of tuples one for each region n
        for m, val in tup[1].items():                   # Element [1] in the tuple contain dictionary of pathlength of n to other regions
            path_lengths_mat[n,m] = val             # Extract value from dictionary and puts it into matrix

    return path_lengths_mat

def shortest_path(adjacency):
    """
    Takes Functional Connectivity Matrix and returns the shortest paths between all nodes as array
    :param FC: NxN dimensional array of the functional connectivity
    :return: dictionary of all shortest path
    """
    assert isinstance(adjacency, np.ndarray), "Adjacency matrix must be np.ndarray."
    adjacency = np.asarray(adjacency)
    N = adjacency.shape[0]
    G = nx.from_numpy_array(adjacency)
    path_dict = nx.shortest_path(G, weight='weight')
    return path_dict # Return path list so it can be indexed in [n,n]

def between_centrality(adjacency):
    """
    Takes Functional Connectivity Matrix and returns the betweeness centrality of all nodes as array
    :param FC: NxN dimensional array of the functional connectivity
    :return: N dimensional array containing the betweeness centrality of all nodes
    """
    assert isinstance(adjacency, np.ndarray), "Adjacency matrix must be np.ndarray."
    adjacency = np.asarray(adjacency)
    N = adjacency.shape[0]
    G = nx.from_numpy_array(adjacency)
    betw_cent_dict = nx.betweenness_centrality(G, k=None, normalized=False, weight='weight')
    betw_cent = np.array([betw_cent_dict[i] for i in range(N)])
    return betw_cent

def clustering(adjacency):
    """
    Takes Functional Connectivity Matrix and returns the clustering coefficient of all nodes as array
    :param FC: NxN dimensional array of the functional connectivity
    :return: N dimensional array containing the betweeness centrality of all nodes
    """
    assert isinstance(adjacency, np.ndarray), "Adjacency matrix must be np.ndarray."
    adjacency = np.asarray(adjacency)
    N = adjacency.shape[0]
    G = nx.from_numpy_array(adjacency)
    clust_dict = nx.clustering(G, weight='weight')
    clust = np.array([clust_dict[i] for i in range(N)])
    return clust

def assortativity(adjacency):
    """
    Computes assortativity coefficient of adjacency matrix using networkx immplementation.
    :param adjacency: array like adjacency matrix of network
    :return: float assortativity coeff
    """
    G = nx.from_numpy_array(adjacency)
    assortativity = nx.degree_pearson_correlation_coefficient(G, weight='weight')
    return assortativity

def MST(adjacency):
    """
    Computes the minimum spanning tree (MST) of the adjacency matrix
    :param np array of adjacency matrix
    :return np array of MST
    """
    G = nx.from_numpy_array(adjacency)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    mst = nx.to_numpy_array(mst)
    return mst