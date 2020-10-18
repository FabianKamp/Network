import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import corr_functions as corr
import from_networkx as fnx
import net_algorithms as alg

class NetworkError(Exception):
    """
    Exception Error to report for Network Errors
    """
    def __init__(self, ErrorMessage):
        print(ErrorMessage)

class network:
    """Defines input as network
    :parameter pd.DataFrame that contains the adjacency matrix of the network, np.ndarray timecourse matrix
    TODO correct clust coeff
    """
    def __init__(self, Adjacency_Matrix, Node_Names):
        if not isinstance(Adjacency_Matrix, np.ndarray):
            raise NetworkError('Input must be numpy.ndarray.')
        if len(Adjacency_Matrix.shape) != 2 or Adjacency_Matrix.shape[0] != Adjacency_Matrix.shape[1] or Adjacency_Matrix.shape[0] < 4:
            raise NetworkError('Adjacency matrix must be a 2 dimensional square matrix with more than 4 nodes.')       
        if len(Node_Names) != Adjacency_Matrix.shape[0]:
            raise NetworkError('Adjacency')       
        if np.any(Adjacency_Matrix<0) or np.any(Adjacency_Matrix.T-Adjacency_Matrix) or np.any(np.diag(Adjacency_Matrix)!=0):
            raise NetworkError('Adjacency matrix must be symmetric and all edges must be positive. Diagonal must be 0.')        

        self.adj_mat = np.asarray(Adjacency_Matrix)
        self.nodes = Node_Names
        self.number_nodes = len(self.nodes)

    def __getitem__(self, index):
        return self.adj_mat.iloc[index]

    def invert_edges(self, adj=None):
        """
        Edge inversion function --> 1/edge, edge != 0
        :return: adjacency matrix with inversed edges
        """
        if adj is None:
            adj_mat = self.adj_mat.copy()
        else: 
            adj_mat = adj.copy()
        
        # Inverting Non-zero values
        inv_edges = np.copy(1 / adj_mat[adj_mat != 0])
        adj_mat[adj_mat!=0] = inv_edges

        return adj_mat
    
    def normalize_net(self):
        adj_mat = self.adj_mat
        max_weight = np.max(adj_mat)
        adj_mat /= max_weight
        return adj_mat

    def binarize_net(self, threshold):
        pass

    def degrees(self, avg=True):
        """
        Calculate the degree of each node in the network.
        :return n dimensional array with degrees of all nodes in the network
        """
        degrees = np.sum(self.adj_mat, axis=-1) 
        if avg:
            return np.mean(degrees)
        else:
            return degrees

    def shortest_path_lengths(self):
        """
        Calculate the shortest path between all nodes in the network using the networkx implementation

        :param: nx boolean value, if true uses networkX to compute the shortest path lengths
                path boolean value, specify if the paths are returned
        :return Dictionary of two nxn dimensional pd.DataFrames with shortest path / shortest distance between all pairs of nodes in the network
        """        
        inv_adj_mat = self.invert_edges()
        # NetworkX implementation of the shortest path length
        shortest_path_matrix = fnx.shortest_path_length(inv_adj_mat)
        return shortest_path_matrix
    
    def shortest_paths(self): 
        inv_adj_mat = self.invert_edges()
        short_paths = fnx.shortest_path(inv_adj_mat)
        return short_paths

    def num_triangles(self):
        """
        Calculate sum of triangles edge weights around each node in network.
        The edge weights are normalized with the largest weight in the network
        :return: n dimensional 
        """
        adj_mat = self.adj_mat.copy()
        triangles = np.zeros(self.number_nodes)
        all_combinations = combinations(np.arange(self.number_nodes), 3) # Create list of all possible triangles
        m_dict={}
        for combi in all_combinations:
            n1_n2 = adj_mat[combi[0],combi[1]]                      # Get path length between pairs in triangle combination
            n1_n3 = adj_mat[combi[0],combi[2]]
            n2_n3 = adj_mat[combi[1],combi[2]]
            m_dict[combi] = (n1_n2*n1_n3*n2_n3)**(1/3)              # Calculate the triangle sum of the combination and save it in dictionary
        for node in range(self.number_nodes):
            triangles[node] = (1/2) * np.sum([m_dict[s] for s in m_dict if node in s])    # Sum all of the triangles that contain the node
        return triangles

    def char_path(self, avg=True):
        """
        Calculate the characteristic path length of the network
        :return: Dictionary with average node distance np.array and characteristic path length np.float object
        """
        sum_shrtpath = np.sum(self.shortest_path_lengths(), axis=-1)
        avg_shrtpath_node = sum_shrtpath / (self.number_nodes-1)                
        if avg:
            char_pathlength = np.mean(avg_shrtpath_node)
            return char_pathlength                                                                      
        else:
            return avg_shrtpath_node              

    def glob_efficiency(self):
        """
        Calculate the global efficiency of the network
        :return: np.float object
        """
        shortestpath = self.shortest_path_lengths()
        # Inverting shortest path
        np.fill_diagonal(shortestpath, 1.)      
        inv_shrtpath=1/shortestpath                                             
        np.fill_diagonal(inv_shrtpath, 0)                                           
        sum_invpath_df = np.sum(inv_shrtpath, axis=-1)                              
        avg_invpath = sum_invpath_df / (self.number_nodes-1)                        
        glob_efficiency= np.mean(avg_invpath)                                      
        return glob_efficiency

    def clust_coeff(self, avg=True):
        """
        Calculate the cluster coefficient of the network
        :param: avg, if False the cluster coeff is outputed for each node
        :return: Network cluster coefficient np.float object or ndim np.array of node by node cluster coefficients
        """
        adj = np.array(self.adj_mat)
        node_clust = fnx.clustering(adj)
        if avg:
            net_clust = np.mean(node_clust)
            return net_clust
        else:
            return node_clust

    def transitivity(self):
        """
        Calculate the transitivity of the network
        :return: np.float
        """
        sum_triangles = np.sum(self.num_triangles()*2)     # Multiply sum of triangles with 2 and sum the array
        degrees = self.degrees()
        degrees *= (degrees-1)
        sum_degrees = np.sum(degrees)
        transitivity = sum_triangles / sum_degrees
        return transitivity

    def closeness_centrality(self, avg=True):
        """
        Calculate the closeness centrality of each node in network.
        Optionally takes in the shortest average path length of each node, which saves computation time.
        :param: n dimensional array or pd.Series that contains the average shortest path for each node
        :return: ndimensional np. array
        """
        node_avg_distance = self.char_path(avg=False)
        if not np.all(node_avg_distance):                                           
            print('Excluding isolated nodes, i.e. nodes with shortest average path length of zero.')
            node_avg_distance[node_avg_distance==0] = np.nan
        close_centr = 1 / node_avg_distance                  
        if avg:
            return np.mean(close_centr)
        else: 
            return close_centr

    def betweenness_centrality(self, avg=True):
        """
        Calculate the betweenness centrality of each node in network
        :return: ndimensional pd.Series
        """
        adj_mat = self.invert_edges()
        betw_cent = fnx.between_centrality(adj_mat)
        if avg: 
            return np.mean(betw_cent)
        else: 
            return betw_cent

    def avg_neigh_degree(self, avg=True):
        """
        Compute the average neighbour degree following the definition of Rubinov and Sporns 2011
        :return: pd.Series, average neighbour degree of each node
        """
        adj = self.adj_mat
        degrees = self.degrees()
        avg_neigh_degree = np.zeros(self.number_nodes)
        for n in range(self.number_nodes):
            avg_neigh_degree[n] = np.sum(adj[n, :] * degrees) / degrees[n]
        if avg:
            return np.mean(avg_neigh_degree)
        else:
            return avg_neigh_degree

    def assortativity(self): 
        """
        Attention: I tested this function with random graphs and it always gave 0.11 as output
        """
        adj = self.adj_mat.copy()
        assortativity = fnx.assortativity(adj)
        return assortativity
    
    def manual_assortativity(self):
        """
        Computes the assortativity coefficient following the definition in Leung and Chau 2007,
        "Weighted assortative and disassortative networks model".

        :return: float, Assortativity coefficient of the network
        """
        adj = self.adj_mat.copy()
        total_weights = 1/np.sum(adj)   
        degrees = self.degrees()        
        add_mat = np.zeros(adj.shape)   
        mult_mat = np.zeros(adj.shape)
        square_mat = np.zeros(adj.shape)
        
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                mult_mat[i,j] = degrees[i] * degrees[j]
                add_mat[i,j] = degrees[i] + degrees[j]
                square_mat[i,j] = degrees[i]**2 + degrees[j]**2
        
        num = total_weights * np.sum(adj * mult_mat) - (total_weights/2 * np.sum(adj * add_mat))**2
        denom = total_weights/2 * np.sum(adj * (square_mat)) - (total_weights/2 * np.sum(adj * add_mat))**2
        assortativity = num / denom
        return assortativity
        
    def MST(self):
        """
        Calculate the minimum spannning tree of network, i.e. inverts the edges, takes mst and reinverts the edges.
        Network adjacency mat is changed to Minimum Spanning Tree
        Returns: minimum spanning tree of the network
        """
        # Invert adjacency mat
        inv_adj_mat = self.invert_edges()
        mst = fnx.MST(inv_adj_mat)
        # Re-inverting the mst
        mst = self.invert_edges(adj=mst)
        self.adj_mat = mst.copy()
        return mst
