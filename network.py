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

        self.adj_mat = pd.DataFrame(Adjacency_Matrix)
        self.nodes = Node_Names
        self.number_nodes = len(self.nodes)

    def __getitem__(self, index):
        return self.adj_mat.iloc[index]

    def invert_edges(self):
        """
        Edge inversion function --> 1/edge, edge != 0
        :return: adjacency matrix with inversed edges
        """
        adj_mat = self.adj_mat
        # Inverting Non-zero values
        inv_edges = np.copy(1 / adj_mat[adj_mat != 0])
        adj_mat[adj_mat!=0] = inv_edges
        return adj_mat

    def binarize_net(self, threshold):
        pass


    def degree(self):
        """
        Calculate the degree of each node in the network.
        :return n dimensional pd.Series with degrees of all nodes in the network
        """
        diag = np.diagonal(np.array(self.adj_mat))
        degree = self.adj_mat.sum(axis=1) - diag                # Calculate degree and return as pd.Series

        return degree

    def short_path_lengths(self, nx=True, normalize=False):
        """
        Calculate the shortest path between all nodes in the network using Dijstrak Algorithm:
        https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm. If nx is set to true uses the networkX implementation.

        :param: nx boolean value, if true uses networkX to compute the shortest path lengths
                path boolean value, specify if the paths are returned
        :return Dictionary of two nxn dimensional pd.DataFrames with shortest path / shortest distance between all pairs of nodes in the network
        """
        adj_mat = np.asarray(self.adj_mat)

        if not np.all(adj_mat>=0):          # Check for negative values
            print('Shortest Path: Compute absolute value for shortest path length.')
            adj_mat = np.abs(adj_mat)       # Take absolute value of adjacency matrix

        # Normalize the edge weights with the maximum
        if normalize:
            max_weight = np.max(adj_mat.to_numpy())
            adj_mat /= max_weight

        # Inverting Non-zero values
        inv_edges = np.copy(1 / adj_mat[adj_mat != 0])
        adj_mat[adj_mat!=0] = inv_edges

        # NetworkX implementation of the shortest path length
        if nx:
            shortestpath_matrix = fnx.shortest_path_length(adj_mat)
            shortestdist_df = pd.DataFrame(np.asarray(shortestpath_matrix), columns=self.nodes, index=self.nodes)
            self.shortest_path_length = shortestdist_df

        # Manual implementation of Dijstrak Algorithm
        else:

            return shortestdist_df

    def num_triangles(self, normalize=False):
        """
        Calculate sum of triangles edge weights around each node in network.
        The edge weights are normalized with the largest weight in the network
        :return: n dimensional pd.Series
        """
        if self.triangles is not None:
            return self.triangles

        adj_mat = self.adj_mat.copy()                               # Create copy of adjacency mat

        if not np.all(adj_mat>=0):
            print('Number Triangles: Not all edges are positive. Compute absolute edge values.')
            adj_mat = np.abs(adj_mat)

        triangles = pd.Series(np.zeros(self.number_nodes), index=self.nodes)

        if normalize:                                               # Normalizes the weights by the maximum weight.
            max_weight = np.max(adj_mat.to_numpy())
            adj_mat /= max_weight

        all_combinations = combinations(self.nodes, 3)              # Create list of all possible triangles
        m_dict={}
        for combi in all_combinations:
            n1_n2 = adj_mat.loc[combi[0],combi[1]]                  # Get path length between pairs in triangle combination
            n1_n3 = adj_mat.loc[combi[0],combi[2]]
            n2_n3 = adj_mat.loc[combi[1],combi[2]]
            m_dict[combi] = (n1_n2*n1_n3*n2_n3)**(1/3)       # Calculate the triangle sum of the combination and save it in dictionary

        for node in self.nodes:
            triangles.loc[node] = (1/2) * np.sum([m_dict[s] for s in m_dict if node in s])    # Sum all of the triangles that contain the node

        if not normalize:
            self.triangles = triangles    # Only saves triangles if it is not normalized, to prevent problems during small-world computation

        return triangles

    def char_path(self, node_by_node=False, nx=True, normalize=False):
        """
        Calculate the characteristic path length of the network
        :return: Dictionary with average node distance np.array and characteristic path length np.float object
        """

        sum_shrtpath = np.sum(np.asarray(self.shortestpath(nx=nx, normalize=normalize)), axis=-1)                      # Sums Shortest Path Dataframe along axis -1
        avg_shrtpath_node = sum_shrtpath / (self.number_nodes-1)                                    # Divide each element in sum array by n-1 regions
        char_pathlength = np.sum(avg_shrtpath_node) / self.number_nodes

        if node_by_node:
            return avg_shrtpath_node                                                            # Return average shortest path node by node
        else:
            return char_pathlength

    def glob_efficiency(self, nx=True):
        """
        Calculate the global efficiency of the network
        :return: np.float object
        """
        shortestpath = np.array(self.shortestpath(nx=nx))
        np.fill_diagonal(shortestpath, 1.)      # Set diagonal to 1

        inv_shrtpath=1/shortestpath                                                 # Computes shortest path and takes inverse
        np.fill_diagonal(inv_shrtpath, 0)                                           # Set Diagonal from 1 -> 0
        sum_invpath_df = np.sum(inv_shrtpath, axis=-1)                              # Sums Shortest Path Dataframe along axis 1
        avg_invpath = sum_invpath_df / (self.number_nodes-1)                        # Divide each element in sum array by n-1 regions
        glob_efficiency= np.sum(avg_invpath) / self.number_nodes                    # Calculate sum of the sum array and take the average

        return glob_efficiency

    def clust_coeff(self, node_by_node=False, normalize=False, nx=True):
        """
        Calculate the cluster coefficient of the network
        :param: node_by_node boolean value that specifies if the cluster coefficient is computed for each node
                normalize boolean value, specifies if the weights are normalized by the largest weight in network
                nx boolean value, specifies if networkX implementation is used or not
        :return: Network cluster coefficient np.float object or ndim np.array of node by node cluster coefficients
        """
        if not isinstance(normalize, bool): raise ValueError('Normalize must be boolean (True/False).')
        adj = np.array(self.adj_mat)
        np.fill_diagonal(adj, 0)                        # Sets the diagonal to zero
        if not np.all(adj>=0):
            print('Take absolute value of the adjacency matrix to compute cluster coefficient.')
            adj = np.abs(adj)

        if nx:
            node_clust = fnx.clustering(adj, normalize=normalize)

        else:
            num_zeros = np.sum(np.isclose(adj, 0), axis=-1)    # Calculates the number of zeros in each row
            degrees = np.full(self.number_nodes, self.number_nodes) - num_zeros
            triangles = np.array(self.num_triangles(normalize=normalize))

            excl_nodes = np.where(degrees < 2); triangles[excl_nodes] = 0; degrees[excl_nodes] = 2     # Sets traingle sum to zero where degree is below 2
            node_clust = (2* triangles) / (degrees*(degrees-1))
            node_clust = pd.Series(node_clust, index=self.nodes)

        if node_by_node:
            return node_clust
        else:
            net_clust = np.sum(node_clust) / self.number_nodes
            return net_clust

    def transitivity(self):
        """
        Calculate the transitivity of the network
        :return: np.float
        """

        sum_triangles = np.sum(np.asarray(self.num_triangles(normalize=False))*2)     # Multiply sum of triangles with 2 and sum the array
        degrees = np.array(self.degree())
        degrees *= (degrees-1)
        sum_degrees = np.sum(degrees)
        transitivity = sum_triangles / sum_degrees

        return transitivity

    def closeness_centrality(self, avg=False, nx=True):
        """
        Calculate the closeness centrality of each node in network.
        Optionally takes in the shortest average path length of each node, which saves computation time.
        :param: n dimensional array or pd.Series that contains the average shortest path for each node
        :return: ndimensional pd.Series
        """
        node_avg_distance = np.asarray(self.char_path(node_by_node=True, nx=nx))    # Compute average shortest path node by node

        if not np.all(node_avg_distance):                                           # Excluding isolated nodes
            print('Excluding isolated nodes, i.e. nodes with shortest average path length of zero.')
            node_avg_distance[node_avg_distance==0] = np.nan

        close_centr = 1 / node_avg_distance                      # Inverts the average shortest path
        close_centr = pd.Series(close_centr, index=self.nodes)    # Converts to pd.Series

        if avg:
            return np.mean(close_centr)

        return close_centr

    def betweenness_centrality(self, avg=False):
        """
        Calculate the betweenness centrality of each node in network
        :return: ndimensional pd.Series
        """
        betw_centrality = pd.Series(np.zeros(self.number_nodes), index=self.nodes)
        shortest_paths = self.shortestpath(paths=True, nx=False)
        for n in self.nodes:
            counter = 0
            mat = shortest_paths.drop(n, axis=0); mat = mat.drop(n, axis=1)  # Drops the nth column and the nth row.
            substr=','+str(n)+','

            for c in mat.columns:
                for e in mat.loc[:c,c]:                                   # Runs only over the upper half of the matrix
                    if e.find(substr) != -1:
                        counter += 1
            betw_centrality.loc[n]= counter / ((self.number_nodes-1)*(self.number_nodes-2))

        if avg:
            return np.mean(betw_centrality)

        return betw_centrality

    def small_worldness(self, nrandnet=1, niter=10, seed=None, nx=True, method='weighted_random', tc=[], normalize=False):
        """
        Computes small worldness (sigma) of network
        :param: seed: float or integer which sets the seed for random network generation
                niter: int of number of iterations that should be done during network generation
                method: string, defines method for random reference generation, must be either rewired_net, weighted_random or hqs
                tc: timecourse as np.ndarray, must  be set for hqs algorithm
        :return: np.float, small-worldness sigma
        """
        import random_reference as randomnet
        if method not in ['rewired_net', 'hqs', 'weighted_random']:
            raise Exception('Method must be rewired_net, rewired_nx or hqs')

        if method == 'hqs':

            tc = np.array(tc)
            if not tc.any() and self.time_course.any():
                tc = self.time_course
            elif not tc.any(): raise ValueError("Timecourse not specified")

            random_net = randomnet.hqs(tc)
            random_clust_coeff = random_net.clust_coeff(node_by_node=False, normalize=normalize, nx=nx)
            random_char_path = random_net.char_path(node_by_node=False, nx=nx)

        else:

            if nrandnet < 1: raise ValueError("Minimum one random network.")
            random_clust_coeff = []
            random_char_path = []
            for i in range(nrandnet):

                if method == 'rewired_net':
                    random_adj = randomnet.rewired_net(self.adj_mat, niter, seed)
                elif method == 'weighted_random':
                    random_adj = randomnet.weighted_random(self.adj_mat, niter=niter)

                random_net = network(random_adj)                                    # Convert random adj to network
                print(f'{i+1} random network generated.')

                random_clust_coeff.append(random_net.clust_coeff(node_by_node=False, normalize=normalize, nx=nx))       # Compute clustering coeff of random network
                random_char_path.append(random_net.char_path(node_by_node=False, normalize=False,nx=nx))                # Compute characteristic pathlength of random network

            random_clust_coeff = np.mean(random_clust_coeff)                        # Take average of random cluster coefficients
            random_char_path = np.mean(random_char_path)                            # Take average of random characteristic paths

        sig_num = (self.clust_coeff(node_by_node=False, normalize=normalize, nx=nx)/random_clust_coeff)                 # Compute numerator
        sig_den = (self.char_path(node_by_node=False, normalize=False, nx=nx)/random_char_path)                         # Compute denumerator

        sigma = sig_num/sig_den                                                     # Compute sigma

        return sigma


    def modularity(self):
        #TODO find algorithm to find modules in network
        pass

    def avg_neigh_degree(self, avg=True):
        """
        Compute the average neighbour degree following the definition of Rubinov and Sporns 2011
        :return: pd.Series, average neighbour degree of each node
        """

        adj = np.array(self.adj_mat)
        np.fill_diagonal(adj, 0)
        degrees = np.sum(adj, axis=-1)  # Weighted Degrees

        avg_neigh_degree = np.zeros(self.number_nodes)
        for n in range(self.number_nodes):
            avg_neigh_degree[n] = np.sum(adj[n, :] * degrees) / degrees[n]

        avg_neigh_degree = pd.Series(avg_neigh_degree, index=self.nodes)

        if avg:
            return np.mean(avg_neigh_degree)

        return avg_neigh_degree

    def assortativity(self):
        """
        Computes the assortativity coefficient following the definition in Leung and Chau 2007,
        "Weighted assortative and disassortative networks model".

        :return: float, Assortativity coefficient of the network
        """

        adj = np.array(self.adj_mat)    # Copy adjacency matrix
        np.fill_diagonal(adj, 0)        # Fills the diagonal with zeros
        assert np.any(adj), "Adjacency matrix must contain non zero values"

        total_weights = 1/np.sum(adj)   # inverse of total weights
        degrees = np.sum(adj, axis=-1)  # Calculate the degree of all edges

        add_mat = np.zeros(adj.shape)   # Initiate matrices for later calculations
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

    def MST(self, invert=True):
        """
        Calculate the minimum spannning tree of network, inverts the edges by default
        Returns: minimum spanning tree of the network
        """
        from from_networkx import convert_to_nx, convert_to_net
        adj_mat = np.copy(self.adj_mat)
        adj_mat += 1
        # Inverting Non-zero values
        if invert:
            inv_edges = 1 / adj_mat[adj_mat != 0]
            adj_mat[adj_mat!=0] = inv_edges
        # Use networkx to calculate the mst
        graph = convert_to_nx(adj_mat)
        mst = nx.minimum_spanning_tree(graph, weight='weight')
        mst = convert_to_net(mst)
        # Re-inverting the mst
        if invert:
            inv_edges = 1 / mst[mst != 0] - 1
            mst[mst!=0] = inv_edges
        return mst






