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

def rewire(adjacency, niter=10, seed=None):
    """
    Takes in adjacency matrix, translates it into a networkX network, generates a random reference network
    :param adjacency: nxn dimensional adjacency matrix
    :param niter: number of rewiring iterations
    :param seed: seed for random number generation
    :return: pd.DataFrame of rewired adjacency matrix
    """
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    if isinstance(adjacency, pd.DataFrame):
        nodes = list(adjacency.index)
    else:
        nodes = np.arange(adjacency.shape[0])
    graph = nx.from_numpy_matrix(np.asarray(adjacency))                 # Convert to networkX graph
    random_nx = nx.random_reference(graph, niter=niter, seed=seed)      # Generate random reference using networkX
    random_adj = nx.to_numpy_matrix(random_nx)                             # Convert to numpy
    random_adj = pd.DataFrame(random_adj, columns=nodes, index=nodes)   # Converts to pd.DataFrame

    return random_adj

def assortativity(self):
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

def betweenness_centrality(self, avg=False):
    """
    Calculate the betweenness centrality of each node in network
    :return: ndimensional pd.Series
    """
    betw_centrality = np.zeros(self.number_nodes)
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