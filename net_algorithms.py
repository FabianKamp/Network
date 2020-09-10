import numpy as np
import random

def primsAlg(net):
    numnodes = net.shape[0]
    nodes = list(range(numnodes))
    MST = np.zeros(net.shape)
    reached = []
    unreached = nodes.copy()
    start = random.choice(nodes)
    reached.append(start)
    unreached.remove(start)
    while len(unreached) != 0:
        record = np.inf
        for i in range(len(reached)):
            for j in range(len(unreached)):
                score = net[reached[i],unreached[j]]
                if score < record:
                    record = score
                    ireached, iunreached = i, j
        MST[unreached[iunreached], reached[ireached]] = record
        reached.append(unreached[iunreached])
        unreached.pop(iunreached)
    MST += MST.T
    return MST

"""Dijstrak
shortestdist_df = pd.DataFrame(np.zeros(adj_mat.shape), columns=self.nodes, index=self.nodes)  # Initialize Path matrix and distance matrix
            shortestpath_df = pd.DataFrame(np.empty(adj_mat.shape, dtype=str), columns=self.nodes, index=self.nodes)

            for n in range(self.number_nodes):
                node_set=pd.DataFrame({'Distance': np.full((self.number_nodes), np.inf),
                                       'Previous': ['']*(self.number_nodes), 'Path': ['']*(self.number_nodes)}, index=self.nodes)
                node_set.loc[self.nodes[n], 'Distance'] = 0
                unvisited_nodes=self.nodes.copy()

                while unvisited_nodes != []:
                    current=node_set.loc[unvisited_nodes,'Distance'].idxmin()    # Select node with minimal Distance of the unvisited nodes
                    unvisited_nodes.remove(current)
                    for k in self.nodes:
                        dist=node_set.loc[current, 'Distance'] + adj_mat.loc[current, k]
                        if node_set.loc[k,'Distance'] > dist:
                            node_set.loc[k,'Distance'] = dist
                            node_set.loc[k,'Previous'] = current
                shortestdist_df.loc[:, n]=node_set.loc[:,'Distance']
                shortestdist_df.loc[n, :]=node_set.loc[:,'Distance']

                if paths:                     # Create Dataframe with string values for the shortest path between each pair of nodes
                    for k in self.nodes:
                        path=str(k)
                        current=k
                        while node_set.loc[current, 'Previous'] != '':
                            current=node_set.loc[current, 'Previous']
                            path=str(current)+','+path
                        node_set.loc[k,'Path'] = path
                    shortestpath_df.loc[:,n]=node_set.loc[:,'Path']
                    shortestpath_df.loc[n,:]=node_set.loc[:,'Path']
            self.shortest_path_length = shortestdist_df
            self.shortest_path = shortestpath_df