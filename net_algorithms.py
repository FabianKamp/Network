import numpy as np
import random
import pandas as pd

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

def dijstrak(adj_mat, paths=True):
    number_nodes = adj_mat.shape[0]
    nodes = np.arange(number_nodes)
    shortestdist_df = pd.DataFrame(np.zeros(adj_mat.shape), columns=nodes, index=nodes)  # Initialize Path matrix and distance matrix
    shortestpath_df = pd.DataFrame(np.empty(adj_mat.shape, dtype=str), columns=nodes, index=nodes)

    for n in range(number_nodes):
        node_set=pd.DataFrame({'Distance': np.full((number_nodes), np.inf),
                               'Previous': ['']*(number_nodes), 'Path': ['']*(number_nodes)}, index=nodes)
        node_set.loc[nodes[n], 'Distance'] = 0
        unvisited_nodes=nodes.copy()

        while unvisited_nodes != []:
            current=node_set.loc[unvisited_nodes,'Distance'].idxmin()    # Select node with minimal Distance of the unvisited nodes
            unvisited_nodes.remove(current)
            for k in nodes:
                dist=node_set.loc[current, 'Distance'] + adj_mat.loc[current, k]
                if node_set.loc[k,'Distance'] > dist:
                    node_set.loc[k,'Distance'] = dist
                    node_set.loc[k,'Previous'] = current
        shortestdist_df.loc[:, n]=node_set.loc[:,'Distance']
        shortestdist_df.loc[n, :]=node_set.loc[:,'Distance']

        if paths: # Create Dataframe with string values for the shortest path between each pair of nodes
            for k in nodes:
                path=str(k)
                current=k
                while node_set.loc[current, 'Previous'] != '':
                    current=node_set.loc[current, 'Previous']
                    path=str(current)+','+path
                node_set.loc[k,'Path'] = path
            shortestpath_df.loc[:,n]=node_set.loc[:,'Path']
            shortestpath_df.loc[n,:]=node_set.loc[:,'Path']

    if paths:
        return shortestpath_df
    else:
        return shortestdist_df
