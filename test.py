import networkx as nx
import numpy as np
from from_networkx import convert_to_nx, convert_to_net

a = np.random.rand(5,5)
print(a)
graph = convert_to_nx(a)
mst = nx.minimum_spanning_tree(graph, weight='weight')
result = convert_to_net(mst)
print(result)