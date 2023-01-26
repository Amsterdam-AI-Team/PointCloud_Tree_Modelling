# Tree_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

""" 
Graph utility methods - Module (Python)
"""

import numpy as np
import networkx as nx
from plyfile import PlyData


def read_ply(ply_file):
    """Function to read graph from ply file."""

    plydata = PlyData.read(ply_file)

    # vertices
    vertices = np.array([[c for c in p] for p in plydata['vertex'].data])
    vertices, reverse_ = np.unique(vertices, axis=0, return_inverse=True)

    # edges
    edges = np.array([reverse_[edge[0]] for edge in plydata['edge'].data])

    # construct graph
    graph = nx.DiGraph()
    for i, vertex in enumerate(vertices):
        graph.add_node(i, x=vertex[0],y=vertex[1],z=vertex[2])
    for i,j in edges:
        graph.add_edge(j, i)

    return graph, vertices, edges


def path_till_split(graph, start_node):
    """Function to retrieve graph path from `start_node` till first split."""
    
    path = [start_node] # TODO check if start exists
    while graph.out_degree(path[-1]) == 1:
        for node in graph.successors(path[-1]):
            path.append(node)
            break

    return path
