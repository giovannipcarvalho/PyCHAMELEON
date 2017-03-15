import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

import metis

def plot_points(points):
    df = pd.DataFrame(points, columns=['x', 'y'])
    df.plot(kind='scatter', x='x', y='y')
    plt.show(block=False)

def plot_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')
    colors = nx.get_node_attributes(graph, 'color').values()
    if colors: # is set
        nx.draw(graph, pos, node_color=colors)
    else:
        nx.draw(graph, pos)
    plt.show(block=False)

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def knn_graph(points, k):
    g = nx.Graph()
    for i, p in enumerate(points):
        distances = map(lambda x: euclidean_distance(p, x), points)
        closests = np.argsort(distances)[1:k+1] # second trough kth closest
        for c in closests:
            g.add_edge(i, c, weight=distances[c])
        g.node[i]['pos'] = p
    return g

colors = ['red', 'green', 'blue', 'pink', 'orange', 'black', 'gray']
def part_graph(graph, k):
    edgecuts, parts = metis.part_graph(graph, k)
    for i, p in enumerate(parts):
        graph.node[i]['color'] = colors[p]
    return graph

def empty_graph_copy(graph):
    g = nx.create_empty_copy(graph)
    for i in range(len(graph.node)):
        g.node[i]['pos'] = graph.node[i]['pos']
        g.node[i]['color'] = graph.node[i]['color']
    return g

def get_cluster(graph, c_colors):
    nodes = [n for n in graph.node if graph.node[n]['color'] in c_colors]
    return graph.subgraph(nodes)

def internal_interconnectivity(graph):
    # average weight of minimum cut edges
    edges = nx.minimum_edge_cut(graph)
    weights = [graph[edge[0]][edge[1]]['weight'] for edge in edges]
    return np.mean(weights)

def internal_closeness(graph):
    pass

if __name__ == "__main__":
    # get a set of data points
    points = [
        (1, 1),
        (1, 2),
        (0, 1),

        (2.1, 1),
        (2.2, 1.5),

        (3, 3),
        (3, 4),
        (4, 3),

        (5, 5),
        (5, 4),
        (4, 5)
    ]

    # plot_points(points)

    # generate KNN graph
    graph = knn_graph(points, 2)

    # partition the graph
    graph = part_graph(graph, 4)

    # recombine clusters