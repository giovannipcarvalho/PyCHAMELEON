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
    # return graph.subgraph(nodes)
    return nodes

def connecting_edges(partitions, graph):
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set

def min_cut_bisector(graph):
    partitions = nx.algorithms.community.kernighan_lin_bisection(graph)
    return connecting_edges(partitions, graph)

def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]['weight'] for edge in edges]

def bisection_weights(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights

def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))

def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)

def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)

def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))

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