import numpy as np
import pandas as pd
import networkx as nx

import pickle
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

import metis

def plot_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')
    c = [colors[i%(len(colors))] for i in nx.get_node_attributes(graph, 'cluster').values()]
    if c: # is set
        nx.draw(graph, pos, node_color=c, node_size=0.2)
    else:
        nx.draw(graph, pos)
    plt.show(block=False)

def plot_data(df):
    df.plot(kind='scatter', c=df['cluster'], cmap='gist_rainbow', x='x', y='y')
    plt.show(block=False)

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def knn_graph(df, k):
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i, p in enumerate(points):
        distances = map(lambda x: euclidean_distance(p, x), points)
        closests = np.argsort(distances)[1:k+1] # second trough kth closest
        for c in closests:
            g.add_edge(i, c, weight=distances[c])
        g.node[i]['pos'] = p
    return g

def part_graph(graph, k, df=None):
    edgecuts, parts = metis.part_graph(graph, k)
    for i, p in enumerate(graph.nodes()):
        graph.node[p]['cluster'] = parts[i]
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph

def get_cluster(graph, clusters):
    nodes = [n for n in graph.node if graph.node[n]['cluster'] in clusters]
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
    graph = graph.copy()
    graph = part_graph(graph, 2)
    partitions = get_cluster(graph, [0]), get_cluster(graph, [1])
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
    if not edges:
        SEC = 1e4
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))

merge_score = lambda g,ci,cj,a: relative_interconnectivity(g,ci,cj) * np.power(relative_closeness(g,ci,cj), a)

def merge_best(graph, df, a, verbose=False):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose: print "Checking c%d c%d" % (i, j)
            ms = merge_score(graph, get_cluster(graph, [i]), get_cluster(graph, [j]), a)
            if verbose: print "Merge score: %f" % (ms)
            if ms > max_score:
                if verbose: print "Better than: %f" % (max_score)
                max_score = ms
                ci, cj = i, j
    
    if max_score > 0:
        if verbose: print "Merging c%d and c%d" % (ci, cj)
        df.loc[df['cluster'] == cj, 'cluster'] = ci
    return max_score > 0

if __name__ == "__main__":
    # get a set of data points
    df = pd.read_csv('two_squares.csv', delimiter=' ', header=None, names=['x', 'y'])
    
    # create knn graph
    graph = knn_graph(df, 6)

    # partition graph
    graph = part_graph(graph, 6, df)
    
    # merge clusters
    while merge_best(graph, df, 1, verbose=True):
        plot_data(df)