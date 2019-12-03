import itertools
from graphtools import *

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

    listToTraverse = [ [] ] * len(clusters)
    for i in range(len(clusters)):
        listToTraverse[i] = get_cluster(graph,[i])
        
    for i in range(len(clusters)):
        j=i+1
        while j < (len(clusters)): 
            if verbose: print "Checking c%d c%d" % (i, j)
            ms = merge_score(graph, listToTraverse[i], listToTraverse[j], a)
            if verbose: print "Merge score: %f" % (ms)
            if ms > max_score:
                if verbose: print "Better than: %f" % (max_score)
                max_score = ms
                ci, cj = clusters[i], clusters[j]
            j+=1  
    
    """
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
    """
