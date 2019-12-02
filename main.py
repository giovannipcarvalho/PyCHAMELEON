import pandas as pd

from visualization import *
from chameleon import *

if __name__ == "__main__":
    # get a set of data points
    df = pd.read_csv('./datasets/two_squares.csv', delimiter=' ', header=None, names=['x', 'y'])
    
    # create knn graph
    graph = knn_graph(df, 6, verbose=True)

    # partition graph
    graph = part_graph(graph, 6, df)

    # merge clusters
    while merge_best(graph, df, 1, verbose=True):
        plot_data(df)
