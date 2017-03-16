import seaborn as sns
import matplotlib.pyplot as plt

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