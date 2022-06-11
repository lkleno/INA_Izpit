from collections import deque
import time
from collections import Counter
from tqdm import tqdm
from scipy import stats

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sb

from time import sleep
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

def read_net(folder, file_name):
    """Read network"""
    G = nx.Graph(name=file_name)  # define empty graph
    with open(os.path.join(folder, file_name), 'r') as f:

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str), int(node2_str))

    return G

#G = read_net("./", "izpitG.txt")

#ii) ################################################################################################################################################################
# I. Adjacency list representation
def read_nets(folder, files_lst, read_net_func):
    # general function for reading networks from folder
    nets = []
    for file in files_lst:
        nets.append(read_net_func(os.path.join(folder, file)))
    return nets

def read_net_undirected(file_path):
    # return graph as adj list
    with open(file_path) as f:
        lines = f.readlines()
        # get nodes number
        node_num = int(lines[0].split()[1])
        # initialization
        is_nodes_finished = False
        G = [[] for _ in range(node_num)]
        # check for end
        for line in lines[1:]:
          if line.startswith('*'):
              is_nodes_finished = True
              continue
          # check if nodes are read
          if is_nodes_finished:
              node1_str, node2_str = line.split()[:2]
              # read edge as a pair of nodes
              node1, node2 = int(node1_str)-1, int(node2_str)-1
              # reduce the nodes ids by 1
              G[node1].append(node2)
              G[node2].append(node1)
              # add the edge to graph
    return G

def read_net_directed(file_path):
    # return graph as adj list
    with open(file_path, 'r') as f:
        for line in f:
            test = 1
        lines = f.readlines()
        # get nodes number
        node_num = int(lines[0].split()[1])
        # initialization
        is_nodes_finished = False
        G = [[] for _ in range(node_num)]
        # check for end
        for line in lines[1:]:
          if line.startswith('*'):
              is_nodes_finished = True
              continue
          # check if nodes are read
          if is_nodes_finished:
              node1_str, node2_str = line.split()[:2]
              # read edge as a pair of nodes
              node1, node2 = int(node1_str)-1, int(node2_str)-1
              # reduce the nodes ids by 1
              G[node1].append(node2)
              # add the edge to graph
    return G

a = read_net_directed('C:\\Users\\kleno\\OneDrive\\Desktop\\INA izpit\\tekst.txt')
s = 1
#II. Basic network statistics

def calc_basic_metrics(G, undirected=True):
    # return number of nodes, number of edges, average degree, density
    n = len(G)
    m = sum([len(node_neighbors) for node_neighbors in G])
    if undirected:
      m//=2
    return n, m, 2*m/n, 2*m/n/(n-1)

"""for i, G in enumerate([toy_G, karate_G, google_G]):
    n, m, degree, density = calc_basic_metrics(G)
    print(f'Graph {files[i]}:')
    print('#Nodes:', n)
    print('#Edges:', m)
    print('Average node degree:', degree)
    print('Undirected density:', density)
    print('------')"""

def is_isolated_node(G, i):
    # return if (i) is isolated node in G
    for j in G[i]:
      if j != i:
        return False
    return True

def calc_nodes_statistics(G):
    # return number of isolated nodes, number of pendant nodes, maximum degree
    n0, n1, k_max = 0, 0, 0
    for i in range(len(G)):
      if is_isolated_node(G, i):
        n0 += 1
      elif len(G[i]) == 1:
        n1 += 1
      if len(G[i]) > k_max:
        k_max = len(G[i])
    return n0, n1, k_max

"""for i, G in enumerate([toy_G, karate_G, google_G]):
    n0, n1, k_max = calc_nodes_statistics(G)
    print(f'Graph {files[i]}:')
    print('#Isolated nodes:', n0)
    print('#Degree-1 nodes:', n1)
    print('Maximum node degree:', k_max)
    print('------')"""

#III. Network connected components
def get_component(G, N, i):
    # return list of nodes in connected component containing (i)
    C = []
    S = []
    S.append(i)
    N.remove(i)
    while S:
      i = S.pop()
      C.append(i)
      for j in G[i]:
        if j in N:
          N.remove(j)
          S.append(j)
    return C

def get_components(G):
    # return list of connected components
    C = []
    N = set(range(len(G)))
    while N:
      C.append(get_component(G, N, next(iter(N))))
    return C

max_component_size = 0
"""for i, G in enumerate([toy_G, karate_G, google_G]):
    print(f'Graph {files[i]}:')
    components = get_components(G)
    max_component_size_G = 0
    for component in components:
        if len(component) > max_component_size_G:
            max_component_size_G = len(component)
    if max_component_size_G > max_component_size:
        max_component_size = max_component_size_G
    print('Max component size:', max_component_size_G)
    print('------')

print()
print('Max component size over all graphs:', max_component_size)"""

#iii) ################################################################################################################################################################
def read_net(folder, file_name):
    """Read network"""
    G = nx.Graph(name=file_name)  # define empty graph
    with open(os.path.join(folder, file_name), 'r') as f:
        f.readline()  # skip the first line

        # add nodes
        for line in f:
            if line.startswith('*'):
                break
            else:
                node = line.split()
                G.add_node(int(node[0]) - 1, label=node[1].strip('\"'))

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str) - 1, int(node2_str) - 1)

    return G

#1. Number and size of connected components

def get_component(G, N, i):
    """Return list of nodes in connected component containing (i)"""
    C = []
    S = []
    S.append(i)
    N.remove(i)
    while S:
      i = S.pop()
      C.append(i)
      for j in G[i]:
        if j in N:
          N.remove(j)
          S.append(j)
    return C

def get_components(G):
    """Return list of connected components"""
    C = []
    N = set(range(len(G)))
    while N:
      C.append(get_component(G, N, next(iter(N))))
    return C

max_component_size = 0
"""for G in [toy_G, karate_G, imdb_G, google_G]:
    print(f'Graph {G.name}:')
    start_time = time.time()
    components = get_components(G)
    print("Time: {:.1f}s".format(time.time() - start_time))
    max_component_size_G = 0
    for component in components:
        if len(component) > max_component_size_G:
            max_component_size_G = len(component)
    if max_component_size_G > max_component_size:
        max_component_size = max_component_size_G
    print('#components:', len(components))
    print(f'Max component size: {max_component_size_G}/{len(G.nodes())}',
          '({:.2f}% of nodes)'.format(100*max_component_size_G/len(G.nodes())))
    print('------')

print()
print('Max component size over all graphs:', max_component_size)"""

#2. Average node distance and network diameter

def get_distance_for_node(G, i):
    """Return list of distances between node i and others"""
    # empty array
    D = [-1] * len(G)
    Q = deque()
    D[i] = 0
    Q.append(i)
    while Q:
        i = Q.popleft()
        for j in G[i]:
            if D[j] == -1:
                D[j] = D[i] + 1
                Q.append(j)
    return [d for d in D if d > 0]


def get_distances_between_nodes(G, n_max=100):
    """Return list of list of distances between nodes (maximum n_max nodes)"""
    nodes = G.nodes()
    D = []
    for i in nodes:
        D.append(get_distance_for_node(G, i))
    return D

"""for G in [toy_G, karate_G, imdb_G, google_G]:
    print(f'Graph {G.name}:')
    start_time = time.time()
    distances = get_distances_between_nodes(G)
    print("Time: {:.1f}s".format(time.time() - start_time))
    distances = [i for d in distances for i in d]
    print("Avg distance: {:.2f}".format(sum(distances) / len(distances)))
    print("Diameter:", max(distances))
    print('------')"""

# 3. Average node clustering coefficient

from networkx.generators import intersection


def get_link_triads(G, i, j):
    """Return link triads containing nodes i and j"""
    return len(set(G[i]).intersection(G[j]))


def get_node_triads(G, i):
    """Return triads containing node i"""
    t = 0
    for j in G[i]:
        if G.degree(i) <= G.degree(j):
            t += get_link_triads(G, i, j) / 2
        else:
            t += get_link_triads(G, j, i) / 2
    return t


def get_node_clustering_coef(G, i):
    """Return node clustering coef for node i"""
    k_i = G.degree[i]
    if k_i <= 1:
        return 0
    return get_node_triads(G, i) / (k_i * (k_i - 1) / 2)


def get_avg_clustering_coef(G, n_max=100):
    """Return average clustering coef for graph G (maximum n_max nodes)"""
    # select <= n_max nodes
    nodes = G.nodes() if len(G) <= n_max else random.sample(G.nodes(), n_max)

    """Return average clustering coef for graph G"""
    C = 0
    for i in nodes:
        C += get_node_clustering_coef(G, i)
    return C / len(nodes)

"""for G in [toy_G, karate_G]:  # only for small networks
    print(f'Graph {G.name}:')
    start_time = time.time()
    avg_clustering_coef = get_avg_clustering_coef(G)
    print("Time: {:.1f}s".format(time.time() - start_time))
    print('#Average clustering coef: {:.4f}'.format(avg_clustering_coef))
    print('------')"""

#4. Erdös-Rényi random graphs and link indexing

def build_erdos_renyi_graph(n, m):
    """Build erdos-renyi graph with n nodes and m edges"""
    H = set()
    G = nx.Graph(name=f"erdos_renyi_{n}_{m}")
    for i in range(n):
      G.add_node(i, label=str(i+1))
    edges=[]
    while len(edges)<m:
      i = random.randint(0,n-1)
      j = random.randint(0,n-1)
      if i!=j:
        edges.append((i,j))
    G.add_edges_from(edges)
    return G


def print_graph_info(G):
    """Print all statistics of graph G"""
    print(f'Graph {G.name}:')

    # makes simple graph from multigraph
    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    start_time = time.time()
    avg_clustering_coef = get_avg_clustering_coef(G)
    distances = get_distances_between_nodes(G)
    print("Computing time: {:.1f}s".format(time.time() - start_time))
    distances = [i for d in distances for i in d]
    print("Avg distance {:.2f}".format(sum(distances) / len(distances)))
    print("Diameter", max(distances))
    print("Avg clustering coef {:.4f}".format(avg_clustering_coef))
    print('=========')


random_graphs = []
"""for G in [toy_G, karate_G]:
    print(f'Original graph {G.name}:')
    start_time = time.time()

    # choose function to use
    random_G = build_erdos_renyi_graph(G.number_of_nodes(), G.number_of_edges())

    print('Building random graph time: {:.1f}s'.format(time.time() - start_time))
    print()
    print_graph_info(random_G)
    random_graphs.append(random_G)"""

#5. Configuration model graphs and link rewiring

def build_simple_configuration(original_edges):
    "Return simple configuration"""
    G = nx.MultiGraph(name="simple_configuration")
    H = set(original_edges)

    # shuffle original_edges
    pass  # TODO

    # shuffle original_edges
    pass  # TODO

    # add edges
    edges = []
    for i in range(0, len(original_edges), 2):
        # (i,j) = original_edges[i], (s,t) = original_edges[i+1]
        pass  # TODO

    G.add_edges_from(edges)
    return G


def build_multi_configuration(degrees):
    """Return pseudo configuration"""
    G = nx.MultiGraph(name="pseudo_configuration")
    # add nodes
    nodes = []
    pass  # TODO

    # shuffle nodes
    pass  # TODO

    # add edges
    edges = []
    for i in range(0, len(nodes), 2):
        pass  # TODO

    G.add_edges_from(edges)
    return G


configuration_graphs = []
"""for G in [toy_G, karate_G, imdb_G, google_G]:
    print(f'Original graph {G.name}:')
    start_time = time.time()

    # choose function to use
    configuration_G = None  # TODO

    print('Building random graph time: {:.1f}s'.format(time.time() - start_time))
    print()
    print_graph_info(configuration_G)
    configuration_graphs.append(configuration_G)"""

#iv) ################################################################################################################################################################

def draw_top_centrality_bar(G, node_centrality, title='', width=15, height=6):
    """
    Plots bar chart for nodes and their centrality values

    Parameters
    ------
    G: nx.Graph
    node_centrality: list of pairs (node, centrality value)
    """

    # seperate lists of nodes and centrality values
    nodes, weights = zip(*node_centrality)
    # get nodes' labels (names of the actors)
    node_to_label = nx.get_node_attributes(G, 'label')
    labels = [node_to_label[node] for node in nodes]

    # create bar chart
    f, ax = plt.subplots(figsize=(width,height))
    plt.bar(labels,weights)
    plt.xticks(labels, rotation=90)
    plt.ylabel('Centrality')
    plt.title(title)
    plt.show()

def plot_graph_with_node_weights(G, nodes_centrality, title='', node_size=10000, layout_fn=nx.spring_layout, width=15, height=8):
    """
    Plots subgraph by nodes in nodes_centrality

    Parameters
    ------
    G: nx.Graph
    node_centrality: list of pairs (node, centrality value)
    layout_fn: function from networkx for positioning of nodes
    """

    # add description to title
    title += ': the darker the node, the higher its centrality value'

    # select subgraph, node weights and positions
    node_to_centrality = dict(nodes_centrality)
    sub_G = G.subgraph(node_to_centrality.keys())
    weights = np.array([node_to_centrality[node] for node in sub_G.nodes()])
    pos = layout_fn(sub_G)

    # plot subgraph
    plt.figure(1, figsize=(width, height))
    nx.draw_networkx(
        sub_G,
        pos=pos,
        nodelist=sub_G.nodes(),
        node_size=node_size,
        node_color=weights,
        labels=nx.get_node_attributes(sub_G, 'label'),
        font_size=10,
        font_color='blue',
        cmap=plt.cm.Oranges,
    )
    plt.title(title)
    plt.show()


def visualize_top_centrality_scores(G, nodes_centrality, title, node_size, layout_fn=nx.spring_layout, width=18, height=9):
    """Combine calls of functions for visualization"""
    draw_top_centrality_bar(G, nodes_centrality, title)
    print()
    print()
    plot_graph_with_node_weights(G,
                                 nodes_centrality,
                                 title,
                                 node_size=node_size,
                                 layout_fn=layout_fn,
                                 width=width,
                                 height=height)

def read_net(folder, file_name):
    """Read network"""
    G = nx.Graph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node = line.split('"')
                G.add_node(int(node[0]) - 1, label = node[1].strip())
        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G

#1. Degree centrality and clustering coefficients

def sort_node_centrality_list(node_centrality):
    """Sorts list of pairs by the second element in pairs"""
    return sorted([(i, c) for i, c in node_centrality], key=lambda x: x[1], reverse=True)


def calc_degree_centrality(G):
    """
    Calcaulates degree centrality for each node

    Return
    ------
    sorted list of pairs (node, degree centrality value) in descending order
    """
    node_centrality = [(i, G.degree(i) / (G.number_of_nodes() - 1)) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

"""nodes_cnt_to_show = 15  # number of top nodes to plot
title = 'Top nodes by Degree centrality'  # title of the chart
node_size = 10000  # size of each node

start_time = time.time()
degree_node_centrality = calc_degree_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                degree_node_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

def get_link_triads(G, i, j):
    """Return link triads containing nodes i and j"""
    t = len(set(G[i]).intersection(G[j]))
    return t


def get_node_triads(G, i):
    """Return triads containing node i"""
    t = 0
    for j in G[i]:
        if G.degree[i] <= G.degree[j]:
            t += get_link_triads(G, i, j) / 2
        else:
            t += get_link_triads(G, j, i) / 2
    return t


def get_node_clustering_coef(G, i):
    """Return node clustering coef for node i"""
    k_i = G.degree[i]
    if k_i <= 1:
        return 0
    return get_node_triads(G, i) * 2 / (k_i*k_i-k_i)


def calc_cluster_centrality(G):
    """
    Calcaulates cluster coeficient for each node

    Return
    ------
    sorted list of pairs (node, cluster coeficient) in descending order
    """
    node_centrality = [(i, get_node_clustering_coef(G,i)) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

"""nodes_cnt_to_show = 15
title = 'Top nodes by Cluster centrality'
node_size = 10000

start_time = time.time()
cluster_node_centrality = calc_cluster_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                cluster_node_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

def calc_mu_cluster_centrality(G):
    """
    Calcaulates µ-corrected cluster coeficient for each node

    Return
    ------
    sorted list of pairs (node,  µ-corrected coeficient) in descending order
    """
    mu = max([get_link_triads(G,i,j) for i,j in G.edges()])
    node_centrality = [(i, get_node_clustering_coef(G,i) * (G.degree(i) - 1) / mu) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

"""nodes_cnt_to_show = 15
title = 'Top nodes by µ-corrected cluster centrality'
node_size = 10000

start_time = time.time()
cluster_mu_node_centrality = calc_mu_cluster_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                cluster_mu_node_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

#2. Closeness and betweenness centrality

def get_distance_for_node(G, i):
    D = [-1] * len(G)
    Q = deque()
    D[i] = 0
    Q.append(i)
    while Q:
        i = Q.popleft()
        for j in G[i]:
            # if j not in D:
            if D[j] == -1:
                D[j] = D[i] + 1
                Q.append(j)
    return [d for d in D if d > 0]


def calc_closeness_centrality(G):
    """
    Calcaulates closeness centrality for each node

    Return
    ------
    sorted list of pairs (node,  closeness centrality) in descending order
    """
    node_centrality = [(i, sum([1/d for d in get_distance_for_node(G,i)])/ (G.number_of_nodes()-1)) for i in tqdm(G.nodes())]
    return sort_node_centrality_list(node_centrality)

"""nodes_cnt_to_show = 15
title = 'Top nodes by Closeness centrality'
node_size = 10000

start_time = time.time()
closeness_centrality = calc_closeness_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                closeness_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

"""def calc_betweenness_centrality(G):

    ##Calcaulates betweenness centrality for each node

    #Return
    #------
    #sorted list of pairs (node,  betweenness centrality) in descending order
    
    node_to_centrality = nx.betweenness_centrality(imdb_G)
    node_centrality= node_to_centrality.items()
    return sort_node_centrality_list(node_centrality)"""

"""nodes_cnt_to_show = 15
title = 'Top nodes by Betweenness centrality'
node_size = 10000

start_time = time.time()
betweenness_centrality = calc_betweenness_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                betweenness_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

#3. Eigenvector centrality and PageRank algorithm
def calc_eigenvector_centrality(G, eps=1e-6):
    """
    Calcaulates eigenvector centrality for each node

    Return
    ------
    sorted list of pairs (node,  eigenvector centrality) in descending order
    """

    # define list of default centrality values for each node (set value = 1)
    E = [1] * G.number_of_nodes()

    # define default difference between old and new values (set value > eps)
    diff = 1

    while diff > eps:
        # U = sum of neighboring centrality values for each node
        U = [sum([E[j] for j in G[i]]) for i in G.nodes()]

        # calculate normalizing constant u
        u = sum(U)

        # update all nodes' values in U according to the algo
        U = [U[i] * G.number_of_nodes()/u for i in G.nodes()]

        # update diff = sum of old value - new value over each node
        diff = sum([abs(E[i]-U[i]) for i in G.nodes()])

        # update E
        E = U

    node_centrality= [(i,E[i]) for i in range(len(E))]
    return sort_node_centrality_list(node_centrality)

"""nodes_cnt_to_show = 15
title = 'Top nodes by Eigenvector centrality'
node_size = 10000

start_time = time.time()
eigenvector_centrality = calc_eigenvector_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                eigenvector_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

def calc_pagerank_centrality(G, alpha=0.85, eps=1e-6):
    """
    Calcaulates pagerank centrality for each node

    Return
    ------
    sorted list of pairs (node,  pagerank centrality) in descending order
    """

    # define list of default centrality values for each node (set value = 1/|nodes|)
    P = [1 / G.number_of_nodes()] * G.number_of_nodes()

    # define default difference between old and new values (set value > eps)
    diff = 1

    while diff > eps:
        # init U according to the formula in the algo
        U = [sum([P[j] * alpha / G.degree(j) for j in G[i]]) for i in G.nodes()]

        # calculate normalizing constant u
        u = sum(U)

        # update all nodes' values in U according to the algo
        U = [U[i] + (1-u) / G.number_of_nodes()  for i in G.nodes()]

        # update diff = sum of old value - new value over each node
        diff = sum([abs(P[i]-U[i]) for i in G.nodes()])

        # update P
        P = U

    # build sorted list of pairs (node, centrality)
    node_centrality = [(i, P[i]) for i in range(len(P))]
    return sort_node_centrality_list(node_centrality)

"""nodes_cnt_to_show = 15
title = 'Top nodes by PageRank centrality'
node_size = 10000

start_time = time.time()
pagerank_centrality = calc_pagerank_centrality(imdb_G)
print("Time: {:.1f}s".format(time.time() - start_time))

visualize_top_centrality_scores(imdb_G,
                                pagerank_centrality[:nodes_cnt_to_show],
                                title,
                                node_size)"""

#v) ################################################################################################################################################################

def draw_top_centrality_bar(G, edge_centrality, title='', width=15, height=6):
    """
    Plots bar chart for edges and their centrality values

    Parameters
    ------
    G: nx.Graph
    edge_centrality: list of pairs ((source, target), centrality value)
    """

    # seperate lists of nodes and centrality values
    edges, weights = zip(*edge_centrality)

    # get nodes' labels
    node_to_label = nx.get_node_attributes(G, 'label')
    labels = [f"{node_to_label[node1]}-{node_to_label[node2]}" for node1, node2 in edges]

    # create bar plot
    f, ax = plt.subplots(figsize=(width, height))
    plt.bar(labels, weights)
    plt.xticks(labels, rotation=90)
    plt.ylabel('centrality')
    plt.title(title)
    plt.show()


def scale_weights(weights, min_new, max_new):
    """Scales weights from [min(weights), max(weights)] to [min_new, max_new]"""
    min_old = min(weights)
    max_old = max(weights)
    diff_old = max_old - min_old
    diff_new = max_new - min_new
    return [weight * diff_new / diff_old + min_new - min_old * diff_new / diff_old
            for weight in weights]


def plot_graph_with_node_weights(G, edges_centrality, title='', node_size=10000, layout_fn=nx.spring_layout, width=15,
                                 height=8):
    """
    Plots subgraph induced by edges in edges_centrality

    Parameters
    ------
    G: nx.Graph
    edges_centrality: list of pairs ((source, target), centrality value)
    layout_fn: function from networkx for positioning of nodes
    """

    # add description to title
    title += ': the wider the edge, the higher its centrality value'

    # select subgraph, node weights and positions
    node_to_label = nx.get_node_attributes(G, 'label')
    edge_to_centrality = dict(edges_centrality)
    sub_G = nx.from_edgelist(edge_to_centrality.keys())

    edge_weights = []
    for node1, node2 in sub_G.edges():
        edge_key = (node1, node2) if (node1, node2) in edge_to_centrality else (node2, node1)
        edge_weights.append(edge_to_centrality[edge_key])
    pos = layout_fn(sub_G)

    # plot subgraph
    plt.figure(1, figsize=(width, height))
    nx.draw_networkx(
        sub_G,
        pos=pos,
        nodelist=sub_G.nodes(),
        node_size=node_size,
        node_color='yellow',
        labels={node: node_to_label[node] for node in sub_G.nodes()},
        font_size=8,
        width=scale_weights(edge_weights, 0.2, 3),
        font_color='blue',
    )
    plt.title(title)
    plt.show()


def visualize_top_centrality_scores(G, nodes_centrality, title, node_size, layout_fn=nx.spring_layout, width=18,
                                    height=9):
    """Combine calls of functions for visualization"""
    draw_top_centrality_bar(G, nodes_centrality, title)
    print()
    print()
    plot_graph_with_node_weights(G,
                                 nodes_centrality,
                                 title,
                                 node_size=node_size,
                                 layout_fn=layout_fn,
                                 width=width,
                                 height=height)


def visualize_graph(G, node_size, width=18, height=9, layout_fn=nx.spring_layout):
    """
    Plots the whole graph

    Parameters
    ------
    layout_fn: function from networkx for positioning of nodes
    """
    pos = layout_fn(G)
    plt.figure(figsize=(width, height))
    nx.draw_networkx(
        G,
        pos=pos,
        nodelist=G.nodes(),
        node_size=node_size,
        labels=dict(zip(G.nodes(), G.nodes())),  # node labels to print
        font_color='white',
        font_size=10,
    )
    plt.axis("off")
    plt.show()

def read_net(folder, file_name):
    """Read network"""
    G = nx.MultiGraph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node = line.split('"')
                G.add_node(int(node[0]) - 1, label = node[1].strip())
        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G


def sort_edge_centrality_list(edge_centrality):
    """Sorts list of pairs by the second element in pairs """
    return sorted([(i, c) for i, c in edge_centrality],
                  key=lambda x: x[1], reverse=True)


"""def get_edges_statistics_in_paths(G):
    Returns dict {
        (source, target): ({edge_ij: g_{st}^{ij}}, g_{st} - number_of_paths from source to target)
    }
    
    edges_to_paths_info = dict()

    # iterate over all pairs of nodes
    for source, target in tqdm(list(combinations(G.nodes(), 2))):
        try:
            # find all shortest paths from source to target (use nx.all_shortest_paths)
            shortest_paths = list(nx.all_shortest_paths(G, source, target))

            # number of paths
            number_of_paths = len(shortest_paths)

            # get all edges in the paths
            edges = []
            for path in shortest_paths:
                edges.extend([tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)])

            # calculate number of edge occurences
            edge_to_cnt = dict(Counter(edges))
            edges_to_paths_info[(source, target)] = (edge_to_cnt, number_of_paths)
        except nx.NetworkXNoPath:
            pass
    return edges_to_paths_info"""


"""def calc_simple_edge_betweenness_centrality(G, normalized=True):
    Calculates edge_betweenness_centrality in a simple way
    nodes_to_paths_info = get_edges_statistics_in_paths(G)

    # calculate sigma_ij
    edge_to_centrality = dict()

    for edge_to_cnt, number_of_paths in nodes_to_paths_info.values():
        for edge, edge_cnt in edge_to_cnt.items():
            if edge in edge_to_centrality:
                edge_to_centrality[edge] += edge_cnt / number_of_paths
            else:
                edge_to_centrality[edge] = edge_cnt / number_of_paths

    # normalization (as it is done in networkx)
    if normalized:
        normalized_factor = 2 / (len(G.nodes()) - 1) / (len(G.nodes()) - 2)
        for edge in edge_to_centrality:
            edge_to_centrality[edge] *= normalized_factor

    return edge_to_centrality"""


"""def calc_edge_betweenness_centrality(G, normalized=True, method='networkx'):
    Calculates edge_betweenness_centrality via given method
    if method == 'networkx':
        edge_to_centrality = nx.edge_betweenness_centrality(G, normalized=normalized)
    elif method == 'simple':
        edge_to_centrality = calc_simple_edge_betweenness_centrality(G, normalized)
    else:
        raise NotImplementedError(f'Method {method} is not implemented')

    # return transformed dict to sorted list of pairs
    edge_centrality_list = list(edge_to_centrality.items())
    return sort_edge_centrality_list(edge_centrality_list)
"""

#2. Watts-Strogatz small-world graphs

# distances

def get_distances_between_nodes_with_sampling(G, n_max=100):
    """Return list of list of distances between nodes (maximum n_max nodes)"""
    # select <= n_max nodes
    nodes = G.nodes() if len(G) <= n_max else random.sample(G.nodes(), n_max)
    # finds all distances between nodes in the list as the source, using nx.shortest_path_length
    D = []
    for i in nodes:
        D.extend([d for d in nx.shortest_path_length(G, source = i).values() if d > 0])
    return D

# avg clustering (use code from previous labs)

def get_link_triads(G, i, j):
    """Return link triads containing nodes i and j"""
    t = len(set(G[i]).intersection(G[j]))
    return t


def get_node_triads(G, i):
    """Return triads containing node i"""
    t = 0
    for j in G[i]:
        if G.degree[i] <= G.degree[j]:
            t += get_link_triads(G, i, j) / 2
        else:
            t += get_link_triads(G, j, i) / 2
    return t


def get_node_clustering_coef(G, i):
    """Return node clustering coef for node i"""
    k_i = G.degree[i]
    if k_i <= 1:
        return 0
    return get_node_triads(G, i) * 2 / (k_i*k_i-k_i)


def get_avg_clustering_coef(G):
    """Return average clustering coef for graph G (maximum n_max nodes)"""
    C = 0
    for i in G.nodes():
        C += get_node_clustering_coef(G, i)
    return C / len(G.nodes())


def get_avg_clustering_coef_with_sampling(G, n_max=100):
    """Return average clustering coef for graph G (maximum n_max nodes)"""
    # select <= n_max nodes
    nodes = G.nodes() if len(G) <= n_max else random.sample(G.nodes(), n_max)
    C = 0
    for i in nodes:
        C += get_node_clustering_coef(G, i)
    return C / len(nodes)


def print_statistics(G):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    5) Average distance
    6) Average clustering coef
    """

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print('#Nodes:', n)
    print('#Edges:', m)

    print("Average node degree: {:.4f}".format(2 * m / n))
    print("Density: {:.4f}".format(2 * m / n / (n - 1)))

    D = get_distances_between_nodes_with_sampling(G)

    print("Average distance: {:.4f}".format(sum(D) / len(D)))

    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    print("Average clustering coef: {:.4f}".format(get_avg_clustering_coef_with_sampling(G)))


def watts_strogatz(n, k, p):
    """Returns small-world model"""
    G = nx.MultiGraph(name="watts_strogatz")
    # add nodes
    for i in G.nodes():
        G.add_node(i)

    # add edges using algorithm given in lectures
    edges = []
    for i in range(n):
        for j in range(i + 1, i + k // 2 + 1):
            edges.append((i, random.randint(0, n - 1) if random.random() < p else j % n))

    # get final graph
    G.add_edges_from(edges)
    return G

import math

def get_k(G):
    """Returns the closest even number to <k>"""
    k = 2* G.number_of_edges() / G.number_of_nodes()
    return math.ceil(k/2)*2


def select_p(G, k):
    """Returns p according to the formula"""
    avg_cluster_coef = get_avg_clustering_coef_with_sampling(G)
    return 1 - (avg_cluster_coef*4*(k-1)/3/(k-2))**(1/3)


"""small_world_Gs = []
for G in [karate_G, highways_G, euroroads_G, darknet_G, imdb_G, google_G]:
    print(f'Original {G.name}')
    print_statistics(G)
    print('------------------')

    n = G.number_of_nodes()
    k = get_k(G)
    p = select_p(G, k)

    start_time = time.time()
    small_world_G = watts_strogatz(n, k, p)
    print("Building Small-World Graph Time: {:.1f}s".format(time.time() - start_time))
    print_statistics(small_world_G)

    small_world_Gs.append(small_world_G)
    print()
    print('==================')
    print()
"""

#vi) ################################################################################################################################################################

def plot_degree_distribution(G):
    """Plots degree distribution in loglog scale"""
    # get dict of degree -> number of nodes that has this degree
    degree_to_cnt = dict(Counter(dict(G.degree()).values()))

    degrees = sorted(degree_to_cnt.keys())
    plt.loglog(degrees, [degree_to_cnt[k] / len(G) for k in degrees], '*k')
    plt.title(G.name)
    plt.ylabel('$log(p_k)$')
    plt.xlabel('$log(k)$')
    plt.show()

def read_net(folder, file_name):
    """Read network"""
    G = nx.MultiGraph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node = line.split('"')
                G.add_node(int(node[0]) - 1, label = node[1].strip())
        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G

# distances

def get_distances_between_nodes_with_sampling(G, n_max=100):
    """Return list of list of distances between nodes (maximum n_max nodes)"""
    # select <= n_max nodes
    nodes = G.nodes() if len(G) <= n_max else random.sample(G.nodes(), n_max)
    # finds all distances between nodes in the list as the source, using nx.shortest_path_length
    D = []
    for i in nodes:
        D.extend([d for d in nx.shortest_path_length(G, source = i).values() if d > 0])
    return D

# avg clustering (use code from previous labs)

def get_link_triads(G, i, j):
    """Return link triads containing nodes i and j"""
    t = len(set(G[i]).intersection(G[j]))
    return t


def get_node_triads(G, i):
    """Return triads containing node i"""
    t = 0
    for j in G[i]:
        if G.degree[i] <= G.degree[j]:
            t += get_link_triads(G, i, j) / 2
        else:
            t += get_link_triads(G, j, i) / 2
    return t


def get_node_clustering_coef(G, i):
    """Return node clustering coef for node i"""
    k_i = G.degree[i]
    if k_i <= 1:
        return 0
    return get_node_triads(G, i) * 2 / (k_i*k_i-k_i)


def get_avg_clustering_coef(G):
    """Return average clustering coef for graph G (maximum n_max nodes)"""
    C = 0
    for i in G.nodes():
        C += get_node_clustering_coef(G, i)
    return C / len(G.nodes())


def get_avg_clustering_coef_with_sampling(G, n_max=100):
    """Return average clustering coef for graph G (maximum n_max nodes)"""
    # select <= n_max nodes
    nodes = G.nodes() if len(G) <= n_max else random.sample(G.nodes(), n_max)
    C = 0
    for i in nodes:
        C += get_node_clustering_coef(G, i)
    return C / len(nodes)

#1. Barabási-Albert and Price scale-free graphs

def calc_power_law_gamma(G, k_min=1):
    """Calculates gamma according to the formula"""
    n = 0
    sumk = 0
    for _, k in G.degree():
        if k >= k_min:
            sumk += math.log(k)
            n += 1
    return 1 + 1 / (sumk / n - math.log(k_min - 0.5)) if n > 0 else math.nan


def print_statistics(G, k_min, fast=False):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    5) Gamma
    6) LCC
    7) Average distance
    6) Average clustering coef
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # print #nodes
    print('#Nodes:', n)
    # print #edges
    print('#Edges:', m)
    # print average degree
    print("Average Degree: {:.4f}".format(2 * m / n))
    # print Density
    print("Density {:.4f}".format(2 * m / n / (n - 1)))
    # print gamma
    print("Gamma = {:.2f}, k_min = {:d}".format(calc_power_law_gamma(G, k_min), k_min))

    # if the graph is not too huge
    if not fast:
        if isinstance(G, nx.DiGraph):
            G = nx.MultiGraph(G)

        # print large connected component size (use nx.connected_components)
        C = list(nx.connected_components(G))
        print("LCC: {:.1f}% ({:,d})".format(100 * max(len(c) for c in C) / n, len(C)))

        # print average distance
        D = get_distances_between_nodes_with_sampling(G)
        print("Average Distance: {:.4f}".format(sum(D) / len(D)))

        # print average clustering coef
        if isinstance(G, nx.MultiGraph):
            G = nx.Graph(G)
        print("Average Clustering Coef: {:.4f}".format(get_avg_clustering_coef_with_sampling(G)))
    print()


def price(n, c, a):
    """Generates Price scale-free network"""
    G = nx.MultiDiGraph(name="price")
    edges = []
    G.add_node(0)
    for i in range(1, c + 1):
        G.add_node(i)
        edges.append((0, i))
    for i in range(len(G), n):
        G.add_node(i)
        for _ in range(c):
            if random.random() < c / (c + a):
                edges.append((i, random.choice(edges)[1]))
            else:
                edges.append((i, random.randint(0, i)))
    G.add_edges_from(edges)
    return G


def barabasi_albert(n, c):
    """Generates Barabási-Albert scale-free network"""

    # Think about how to use price function
    return nx.MultiDiGraph(price(n, c, c), name="barabasi_albert")

k_min = 5  #fix k_min for all graphs

for graph_name in ["enron"]:
    G = read_net('./', graph_name + '.net')
    if_print_fast = len(G.nodes()) > 100000

    # print info about graph
    print('Graph:', graph_name)
    print_statistics(G, k_min, if_print_fast)
    plot_degree_distribution(G)

    # get params for scale-free networks
    n = G.number_of_nodes()
    m = G.number_of_edges()
    c = round(m / n)

    gamma = calc_power_law_gamma(G, k_min = k_min)
    a = c if gamma <= 2 else c * (gamma - 2)  # calculate a

    # generates Price scale-free network and prints information
    start_time = time.time()
    G = price(n, c, a)
    end_time = time.time() - start_time
    print(f'Graph: {G.name}_{graph_name}')
    print("Time to build: {:.1f}s".format(end_time))
    print_statistics(G, k_min, if_print_fast)
    plot_degree_distribution(G)

    # generates Barabási-Albert scale-free network and prints information
    start_time = time.time()
    G = barabasi_albert(n, c)
    end_time = time.time() - start_time
    print(f'Graph: {G.name}_{graph_name}')
    print("Time to build: {:.1f}s".format(end_time))
    print_statistics(G, k_min, if_print_fast)
    plot_degree_distribution(G)

    print('====================================================================================')

n = 100000  # set number of nodes
c = 10  # set average degree
k_min = 25  # set k_min

# generate graphs using different gammas using price function
for gamma in [2.01, 2.25, 2.5, 2.75, 3., 4., 5.]:
  G=price(n,c,c*(gamma-2))
  print(f'Graph: {G.name}_{gamma}')
  print_statistics(G, k_min, fast=False)
  plot_degree_distribution(G)

#vii) ################################################################################################################################################################

def plot_orbits(G, orbits):
    # init figure and set it suptitle
    fig = plt.figure()
    fig.suptitle(G.name)

    # draw 15 orbits in a cycle
    for o in range(15):
        nk = {}
        for i in range(len(G)):
            k = orbits[i][o]
            if k not in nk:
                nk[k] = 0
            nk[k] += 1
        ks = sorted(nk.keys())

        # plot in loglog scale
        plt.subplot(3, 5, o + 1)
        plt.loglog(ks, [nk[k] / len(G) for k in ks], 'ok', markersize=1)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def read_net(folder, graph_name):
    """Read network"""
    file_name = graph_name + '.net'
    G = nx.MultiGraph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                # starts *arcs => digraph
                if line.startswith("*arcs"):
                    G = nx.MultiDiGraph(G)
                break
            else:
                node_info = line.split("\"")
                node = int(node_info[0]) - 1
                label = node_info[1]
                weight = float(node_info[2]) if len(node_info) > 2 and len(node_info[2].strip()) > 0 else None
                G.add_node(node, label=label, weight=weight)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G


# distances and power law

def get_distances_between_nodes_with_sampling(G, n_max=100):
    """Return list of list of distances between nodes (maximum n_max nodes)"""
    # select <= n_max nodes
    nodes = G.nodes() if len(G) <= n_max else random.sample(G.nodes(), n_max)
    # finds all distances between nodes in the list as the source, using nx.shortest_path_length
    D = []
    for i in nodes:
        D.extend([d for d in nx.shortest_path_length(G, source = i).values() if d > 0])
    return D


def calc_power_law_gamma(G, k_min = 1):
    """Calculates gamma according to the formula"""
    # Use code from the previous lab
    n = 0
    sumk = 0
    for _, k in G.degree():
        if k >= k_min:
            sumk += math.log(k)
            n += 1
    return 1 + 1 / (sumk / n - math.log(k_min - 0.5)) if n > 0 else math.nan

def get_degree_mixing(G):
    """Calculates degree mixing of graph"""
    x, y = [], []  # init values of k and k'
    # iterate over all edges
    for i, j in G.edges():
        # add degrees of i's and j's nodes to x and y: x<-[deg_i,deg_j], y<-[deg_j,deg_i]
       x.append(G.degree(i))
       y.append(G.degree(j))
       x.append(G.degree(j))
       y.append(G.degree(i))
    return stats.pearsonr(x, y)[0]

#statistics

def print_statistics(G, k_min, fast=False):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    5) Gamma
    6) Degree mixing
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # print #nodes
    print('#Nodes:', n)
    # print #edges
    print('#Edges:', m)
    # print average degree
    print("Average Degree: {:.4f}".format(2 * m / n))
    # print Density
    print("Density {:.4f}".format( 2 * m / n / (n - 1)))

    # if the graph is not too huge
    if not fast:
        # print gamma
        print("Gamma = {:.2f}, k_min = {:d}".format(calc_power_law_gamma(G, k_min), k_min))
        # print degree mixing
        print("Degree mixing = {:.4f}".format(get_degree_mixing(G)))
    print()

k_min = 10  # set k_min for power law

Gs = []
for name in ["karate_club", "dolphins", "java", "darknet", "collaboration_imdb", "gnutella", "facebook", "nec"]:
    G = read_net('./', name)
    Gs.append(G)
    print('Name:', G.name)
    print_statistics(G, k_min, fast = False)

def generate_random_graph(G):
    """"Generate random graph using nx.gnm_random_graph"""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return nx.gnm_random_graph(n,m)

k_min = 10  # set k_min for power law

for G in Gs:
    print('Name of random graph:', G.name)
    G_random = generate_random_graph(G)
    print_statistics(G_random, k_min, fast = False)


def get_degree_mixing_directed(G, source_direction, target_direction):
    """Calculates degree mixing of graph according to directions

    Parameters
    ------
    source_direction: str "in" or "out"
    target_direction: str "in" or "out"
    """
    x, y = [], []  # init values of k and k'
    for i, j in G.edges():
        # add in/out degree of i based on source_direction
        x.append(G.out_degree(i) if source_direction == 'out' else G.in_degree(i))
        # add in/out degree of j based on target_direction
        y.append(G.in_degree(j) if target_direction == 'in' else G.out_degree(j))
    return stats.pearsonr(x, y)[0]

k_min = 10  # set k_min for power law

for G in Gs:
    rii = math.nan
    if isinstance(G, nx.DiGraph):
        rii = get_degree_mixing_directed(G, 'in', 'in')

    rio = math.nan
    if isinstance(G, nx.DiGraph):
        rio = get_degree_mixing_directed(G, 'in', 'out')

    roi = math.nan
    if isinstance(G, nx.DiGraph):
        roi = get_degree_mixing_directed(G, 'out', 'in')

    roo = math.nan
    if isinstance(G, nx.DiGraph):
        roo = get_degree_mixing_directed(G, 'out', 'out')
    print('Name:', G.name)
    print('r_ii: {:^7.3f}'.format(rii))
    print('r_io: {:^7.3f}'.format(rio))
    print('r_oi: {:^7.3f}'.format(roi))
    print('r_oo: {:^7.3f}'.format(roo))
    print()

#2. Structurally disassortative networks by degree

def to_hash(i, j):
    """Simple linearization of pair: (i,j)->id"""
    if i <= j:
        i, j = j, i
    return i * (i - 1) // 2 + j


def build_set_of_hashed_edges(G):
    """Build set of hashed edges and corresponding list of edges that were selected"""
    # init empty set of hashed edges
    H = set()
    # init empty list of edges
    edges = []
    for edge in G.edges():
        i, j = edge
        # get hash of pair
        h = to_hash(i,j)
        # if h is not in hashed edges add [i,j] to the list of edges and add h to hashed edges
        if h not in H:
          edges.append([i,j])
          H.add(h)
    return H, edges


def rewire_graph(G, num = 100):
    H, edges = build_set_of_hashed_edges(G)
    for _ in range(num):
        # select two random edges ids
        eij = random.randint(0, len(edges)-1)
        euv = random.randint(0, len(edges)-1)
        if eij != euv:
            i, j = edges[eij]
            u, v = edges[euv]
            # if any event with probability 0.5 is True (e.g. random number < 0.5)
            if random.random() < 0.5:
                if i != v and u != j:
                    # get hash of (i, v)
                    hiv = to_hash(i,v)
                    # get hash of (u, j)
                    huj = to_hash(u,j)
                    if hiv not in H and huj not in H:
                        # transform edges
                        edges[eij][1] = v
                        edges[euv][1] = j
                        # remove hashes of old edges from H
                        H.remove(to_hash(i,j))
                        H.remove(to_hash(u,v))
                        # add hashes of new edges
                        H.add(hiv)
                        H.add(huj)
            else:
                if i != u and j != v:
                    # get hash of (i, u)
                    hiu = to_hash(i,u)
                    # get hash of (j, v)
                    hjv = to_hash(j,v)
                    if hiu not in H and hjv not in H:
                        edges[eij][1] = u
                        edges[euv][0] = j
                        # remove hashes of old edges from H
                        H.remove(to_hash(i,j))
                        H.remove(to_hash(u,v))
                        # add hashes of new edges
                        H.add(hiu)
                        H.add(hjv)
    G = nx.empty_graph(len(G))
    G.add_edges_from(edges)
    return G

k_min = 10  # set k_min for power law

for G in Gs:
    m = G.number_of_edges()
    G_rewired = rewire_graph(G, 10 * m)
    print('Name:', G.name)
    print('Degree mixing: {:^7.3f}'.format(get_degree_mixing(G_rewired)))
    print()

# 3. Node mixing by not degree

def get_assortativity_coefficient(G, attribute_name):
    """Calculates assortativity coefficient using nx.numeric_assortativity_coefficient"""
    return nx.numeric_assortativity_coefficient(G, attribute_name)

for name in ["highways"]:
  G = read_net('./', name)
  rw = get_assortativity_coefficient(G, "weight")
  print("{:>21s} | {:^7.3f}".format("'" + G.name + "'", rw))
print()

# 4. Graphlet degree distributions

def save_for_orca(G):
    """Save graph in the format <i j> on each line where (i,j) is edge"""
    with open(G.name + ".in", 'w') as file:
        file.write(str(G.number_of_nodes()) + " " + str(G.number_of_edges()) + "\n")
        # save edges
        for edge in G.edges():
          file.write(str(edge[0]) + " " + str(edge[1]) + '\n')


def orca(G):
    """Call program with algorithm"""
    save_for_orca(G)
    os.system("./orca_program node 4 " + G.name + ".in " + G.name + ".orca")
    os.remove(G.name + ".in")


def get_orbits(G):
    """Get orbits via ocra"""
    orca(G)  # saves files
    orbits = []
    # read orbits from the file
    with open(G.name + ".orca", 'r') as file:
      for line in file:
        orbits.append([int(k) for k in line.split()])

    return orbits

k_min = 10

for name in ["java" , "darknet", "collaboration_imdb", "gnutella", "facebook", "nec"]:
    G = nx.Graph(read_net('./', name))
    print_statistics(G, k_min)
    orbits = get_orbits(G)
    plot_orbits(G, orbits)

#viii) ################################################################################################################################################################

def plot_clusters(G, clusters_G, node_poses, title=''):
    """Plot clusters"""
    #viz.plot_network_clusters(G, clusters_G, node_poses)
    plt.title(title)
    plt.show()


def plot_ideal_and_algo_clusters(G, ideal_partition, algo_fn):
    """Plot ideal clusters and built clusters via algo_fn"""
    node_poses = nx.spring_layout(G)
    plot_clusters(G, ideal_partition, node_poses, 'Ideal')
    plot_clusters(G, algo_fn(G), node_poses, 'Custom algo')

def read_net(folder, graph_name):
    """Read network"""
    file_name = graph_name + '.net'
    G = nx.MultiGraph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node_info = line.split("\"")
                node = int(node_info[0]) - 1
                label = node_info[1]
                cluster = int(node_info[2]) if len(node_info) > 2 and len(node_info[2].strip()) > 0 else None
                G.add_node(node, label=label, cluster=cluster)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G

#statistics

def print_statistics(G):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # print #nodes
    print('#Nodes:', n)
    # print #edges
    print('#Edges:', m)
    # print average degree
    print("Average Degree: {:.4f}".format(2 * m / n))
    # print Density
    print("Density {:.4f}".format( 2 * m / n / (n - 1)))

def is_labeling_complete(node_to_label, G):
    """Determine whether or not LPA is done.
    Label propagation is complete when all nodes have a label that is
    in the set of highest frequency labels amongst its neighbors.
    """
    return all(
        node_to_label[v] in neighbors_most_frequent_labels(v, node_to_label, G) for v in G if len(G[v]) > 0
    )


def color_network(G):
    """Color the network so that neighboring nodes all have distinct colors.
    Returns a dict keyed by color to a set of nodes with that color.
    """
    color_to_nodes = dict()  # color => set(node)
    node_to_color = nx.coloring.greedy_color(G)
    for node, color in node_to_color.items():
        if color in color_to_nodes:
            color_to_nodes[color].add(node)
        else:
            color_to_nodes[color] = {node}
    return color_to_nodes


def neighbors_most_frequent_labels(node, node_to_label, G):
    """Return a set of labels of node's neighbors with maximum frequency in node_to_label"""

    # Nodes with no neighbors are themselves a community and are labeled
    if not G[node]:
        return {node_to_label[node]}

    # Compute the frequencies of all neighbours of node
    freqs = Counter(node_to_label[adj_node] for adj_node in G[node])
    max_freq = max(freqs.values())
    return {label for label, freq in freqs.items() if freq == max_freq}


def update_node_label(node, node_to_label, G):
    """Update the label of a node choosing label with maximum frequency among its neighbors"""
    high_labels = neighbors_most_frequent_labels(node, node_to_label, G)
    if len(high_labels) == 1:
        node_to_label[node] = high_labels.pop()
    elif len(high_labels) > 1:
        # Prec-Max
        if node_to_label[node] not in high_labels:
            node_to_label[node] = max(high_labels) # for improving convergence


def my_partition_algo(G):
    """Implementation of your partition algorithm implementation. Returns graph partition in cdlib format"""

    node_to_label = {node: label for label, node in enumerate(G)}
    coloring = color_network(G)
    max_iters = 30
    iter_num = 0
    while not is_labeling_complete(node_to_label, G) and iter_num < max_iters:
        # Update the labels of every node with the same color.
        for color, nodes in coloring.items():
            for n in nodes:
                update_node_label(n, node_to_label, G)
        iter_num += 1

    label_to_nodes = {}
    for node, label in node_to_label.items():
        if label in label_to_nodes:
            label_to_nodes[label].append(node)
        else:
            label_to_nodes[label] = [node]
    node_groups = list(label_to_nodes.values())
    #return NodeClustering(node_groups, G, 'My algorithm')

# 1. Small networks with known partitioning

def get_graph_ideal_partition(G):
    """"Transforms graph idea partition to cdlib format"""
    # Transform nodes -> [list of nodes in cluster 1, list of nodes in cluster 2, ....]
    # Use cluster attribute of each node
    P = {}
    for node in G.nodes(data=True):
      if node[1]['cluster'] not in P:
        P[node[1]['cluster']]=[]
      P[node[1]['cluster']].append(node[0])
    node_clusters = P.values()
    #return NodeClustering(list(node_clusters), G, 'Ideal')


def compare(G, ideal_partition, algo_fn, iter_num=10):
    """Compare partition built by algo_fn and ideal_partition"""
    # use metrics from cdlib
    print("{:5s}  {:^5s}  {:^5s}  {:^5s}".format('Count', 'NMI', 'ARI', 'VI'))
    clusters_num, NMI, ARI, VI = 0, 0, 0, 0  # initialize metrics
    for _ in range(iter_num):
        clustered_G = algo_fn(G)  # call the algorithm
        clusters_num += len(clustered_G.communities) / iter_num  # number of communities
        NMI += clustered_G.normalized_mutual_information(ideal_partition).score / iter_num  # normalized mutual information
        ARI += clustered_G.adjusted_rand_index(ideal_partition).score / iter_num  # adjusted_rand_index
        VI += clustered_G.variation_of_information(ideal_partition).score / iter_num  # variation_of_information
    # TODO if necessary
    print("{:>5.0f}  {:5.3f}  {:5.3f}  {:5.3f}".format(clusters_num, NMI, ARI, VI))

partition_algo_fn = my_partition_algo  # name of the function with your algorithm
iter_num = 10  # number times to run algorithm

for name in ["karate_club", "southern_women", "dolphins", "american_football"]:
    G = read_net('', name)
    print('Graph:', name)
    print()
    print_statistics(G)
    print('----------------------')
    ideal_partition = get_graph_ideal_partition(G)
    compare(G, ideal_partition, partition_algo_fn, iter_num=iter_num)
    print('----------------------')
    plot_ideal_and_algo_clusters(G, ideal_partition, partition_algo_fn)
    print()
    print('======================'*3)

# 2. Larger networks with node metadata

partition_algo_fn = my_partition_algo  # name of the function with your algorithm
iter_num = 10  # number times to run algorithm

for name in ["cdn_java", "dormitory", "wikileaks", "youtube"]:
    G = read_net('', name)
    print('Graph:', name)
    print()
    print_statistics(G)
    print('----------------------')
    ideal_partition = get_graph_ideal_partition(G)
    compare(G, ideal_partition, partition_algo_fn, iter_num=iter_num)
    print()
    print('======================'*3)

# 3. Synthetic graphs with planted partition

def build_synhtetic_graph(mu = 0.5):
    """
    Implementation of chosen community detection algorithm
    Returns graph where each node has attribute "cluster" - number of community
    """
    G = nx.Graph(name = "girvan_newman")
    n = 128
    cluster_div = 32
    for i in range(n):
        G.add_node(i, cluster = i // cluster_div + 1)
    for i in range(n):
        for j in range(i + 1, n):
            if G.nodes[i]['cluster'] == G.nodes[j]['cluster']:
                # want the probability of adding edges to be not less than (1-mu)/2
                if random.random() < cluster_div /2 * (1-mu) / (cluster_div-1):
                  G.add_edge(i,j)
            else:
                # want the probability of adding edges in between nodes
                # in defferent clusters to be mu/6
                if random.random() < mu /6:
                  G.add_edge(i,j)
    return G


def compare_on_synthetic(algo_fn, mus, iter_num=10):
    for mu in mus:
        NMI = 0  # init NMI
        for _ in range(iter_num):
            G = build_synhtetic_graph(mu)  # build synthetic
            clustered_G = algo_fn(G)  # apply algo
            partition = get_graph_ideal_partition(G)  # get ideal partition
            NMI += clustered_G.normalized_mutual_information(partition).score / iter_num  # calc metric
        print("mu={:.2f}, NMI: {:5.3f}".format(mu, NMI))

partition_algo_fn = my_partition_algo  # name of the function with your algorithm
iter_num = 10  # number times to run algorithm
mus = [0.05 * i for i in range(1, 13)]  # possible values of mu

compare_on_synthetic(partition_algo_fn, mus, iter_num)

#ix) ################################################################################################################################################################

def plot_clusters(G, clusters_G, node_poses, title=''):
    """Plot clusters"""
    #viz.plot_network_clusters(G, clusters_G, node_poses)
    plt.title(title)
    plt.show()


def plot_ideal_and_algo_clusters(G, ideal_partition, algo_fn):
    """Plot ideal clusters and built clusters via algo_fn"""
    node_poses = nx.spring_layout(G)
    plot_clusters(G, ideal_partition, node_poses, 'Ideal')
    plot_clusters(G, algo_fn(G), node_poses, 'Custom algo')

def read_net(folder, graph_name):
    """Read network"""
    file_name = graph_name + '.net'
    G = nx.MultiGraph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node_info = line.split("\"")
                node = int(node_info[0]) - 1
                label = node_info[1]
                cluster = int(node_info[2]) if len(node_info) > 2 and len(node_info[2].strip()) > 0 else 0
                G.add_node(node, label=label, cluster=cluster)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G

#statistics

def print_statistics(G):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # print #nodes
    print('#Nodes:', n)
    # print #edges
    print('#Edges:', m)
    # print average degree
    print("Average Degree: {:.4f}".format(2 * m / n))
    # print Density
    print("Density {:.4f}".format( 2 * m / n / (n - 1)))

# 1. Stochastic block models vs community detection

"""algs = {
    "Louvain": algorithms.louvain,                # Louvain algorithm 
    "Infomap": algorithms.infomap,                # Infomap algorithm 
    "EM(MM)": lambda G: algorithms.em(G, k=2),                 # Mixture model based on expectation maximization
    "(DC)SBM": algorithms.sbm_dl,                # degree-corrected stochastic block model 
}"""

def get_graph_ideal_partition(G):
    """"Transforms graph idea partition to cdlib format"""
    # Transform nodes -> [list of nodes in cluster 1, list of nodes in cluster 2, ....]
    # Use cluster attribute of each node
    P = {}
    for node in G.nodes(data = True):
        if node[1]['cluster'] not in P:
            P[node[1]['cluster']] = []
        P[node[1]['cluster']].append(node[0])
    node_clusters = P.values()
    #return NodeClustering(list(node_clusters), G, 'Ideal')


def compare(G, ideal_partition, algo_fn, iter_num=10):
    """Compare partition built by algo_fn and ideal_partition"""
    # use metrics from cdlib
    print("{:5s}  {:^5s}  {:^5s}  {:^5s}".format('Count', 'NMI', 'ARI', 'VI'))
    clusters_num, NMI, ARI, VI = 0, 0, 0, 0  # initialize metrics
    for _ in range(iter_num):
        clustered_G = algo_fn(G)  # call the algorithm
        clusters_num += len(clustered_G.communities) / iter_num  # number of communities
        NMI += clustered_G.normalized_mutual_information(ideal_partition).score / iter_num  # normalized mutual information
        ARI += clustered_G.adjusted_rand_index(ideal_partition).score / iter_num  # adjusted_rand_index
        VI += clustered_G.variation_of_information(ideal_partition).score / iter_num  # variation_of_information
    print("{:>5.0f}  {:5.3f}  {:5.3f}  {:5.3f}".format(clusters_num, NMI, ARI, VI))

iter_num = 10  # number times to run algorithm

"""for name in ["karate_club", "southern_women"]:
    G = read_net('', name)
    print('Graph:', name)
    print()
    print_statistics(G)
    print('----------------------')
    ideal_partition = get_graph_ideal_partition(G)
    for algo_name in algs:
        print('ALGORITHM:', algo_name)
        compare(G, ideal_partition, algs[algo_name], iter_num=iter_num)
        print('----------------------')
        plot_ideal_and_algo_clusters(G, ideal_partition, algs[algo_name])
        print()
        print('======================'*3)

    print()
    print('======================'*3) 
    print('======================'*3)  """

def build_synhtetic_graph(mu = 0.5):
    """
    Implementation of chosen community detection algorithm
    Returns graph where each node has attribute "cluster" - number of community
    """
    G = nx.Graph(name = "girvan_newman")
    n = 128
    cluster_div = 32
    for i in range(n):
        G.add_node(i, cluster = i // cluster_div + 1)
    for i in range(n):
        for j in range(i + 1, n):
            if G.nodes[i]['cluster'] == G.nodes[j]['cluster']:
                # want the probability of adding edges to be not less than (1-mu)/2
                if random.random() < cluster_div / 2 * (1 - mu) / (cluster_div-1):
                    G.add_edge(i, j)
            else:
                # want the probability of adding edges in between nodes
                # in defferent clusters to be mu/6
                if random.random() < mu / 6:
                    G.add_edge(i, j)
    return G


def compare_on_synthetic(algo_fn, mus, iter_num=10):
    for mu in mus:
        NMI = 0  # init NMI
        for _ in range(iter_num):
            G = build_synhtetic_graph(mu)  # build synthetic
            clustered_G = algo_fn(G)  # apply algo
            partition = get_graph_ideal_partition(G)  # get ideal partition
            NMI += clustered_G.normalized_mutual_information(partition).score / iter_num  # calc metric
        print("mu={:.2f}, NMI: {:5.3f}".format(mu, NMI))

iter_num = 10  # number times to run algorithm
mus = [0.05 * i for i in range(1, 13)]  # possible values of mu

"""for algo_name in algs:
    print('ALGORITHM:', algo_name)
    compare_on_synthetic(algs[algo_name], mus, iter_num)
    print('======================'*3)  """

# 2. Network  k -cores decomposition

def remove_nodes_by_degree_limit(G, k):
    """Removes nodes which has degree less than k"""
    changed = True
    while changed:
      changed=False
      for i in list(G.nodes()):
        if G.degree(i) < k:
          G.remove_node(i)
          changed = True


def get_k_cores_count(G, k):
    """Gets number of k-cores (= number of connected components)"""
    return len(list(nx.connected_components(G)))

for name in ["cdn_jung", "cdn_java", "wikileaks", "collaboration_imdb"]:
    G = read_net('', name)
    print('Graph:', name)
    print()
    print_statistics(G)
    print('----------------------')
    G_copy = nx.MultiGraph(G)  #copy G

    # run iterative k_max search. Print number of k-cores for all intermediate k
    k = 1
    while True:
        remove_nodes_by_degree_limit(G_copy, k)
        k_cores_cnt = get_k_cores_count(G_copy, k)
        if k_cores_cnt == 0:
          k -= 1
          break
        print("k = {:d} | k-cores number = {:d}".format(k, k_cores_cnt))
        k += 1
    print()
    # Here k = k_max

    # Find k_max-core and print its node labels
    remove_nodes_by_degree_limit(G, k)
    node_labels = sorted(node["label"] for _, node in G.nodes(data=True))
    print("{:>12s} | {:s}\n".format(str(k) + '-cores', "; ".join(node_labels)))
    print()
    print('======================'*3)



#xi) ################################################################################################################################################################

def read_pajek_net(folder, graph_name):
    """Read network in pajek format"""
    file_name = graph_name + '.net'
    G = nx.MultiGraph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node = int(line.split("\"")[0]) - 1
                G.add_node(node)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G


def read_edge_list_net(folder, graph_name):
    """Read network in the format of edge list"""
    file_name = graph_name + '.adj'
    G = nx.read_edgelist(os.path.join(folder, file_name), nodetype = int)
    G = nx.convert_node_labels_to_integers(G)  # relabel nodes into [0..#nodes-1]
    G.name = graph_name
    return G

#statistics

def print_statistics(G):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    print('#Nodes:', n)
    print('#Edges:', m)
    print("Average Degree: {:.4f}".format(2 * m / n))
    print("Density {:.4f}".format( 2 * m / n / (n - 1)))

# 1. Estimation by random-walk sampling

def get_subgraph_by_lcc(G):
    """Get the subgraph induced by the largest connected component"""
    connected_components = nx.connected_components(G)
    max_cc = max(connected_components, key=len)
    return G.subgraph(max_cc)


def get_neighbors_mapping(G):
    """
    Build dict node -> its neighbors (each neighbor counts the number of times
    the number of parallel connections it has with the node )
    """
    node_to_nn = {node: [] for node in G}
    for i in G.nodes():
        for j in G[i]:
            node_to_nn[i].extend([j] * len(G[i][j]))

    return node_to_nn


def sample_nodes(G, node_to_neighbors, size_ratio=0.15):
    """Sample nodes. Final number >= #nodes * size_ratio"""
    desired_len = G.number_of_nodes() * size_ratio
    # initialize empty list of sampled nodes
    res_nodes = []
    # Sample the first node and add it to the list
    cur_node = np.random.choice(G.number_of_nodes(), 1)[0]
    res_nodes.append(cur_node)

    while len(res_nodes) < desired_len:
        # sample the next node which is the neighbor of the previous one
        cur_node = np.random.choice(node_to_neighbors[cur_node], 1)[0]
        res_nodes.append(cur_node)
        # add it to the list
    return res_nodes

def estimate_k(G, size_ratio=0.15):
    """Estimate ⟨k⟩"""
    node_to_neighbors = get_neighbors_mapping(G)
    sampled_nodes = sample_nodes(G, node_to_neighbors, size_ratio)

    # calculate biased and corrected average
    s = len(sampled_nodes)
    estimated = 0
    corrected = 0
    for node in sampled_nodes:
      node_degree = len(node_to_neighbors[node])
      estimated += node_degree
      corrected += 1/ node_degree
    return estimated/s, s/corrected

for name in ["nec", "facebook", "enron", "www_google"]:
    G =  get_subgraph_by_lcc(read_pajek_net('./', name))
    print('Graph:', name)
    print_statistics(G)

    start_time = time.time()
    estimated, corrected = estimate_k(G)
    end_time = time.time()
    print("{:>12s} | {:.2f}".format('Estimated', estimated))
    print("{:>12s} | {:.2f}".format('Corrected', corrected))
    print("{:>12s} | {:.1f} s".format('Time', end_time - start_time))
    print()

# 2. Sampling Facebook social network

# Helpful functions for understanding structures of the networks

def fast_label_propagation(G):
    """Fast label propagation algorithm"""
    nodes = list(G.nodes())
    random.shuffle(nodes)

    Q = deque(nodes)

    # S[i] = True if community for i is found.
    # At the beginning every node has its own community
    S = [True] * len(G)

    # C[i] - representor of community that i belongs to
    C = [i for i in range(len(G))]

    while Q:
        i = Q.popleft()
        S[i] = False

        if len(G[i]) > 0:
            # Count frequences of communities among neighbors
            N = {}
            for j in G[i]:
                if C[j] in N:
                    N[C[j]] += 1  # len(G[i][j])
                else:
                    N[C[j]] = 1  # len(G[i][j])

            # Select the most popular community among neighbors
            maxn = max(N.values())
            c = random.choice([c for c in N if N[c] == maxn])

            if C[i] != c:
                # Push i into selected community and add its neighbors to Q, if necessary
                C[i] = c
                for j in G[i]:
                    if C[j] != c and not S[j]:
                        Q.append(j)
                        S[j] = True

    # Get final node groups representing communities
    L = {}
    for i, c in enumerate(C):
        if c in L:
            L[c].append(i)
        else:
            L[c] = [i]
    #return NodeClustering(list(L.values()), G, 'FLPA')


def print_info_about_clusters(G):
    """Prints:
    1) Size of the largest connected component
    2) Modularity of the fast label propagation clustering
    """
    # largest connected component
    C = list(nx.connected_components(G))
    print("LCC: {:.3f}% ({:,d})".format(100 * max(len(c) for c in C) / len(G), len(C)))

    # modularity of label propagation
    C = fast_label_propagation(G)
    print("{:>12s} | {:.3f} ({:,d})".format('Q', C.newman_girvan_modularity().score, len(C.communities)))

# Cleaning memory

import gc
del G
gc.collect()

for fb_num in [1, 2]:
    G = read_edge_list_net('./', f'facebook_{fb_num}_sampled')
    print('Graph:', G.name)
    print_statistics(G)
    print_info_about_clusters(G)
    print()

#xiii) ################################################################################################################################################################

def read_net(folder, graph_name):
    """Read network"""
    file_name = graph_name + '.net'
    G = nx.Graph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node = int(line.split("\"")[0]) - 1
                G.add_node(node)

        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1)
    return G


# statistics

def print_statistics(G):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    """
    print('Name:', G.name)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # print #nodes
    print('#Nodes:', n)
    # print #edges
    print('#Edges:', m)
    # print average degree
    print("Average Degree: {:.4f}".format(2 * m / n))
    # print Density
    print("Density: {:.4f}".format(2 * m / n / (n - 1)))

    # print number of connecnted components
    C = list(nx.connected_components(G))
    print("LCC: {:.1f}% ({:,d})".format(100 * max(len(c) for c in C) / n, len(C)))

# Read networks from files
Gs = []
for name in ["karate_club", "southern_women", "dolphins", "foodweb_baydry", "foodweb_baywet"]:
    G = read_net('./', name)
    print_statistics(G)
    print('===============')
    Gs.append(G)

# Create synthetic networks

N = 500  # number of nodes
M = 1500
K = M * 2 // N    # average node degree
graphs_cnt = 3  # number of graphs to create

# Erdős–Rényi graph
G = nx.gnm_random_graph(N, N * K // 2)
G.name = "erdos_renyi"
print_statistics(G)
for i in range(graphs_cnt):
    G = nx.gnm_random_graph(N, N * K // 2)
    G.name = "erdos_renyi_" + str(i + 1)
    Gs.append(G)
print('===============')

# Barabási–Albert graph
G = nx.barabasi_albert_graph(N, round(K / 2))
G.name = "barabasi_albert"
print_statistics(G)
for i in range(graphs_cnt):
    G = nx.barabasi_albert_graph(N, round(K / 2))
    G.name = "barabasi_albert_" + str(i + 1)
    Gs.append(G)
print('===============')

# Watts–Strogatz graph
G = nx.watts_strogatz_graph(N, K, 0.1)
G.name = "watts_strogatz"
print_statistics(G)
for i in range(graphs_cnt):
    G = nx.watts_strogatz_graph(N, K, 0.1)
    G.name = "watts_strogatz_" + str(i + 1)
    Gs.append(G)

# 1. Network comparison

def get_distances_matrix(Gs, graphs_dist_fn):
    """Gets matrix of paired distances between graphs"""
    dist = [[0]*len(Gs) for _ in range(len(Gs))]
    for i in range(len(Gs)):
      for j in range(i, len(Gs)):
        dist[i][j] = graphs_dist_fn(Gs[i],Gs[j])
        dist[j][i] = dist[i][j]
    return dist

def plot_distance_matrix(G_names, distance_matrix, title='', size=(15,15)):
    """Plots matrix of paired distances between graphs"""
    fig, ax = plt.subplots(figsize=size)

    #dist_matrix_df = pd.DataFrame(data=distance_matrix, columns=G_names, index = G_names)

    #sb.heatmap(dist_matrix_df, square=True, annot=True,cmap="YlGnBu",ax=ax)

    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b,t)
    ax.set_title(title)
    plt.show()

def save_for_orca(G):
    """Save graph in the format <i j> on each line where (i,j) is edge"""
    with open(G.name + ".in", 'w') as file:
        file.write(str(G.number_of_nodes()) + " " + str(G.number_of_edges()) + "\n")
        # save edges
        for edge in G.edges():
            file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

def orca(G, size):
    """Call program with algorithm"""
    save_for_orca(G)
    os.system(f"./orca_program node {size} " + G.name + ".in " + G.name + ".orca")
    os.remove(G.name + ".in")

def get_orbits(G, size):
    """Get orbits via ocra"""
    orca(G, size)  # saves files
    orbits = []
    # read orbits from the file
    with open(G.name + ".orca", 'r') as file:
        for line in file:
            orbits.append([int(k) for k in line.split()])
    os.remove(G.name + ".orca")
    return orbits


def orbit_distributions(orbits: list):
    """
    Gets orbits distribution
    for each node orbits is a list counts of graphlets containing the node
    """

    dists = []
    for o in range(len(orbits[0])):
        pk = {}
        # get ditribution of counts for the graphlet with index o
        for i in range(len(orbits)):
            k = orbits[i][o]
            if k not in pk:
                pk[k] = 0
            pk[k] += 1 / k if k > 0 else 0

        # for scaling to [0,1]
        p = sum(pk.values())

        # normalize and add to dists
        dists.append({k: pk[k] / p if p > 0 else pk[k] for k in pk})
    return dists


def graphlet_disagreement(G1, G2, size=4):
    """Calculates graphlet disagreement"""
    # Get orbit_distributions for G1 and G2
    P = orbit_distributions(get_orbits(G1, size=size))
    Q = orbit_distributions(get_orbits(G2, size=size))
    A = 0
    for o in range(len(P)):
        a = 0
        # sum differences over all k calculating formula (1)
        for k in set.union(set(P[o].keys()), Q[o].keys()):
            p = P[o][k] if k in P[o] else 0
            q = Q[o][k] if k in Q[o] else 0
            a += (p - q) ** 2
        # formula (2)
        A += 1 - (a / 2) ** 0.5
    return 1 - A / len(P)


from math import log, sqrt


def get_paths_lengths_pdf_for_node(G, i):
    """Calculates pdf of paths lengths from i to other nodes"""
    P = [0] * len(G)
    all_shortest_paths = nx.single_source_shortest_path_length(G, i)

    # for each length get its normalized number of occurrences
    for j in all_shortest_paths.values():
        P[j] += 1 / G.number_of_nodes()
    return P


def get_paths_lengths_pdfs_for_nodes(G):
    """For all nodes calculates pdfs of paths lengths to others"""
    # call get_paths_lengths_pdf_for_node for each node
    return [get_paths_lengths_pdf_for_node(G, i) for i in G.nodes()]


def mu(Ps):
    """For each node calculate its average length of paths to other nodes"""
    return [sum(P[j] for P in Ps) / len(Ps) for j in range(len(Ps[0]))]


def jensen_shannon(Ps, M):
    """Calculates Jensen–Shannon divergence"""
    res = 0
    for i in range(len(Ps)):
        for j in range(len(M)):
            if Ps[i][j] > 0:
                res += Ps[i][j] * log(Ps[i][j] / M[j])
    return res / len(Ps)


def node_dispersion(Ps, M):
    """Calculates node dispersion"""
    # Think how network diameter can be extracted from Ps or M
    return jensen_shannon(Ps, M) / log(M.index(0) + 1)


def node_distances(M1, M2):
    """Calculates distances between nodes"""
    # truncate the largest list to the size of shortest one
    if len(M1) < len(M2):
        M2 = M2[:len(M1)]
    elif len(M1) > len(M2):
        M1 = M1[:len(M2)]
    # calculate jensen_shannon divergence for [M1, M2] and mu([M1, M2])
    return jensen_shannon([M1, M2], mu([M1, M2]))


def simplified_dmeasure(Ps1, M1, Ps2, M2):
    """Calculates simplified D-measure"""
    return (sqrt(node_distances(M1, M2) / log(2)) + abs(
        sqrt(node_dispersion(Ps1, M1)) - sqrt(node_dispersion(Ps2, M2)))) / 2


def dmeasure(G1, G2):
    """Returns D-measure"""
    # get pdfs for G1
    Ps1 = get_paths_lengths_pdfs_for_nodes(G1)
    # get pdfs for G2
    Ps2 = get_paths_lengths_pdfs_for_nodes(G2)
    # Calculate simplified_dmeasure
    return simplified_dmeasure(Ps1, mu(Ps1), Ps2, mu(Ps2))

dmeasure_distances_matrix = get_distances_matrix(Gs, dmeasure)     # takes ~1 minute
#plot_distance_matrix(G_names, dmeasure_distances_matrix, "Distances based on D-measure")

# 2. Visualization

# General function for visualization

def visualize_graph(G, pos, width=15, height=8):
    """
    Plots the whole graph

    Parameters
    ------
    pos: dict <node label: coordinates>
    """
    # use nx.draw_networkx
    plt.figure(figsize=(width,height))
    nx.draw_networkx(
        G,
        pos=pos,
        nodelist=G.nodes(),
        labels=dict(zip(G.nodes(),G.nodes())),
        font_color='white',
        font_size=10,
    )
    plt.axis("off")
    plt.show()

pos = nx.spring_layout(G)  # Try different functions
visualize_graph(G, pos=pos)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_embeddings(G, size=15):
    """Get embeddings based on SVD-decomposition"""
    A = nx.adjacency_matrix(G).todense()
    U, _, _ = np.linalg.svd(A)
    embs = U[:,:size]
    return embs

def get_positions_by_embs(G):
    embs = get_embeddings(G)
    embs_2 = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embs)
    # embs_2 = PCA(n_components=2).fit_transform(embs)
    return dict(zip(G.nodes(), embs_2))

pos = get_positions_by_embs(G)
visualize_graph(G, pos=pos)


#xiv) ################################################################################################################################################################

def show_matrix(df, size=(8,8), title='', xlabel='', ylabel=''):
    """Plots datafram in the form of heatmap"""
    fig, ax = plt.subplots(figsize=size)
    sb.heatmap(df,square=True,annot=True, xticklabels=df.columns, yticklabels=df.columns, cmap='YlGnBu_r', ax=ax)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


def plot_bar(x, y, title='', size=(15,15), rot=0):
    """Plots bar chart"""
    fig, ax = plt.subplots(figsize=size)
    ax.bar(x, y)
    ax.set_yticks(y)
    ax.set_title(title)
    plt.xticks(rotation=rot)
    plt.show()


# for printing confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """Plots confusion matrix in the form of heatmap"""

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cur_df = pd.DataFrame(cm, columns=classes)
    size = (5, 5)
    if len(classes) <= 5:
        size = (5, 5)
    elif len(classes) <= 15:
        size = (7, 7)
    else:
        size = (12, 12)
    show_matrix(cur_df, size, title, xlabel='Predicted', ylabel='True')


def show_clf_metrics(y_true, y_preds):
    """Shows confusion matrix and """
    np.set_printoptions(precision=2)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_preds)
    plot_confusion_matrix(y_true, y_preds, classes=classes,
                          normalize=True, title='Normalized confusion matrix')
    print()
    print(classification_report(y_true, y_preds, labels=classes))

# visualization of classification

def plot_clusters(G, clusters_G, node_poses, title=''):
    """Plots clusters"""
    #viz.plot_network_clusters(G, clusters_G, node_poses, figsize=(6,6))
    plt.title(title)
    plt.show()


def plot_graph_clusters_and_preds(G, true_partition, pred_partition):
    """Plots true and predicted clusters"""
    node_poses = nx.spring_layout(G)
    plot_clusters(G, true_partition, node_poses, 'True')
    plot_clusters(G, pred_partition, node_poses, 'Predicted')

def read_net(folder, file_name):
    """Read network"""
    G = nx.Graph(name = file_name)
    with open(os.path.join(folder, file_name), 'r', encoding='utf8') as f:
        f.readline()
        # add nodes
        for line in f:
            if line.startswith("*"):
                break
            else:
                node_info = line.split("\"")
                node = int(node_info[0]) - 1
                label = node_info[1]
                cluster = int(node_info[2]) if len(node_info) > 2 and len(node_info[2].strip()) > 0 else 0
                G.add_node(node, label=label, cluster=cluster)
        # add edges
        for line in f:
            node1_str, node2_str = line.split()[:2]
            G.add_edge(int(node1_str)-1, int(node2_str)-1, weight = 1)
    return G

#statistics

def print_statistics(G):
    """Prints statistical info about the graph G:
    1) #Nodes
    2) #Edges
    3) Average node degree
    4) Density
    """
    print('Graph name:', G.name)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # print #nodes
    print('#Nodes:', n)
    # print #edges
    print('#Edges:', m)
    # print average degree
    print("Average Degree: {:.4f}".format(2 * m / n))
    # print Density
    print("Density {:.4f}".format( 2 * m / n / (n - 1)))

    # print LCC
    C = list(nx.connected_components(G))
    print("LCC: {:.2f}% ({:,d})".format(100 * max(len(c) for c in C) / n, len(C)))

# I. Node classification with features

def get_graph_partition_for_viz(G, node_to_label):
    """"Transforms graph idea partition to cdlib format"""
    # Transform nodes -> [list of nodes in cluster 1, list of nodes in cluster 2, ....]
    # Use cluster attribute of each node
    P = {}
    for node in G.nodes():
        if node_to_label[node] not in P:
            P[node_to_label[node]] = []
        P[node_to_label[node]].append(node)
    node_clusters = P.values()
    #return NodeClustering(list(node_clusters), G, '')

# train and evaluate classification
from sklearn.ensemble import RandomForestClassifier

def train_ml_model(X_train, y_train):
    """Inits and trains classifier"""
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_ml_model(clf, X_test, y_test):
    """Evaluates classfier"""
    preds = clf.predict(X_test)
    show_clf_metrics(y_test, preds)
    return preds


def plot_feature_importances(feature_names, clf):
    """Plots feature importances of your favorite classifier (example for RF)"""
    forest_importances = pd.Series(clf.feature_importances_, index=feature_names)  # importances of all features
    tree_std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)  # std of tree importances for each feature
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=tree_std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

# fearture extraction

def get_clusters_using_cdlib(G, algo_fn):
    """Gets dict node to cluster based on algo_fn"""
    clusters = {}
    for cluster_label, cluster_nodes in enumerate(algo_fn(G).communities):
        for node in cluster_nodes:
            clusters[node] = cluster_label
    return clusters


def features_dict_to_df(G, features_dict):
    """Transforms features from dicts to lists and then to dataframe"""
    res_features_dict = {}
    for feature_name in features_dict:
        res_features_dict[feature_name] = [node for node in G.nodes()]
    return pd.DataFrame().from_dict(res_features_dict)


def extract_node_features(G):
    """Extracts node features"""
    features = {}
    # get centralities
    features['pagerank'] = nx.pagerank(G) # pagerank
    features['degree'] = nx.degree_centrality(G)  # degree
    features['closeness'] = nx.closeness_centrality(G)  # closeness
    features['betweenness'] = nx.betweenness_centrality(G)  # betweenness

    # extract clusters using different methods
    features['nx_clustering'] = nx.clustering(nx.Graph(G))  # clustering from networkx
    #features['louvain'] = get_clusters_using_cdlib(G, algorithms.louvain)  # louvain
    #features['infomap'] = get_clusters_using_cdlib(G, algorithms.infomap)  # infomap
    #features['sbm'] = get_clusters_using_cdlib(G, algorithms.sbm_dl)  # sbm
    features['node_id'] = list(G.nodes())  # add nodes ids

    return features_dict_to_df(G, features)


# training pipeline on node classification task

def node_classification_pipeline(G, node_features, test_rate=0.2, plot_graph=False, plot_feat_importances=True):
    """Pipeline for node classification"""

    # get dict node to its cluster in G
    node_to_cluster = nx.get_node_attributes(G, "cluster")
    # get list of cluster labels in the order of node_id in node_features
    cluster_labels = [node_to_cluster[node] for node in node_features['node_id']]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(node_features, cluster_labels, test_size=test_rate,
                                                        random_state=42)

    # Train classfier
    clf = train_ml_model(X_train.drop(['node_id'], axis=1), y_train)

    # Evaluate classifier
    y_preds = evaluate_ml_model(clf, X_test.drop(['node_id'], axis=1), y_test)

    # Plot feature importances
    if plot_feat_importances:
        plot_feature_importances(node_features.drop(['node_id'], axis=1).columns, clf)

    # Plot graph with predicted labels
    if plot_graph:
        node_to_cluster.update({node: -1 for i, node in enumerate(X_train['node_id'])})
        true_partition = get_graph_partition_for_viz(G, node_to_cluster)
        node_to_cluster.update({node: y_preds[i] for i, node in enumerate(X_test['node_id'])})
        pred_partition = get_graph_partition_for_viz(G, node_to_cluster)
        plot_graph_clusters_and_preds(G, true_partition, pred_partition)

max_node_cnt_for_plot = 120

for G in Gs:
    print(G.name)
    node_features = extract_node_features(G)
    print()
    print('Starting node classification pipeline...')
    plot_graph = G.number_of_nodes() <= max_node_cnt_for_plot  # we don't plot too large graphs
    node_classification_pipeline(G, node_features, plot_graph=plot_graph)
    print('==============================================')
    print('==============================================')
    print('==============================================')

# Train nodes are those colored in red. Othere are test nodes

# II. Node classification with embeddings

def get_node_embeddings(G, dims = 32, p = 1, q = 1, walks = 32, length = 32, seed=10):
    """Trains Node2Vec"""
    #node2vec = Node2Vec(G, dimensions = dims, p = p, q = q, num_walks = walks, walk_length = length, workers = 8, quiet = True, seed=seed)
    #model = node2vec.fit()
    #return model.wv  # node embeddings


def node_embeddings_to_features(G, node_embeddings):
    """Transforms node embeddings to features"""
    data = []
    for node in G.nodes():
        embedding = node_embeddings[str(node)]
        data.append(embedding.tolist() + [node])
    feature_names = list(range(node_embeddings.vector_size)) + ['node_id']
    return pd.DataFrame(data=data, columns=feature_names)

max_node_cnt_for_plot = 120

for G in Gs:
    print(G.name)
    node_embeddings = get_node_embeddings(G)
    node_features = node_embeddings_to_features(G, node_embeddings)
    print()
    print('Starting node classification pipeline...')
    plot_graph = G.number_of_nodes() <= max_node_cnt_for_plot  # we don't plot too large graphs
    node_classification_pipeline(G, node_features, plot_graph=plot_graph, plot_feat_importances=False)
    print('==============================================')
    print('==============================================')
    print('==============================================')

# Train nodes are those colored in red. Othere are test nodes

# III. Link prediction with features

# train-test split for link prediction

random_generator = np.random.RandomState(seed=42)


def split_graph_for_link_pred(G, train_rate=0.8):
    """Split graph into train and test parts"""
    nodes = list(G.nodes())
    edges = list(G.edges())
    random_generator.shuffle(edges)

    # obtain negative edges
    non_edges = []
    while len(non_edges) < len(edges):
        i = random_generator.choice(nodes, 1)[0]
        j = random_generator.choice(nodes, 1)[0]
        if i != j and not G.has_edge(i, j):
            non_edges.append((i, j))

    # get id of edge as a border between train and test
    train_id_border = int(train_rate * len(edges))

    # get train graph
    G = nx.Graph(G)  # make a copy
    G.remove_edges_from(edges[train_id_border:])

    # get edges and their classes
    train_edges = edges[:train_id_border] + non_edges[:train_id_border]
    train_classes = [1] * train_id_border + [0] * len(non_edges[:train_id_border])

    test_edges = edges[train_id_border:] + non_edges[train_id_border:]
    test_classes = [1] * len(edges[train_id_border:]) + [0] * len(non_edges[train_id_border:])

    return G, dict(zip(train_edges, train_classes)), dict(zip(test_edges, test_classes))


# training pipeline on link prediction task

def prepare_features_labels_link_pred(edge_to_class_train, edge_to_class_test, edge_features):
    """Matches features and labels of the corresponding edges"""
    X_train, X_test, y_train, y_test = [], [], [], []
    edge_features_wout_ids = edge_features.drop(['v1', 'v2'], axis=1)
    edges = list(zip(edge_features['v1'], edge_features['v2']))
    # fill lists using cycle in edge_features_wout_ids
    for i, row in edge_features_wout_ids.iterrows():
        edge = edges[i]
        if edge in edge_to_class_train:
            X_train.append(row.values.tolist())
            y_train.append(edge_to_class_train[edge])
        else:
            X_test.append(row.values.tolist())
            y_test.append(edge_to_class_test[edge])
    return np.array(X_train), np.array(X_test), y_train, y_test


def link_prediction_pipeline(edge_to_class_train, edge_to_class_test, edge_features, plot_feat_importances=True):
    # get prepared data
    X_train, X_test, y_train, y_test = prepare_features_labels_link_pred(edge_to_class_train,
                                                                         edge_to_class_test,
                                                                         edge_features)
    # Train and evaluate a model
    clf = train_ml_model(X_train, y_train)
    y_preds = evaluate_ml_model(clf, X_test, y_test)

    # Plot feature importances
    if plot_feat_importances:
        plot_feature_importances(edge_features.drop(['v1', 'v2'], axis=1).columns, clf)


def preferential(G, i, j):
    return next(nx.preferential_attachment(G, [(i, j)]))[2]


def jaccard(G, i, j):
    return next(nx.jaccard_coefficient(G, [(i, j)]))[2]


def adamic_adar(G, i, j):
    return next(nx.adamic_adar_index(G, [(i, j)]))[2]


def extract_edge_features(G, edges):
    features_dict = defaultdict(list)

    # extract clusters using different methods
    #louvain = get_clusters_using_cdlib(G, algorithms.louvain)  # louvain
    #infomap = get_clusters_using_cdlib(G, algorithms.infomap)  # infomap
    #sbm = get_clusters_using_cdlib(G, algorithms.sbm_dl)  # sbm

    """for i, j in edges:
        # TODO
        features_dict['louvain'].append(int(louvain[i] == louvain[j]))  # have equal louvain clusters
        features_dict['infomap'].append(int(infomap[i] == infomap[j]))  # have equal infomap clusters
        features_dict['sbm'].append(int(sbm[i] == sbm[j]))  # have equal preferential clusters
        features_dict['preferential'].append(preferential(G, i, j))
        features_dict['jaccard'].append(jaccard(G, i, j))
        features_dict['adamic_adar'].append(adamic_adar(G, i, j))
        features_dict['v1'].append(i)  # add node1
        features_dict['v2'].append(j)  # add node2
    return pd.DataFrame().from_dict(features_dict)"""


for G in Gs:
    print(G.name)
    G_train, edge_to_class_train, edge_to_class_test = split_graph_for_link_pred(G)
    all_edges = {**edge_to_class_train, **edge_to_class_test}.keys()
    edge_features = extract_edge_features(G_train, all_edges)
    print()
    print('Starting link prediction pipeline...')

    link_prediction_pipeline(edge_to_class_train, edge_to_class_test, edge_features)
    print('==============================================')
    print('==============================================')
    print('==============================================')

# IV. Link prediction with embeddings

def edge_embeddings_to_features(edge_embeddings, edges):
    """Transforms embeddings to features"""
    data = []
    for (i, j) in edges:
        embedding = edge_embeddings[(str(i), str(j))]
        data.append(embedding.tolist() + [i, j])
    feature_names = list(range(edge_embeddings.kv.vector_size)) + ['v1', 'v2']
    return pd.DataFrame(data=data, columns=feature_names)


#from node2vec.edges import AverageEmbedder

for G in Gs:
    print(G.name)
    G_train, edge_to_class_train, edge_to_class_test = split_graph_for_link_pred(G)
    all_edges = {**edge_to_class_train, **edge_to_class_test}.keys()
    node_embeddings = get_node_embeddings(G_train, seed=7)
    #edge_embeddings = AverageEmbedder(node_embeddings, quiet=True)
    #edge_features = edge_embeddings_to_features(edge_embeddings, all_edges)
    print()
    print('Starting link prediction pipeline...')

    #link_prediction_pipeline(edge_to_class_train, edge_to_class_test, edge_features)
    print('==============================================')
    print('==============================================')
    print('==============================================')