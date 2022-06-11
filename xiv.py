import random

import networkx as nx
from cdlib import algorithms

from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder

PATH = "/Users/lovre/Documents/office/coding/repositories/www/ina/nets"


def info(G):
  print("{:>12s} | '{:s}'".format('Graph', G.name))

  n = G.number_of_nodes()
  m = G.number_of_edges()
  
  print("{:>12s} | {:,d} ({:,d})".format('Nodes', n, nx.number_of_isolates(G)))
  print("{:>12s} | {:,d} ({:,d})".format('Edges', m, nx.number_of_selfloops(G)))
  print("{:>12s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, max([k for _, k in G.degree()])))
  
  C = list(nx.connected_components(G))
    
  print("{:>12s} | {:.1f}% ({:,d})".format('LCC', 100 * max(len(c) for c in C) / n, len(C)))
  print()
  
  return G
  
def train_graph(G, train = 0.8):
  nodes = list(G.nodes())
  edges = list(G.edges())
  random.shuffle(edges)
  
  non_edges = []
  while len(non_edges) < len(edges):
    i = random.choice(nodes)
    j = random.choice(nodes)
    if i != j and not G.has_edge(i, j):
      non_edges.append((i, j))
  
  train = int(train * len(edges))
  G = nx.Graph(G)
  G.remove_edges_from(edges[train:])
  
  return G, {"train": {1: edges[:train], 0: non_edges[:train]}, "test": {1: edges[train:], 0: non_edges[train:]}}





for name in ["karate_club", "southern_women", "american_football", "sicris_collaboration", "cdn_java", "board_directors"]:
  G = read(name)

  info(G)
  
  node_features(G)

  node_embeddings(G)
  
  G, edges = train_graph(G)
  
  edge_features(G, edges)

  edge_embeddings(G, edges)
