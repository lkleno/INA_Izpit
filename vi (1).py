import time
import math
import random
import operator

import networkx as nx
import matplotlib.pyplot as plt


  


  
def read(name, path = "/Users/lovre/Downloads"):
  G = nx.MultiGraph(name = name)
  with open(path + "/" + name + ".net", 'r') as file:
    file.readline()

    for line in file:
      if line.startswith("*"):
        break
      else:
        node = line.split("\"")
        G.add_node(int(node[0]) - 1, label = node[1])

    for line in file:
      i, j = (int(x) - 1 for x in line.split()[:2])
      G.add_edge(i, j)
      
  return G
  
def info(G, kmin = 5, fast = False):
  print("{:>12s} | '{:s}'".format('Graph', G.name))

  tic = time.time()
  n = G.number_of_nodes()
  m = G.number_of_edges()
  
  print("{:>12s} | {:,d} ({:,d})".format('Nodes', n, nx.number_of_isolates(G)))
  print("{:>12s} | {:,d} ({:,d})".format('Edges', m, nx.number_of_selfloops(G)))
  print("{:>12s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, max([k for _, k in G.degree()])))
  print("{:>12s} | {:.2e}".format('Density', 2 * m / n / (n - 1)))
  
  print("{:>12s} | {:.2f} ({:d})".format('Gamma', power_law(G, kmin), kmin))
  
  if not fast:
    if isinstance(G, nx.DiGraph):
      G = nx.MultiGraph(G)

    C = list(nx.connected_components(G))

    print("{:>12s} | {:.1f}% ({:,d})".format('LCC', 100 * max(len(c) for c in C) / n, len(C)))

    D = distances(G)

    print("{:>12s} | {:.2f} ({:,d})".format('Distance', sum(D) / len(D), max(D)))

    if isinstance(G, nx.MultiGraph):
      G = nx.Graph(G)

    print("{:>12s} | {:.4f}".format('Clustering', nx.average_clustering(G)))
  
    print("{:>12s} | {:.1f} s".format('Time', time.time() - tic))
  print()
  
  return G
  


for name in ["karate_club", "darknet", "collaboration_imdb", "wikileaks", "enron", "www_google"]:
  G = read(name, path = "/Users/lovre/Documents/office/coding/repositories/www/ina/nets")

  info(G)
  plot(G)

  n = G.number_of_nodes()
  m = G.number_of_edges()
  c = round(m / n)

  gamma = power_law(G, kmin = 5)
  a = c if gamma <= 2 else c * (gamma - 2)

  G = price(n, c, a)

  info(G)
  plot(G)

  G = barabasi_albert(n, c)
  # G = nx.barabasi_albert_graph(n, c)

  info(G)
  plot(G)

n = 100000
c = 10

