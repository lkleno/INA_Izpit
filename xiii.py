
import networkx as nx

import portrait_divergence as pd
import simplified_dmeasure as dm
import graphlet_aggrement as ga

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PATH = "/Users/lovre/Documents/office/coding/repositories/www/ina/nets"

def read(name, path = PATH):
  G = nx.Graph(name = name)
  with open(path + "/" + name + ".net", 'r') as file:
    file.readline()

    for line in file:
      if line.startswith("*"):
        break
      else:
        G.add_node(int(line.split(" ")[0]) - 1)

    for line in file:
      i, j = (int(x) - 1 for x in line.split()[:2])
      if i != j:
        G.add_edge(i, j)
      
  return G
  
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
  

  
MEASURES = {"dmeasure": dm.dmeasure, "portraits": pd.portrait_divergence, "graphlets": ga.graphlet_disaggrement} # "edit": nx.algorithms.similarity.graph_edit_distance

N = 500
K = 6

Gs = []
for name in ["karate_club", "southern_women", "dolphins", "foodweb_littlerock", "foodweb_baydry", "foodweb_baywet"]:
  Gs.append(info(read(name)))

for i in range(3):
  G = nx.gnm_random_graph(N, N * K // 2)
  G.name = "erdos_renyi_" + str(i + 1)
  if i == 0:
    info(G)
  Gs.append(G)

for i in range(3):
  G = nx.barabasi_albert_graph(N, round(K / 2))
  G.name = "barabasi_albert_" + str(i + 1)
  if i == 0:
    info(G)
  Gs.append(G)

for i in range(3):
  G = nx.watts_strogatz_graph(N, K, 0.1)
  G.name = "watts_strogatz_" + str(i + 1)
  if i == 0:
    info(G)
  Gs.append(G)

for label, measure in MEASURES.items():
  plot(Gs, dists(Gs, measure), label)
