import time
import random
import networkx as nx
from collections import deque

def isolated(G, i):
  for j in G[i]:
    if j != i:
      return False
  return True
  
def component(G, N, i):
  C = []
  S = []
  N.remove(i)
  S.append(i)
  while S:
    i = S.pop()
    C.append(i)
    for j in G[i]:
      if j in N:
        N.remove(j)
        S.append(j)
  return C

def components(G):
  C = []
  N = set(G.nodes())
  while N:
    C.append(component(G, N, next(iter(N))))
  return C
  
def distance(G, i):
  D = [-1] * len(G) # D = {}
  Q = deque()
  D[i] = 0
  Q.append(i)
  while Q:
    i = Q.popleft()
    for j in G[i]:
      if D[j] == -1: # if j not in D:
        D[j] = D[i] + 1
        Q.append(j)
  return [d for d in D if d > 0]
  


  



for name in ["toy", "karate_club", "collaboration_imdb", "www_google"]:
  # G = nx.Graph(nx.read_pajek("/Users/lovre/Downloads/" + name + ".net"))
  # G.name = name
  
  G = nx.Graph(name = name)
  with open("/Users/lovre/Downloads/" + name + ".net", 'r') as file:
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
  
  info(G)
  
  info(erdos_renyi(G.number_of_nodes(), G.number_of_edges()))
  
  info(configuration([k for _, k in G.degree()]))
