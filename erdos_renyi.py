def erdos_renyi(n, m):
  G = nx.MultiGraph(name = "erdos_renyi")
  for i in range(n):
    G.add_node(i)
  edges = []
  while len(edges) < m:
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    if i != j:
      edges.append((i, j))
  G.add_edges_from(edges)
  return G