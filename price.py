def price(n, c, a):
  G = nx.MultiDiGraph(name = "price")
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