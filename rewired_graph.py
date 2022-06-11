def to_hash(i, j):
  if i <= j:
    i, j = j, i
  return i * (i - 1) // 2 + j


def rewired_graph(G, num = 100):
  H = set()
  edges = []
  for edge in G.edges():
    i, j = edge
    h = to_hash(i, j)
    if h not in H:
      edges.append([i, j])
      H.add(h)
  for _ in range(num):
    eij = random.randint(0, len(edges) - 1)
    euv = random.randint(0, len(edges) - 1)
    if eij != euv:
      i, j = edges[eij]
      u, v = edges[euv]
      if random.random() < 0.5:
        if i != v and u != j:
          hiv = to_hash(i, v)
          huj = to_hash(u, j)
          if hiv not in H and huj not in H:
            edges[eij][1] = v
            edges[euv][1] = j
            H.remove(to_hash(i, j))
            H.remove(to_hash(u, v))
            H.add(hiv)
            H.add(huj)
      else:
        if i != u and j != v:
          hiu = to_hash(i, u)
          hjv = to_hash(j, v)
          if hiu not in H and hjv not in H:
            edges[eij][1] = u
            edges[euv][0] = j
            H.remove(to_hash(i, j))
            H.remove(to_hash(u, v))
            H.add(hiu)
            H.add(hjv)
  G = nx.empty_graph(len(G))
  G.add_edges_from(edges)
  return G