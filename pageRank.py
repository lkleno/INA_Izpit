def pagerank(G, alpha = 0.85, eps = 1e-6):
  P = [1 / G.number_of_nodes()] * G.number_of_nodes()
  diff = 1
  while diff > eps:
    U = [sum([P[j] * alpha / G.degree(j) for j in G[i]]) for i in G.nodes()]
    u = sum(U)
    U = [U[i] + (1 - u) / G.number_of_nodes() for i in G.nodes()]
    diff = sum([abs(P[i] - U[i]) for i in G.nodes()])
    P = U
  return {i: P[i] for i in range(len(P))}
