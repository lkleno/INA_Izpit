def closeness_centrality(G):
  return {i: sum([1 / d for d in distance(G, i)]) / (G.number_of_nodes() - 1) for i in G.nodes()}
