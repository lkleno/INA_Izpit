def degree_mixing(G, source = None, target = None):
  x, y = [], []
  for i, j in G.edges():
    if source != None and target != None:
      x.append(G.out_degree(i) if source == 'out' else G.in_degree(i))
      y.append(G.in_degree(j) if target == 'in' else G.out_degree(j))
    else:
      x.append(G.degree(i))
      y.append(G.degree(j))
      x.append(G.degree(j))
      y.append(G.degree(i))
  return stats.pearsonr(x, y)[0]