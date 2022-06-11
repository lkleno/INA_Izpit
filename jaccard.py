def jaccard(G, i, j):
    return next(nx.jaccard_coefficient(G, [(i, j)]))[2]
