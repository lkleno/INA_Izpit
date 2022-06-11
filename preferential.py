def preferential(G, i, j):
    return next(nx.preferential_attachment(G, [(i, j)]))[2]
