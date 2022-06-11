def adamic_adar(G, i, j):
    return next(nx.adamic_adar_index(G, [(i, j)]))[2]
