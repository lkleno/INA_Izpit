def isolated(G, i):
    for j in G[i]:
        if j != i:
            return False
    return True
