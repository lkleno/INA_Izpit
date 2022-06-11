def eigenvector_centrality(G, eps=1e-6):
    E = [1] * G.number_of_nodes()
    diff = 1
    while diff > eps:
        U = [sum([E[j] for j in G[i]]) for i in G.nodes()]
        u = sum(U)
        U = [U[i] * G.number_of_nodes() / u for i in G.nodes()]
        diff = sum([abs(E[i] - U[i]) for i in G.nodes()])
        E = U
    return {i: E[i] for i in range(len(E))}
