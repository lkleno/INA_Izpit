#def distances(G, n=100):
#    D = []  # D = {}
#    for i in G.nodes() if len(G) <= n else random.sample(G.nodes(), n):
#        D.append(distance(G, i))  # D[i] = distance(G, i)
#    return D


def distances(G, n=100):
    D = []
    for i in G.nodes() if len(G) <= n else random.sample(list(G.nodes()), n):
        D.extend([d for d in nx.shortest_path_length(G, source=i).values() if d > 0])
    return D

