def configuration(degrees):
    G = nx.MultiGraph(name="configuration")
    nodes = []
    for i in range(len(degrees)):
        G.add_node(i)
        for _ in range(degrees[i]):
            nodes.append(i)
    random.shuffle(nodes)
    edges = []
    for i in range(0, len(nodes), 2):
        edges.append((nodes[i], nodes[i + 1]))
    G.add_edges_from(edges)
    return G
