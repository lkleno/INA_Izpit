def girvan_newman(mu=0.5):
    G = nx.Graph(name="girvan_newman")
    for i in range(128):
        G.add_node(i, cluster=i // 32 + 1)
    for i in range(128):
        for j in range(i + 1, 128):
            if G.nodes[i]['cluster'] == G.nodes[j]['cluster']:
                if random.random() < 16 * (1 - mu) / 31:
                    G.add_edge(i, j)
            else:
                if random.random() < mu / 6:
                    G.add_edge(i, j)
    return G