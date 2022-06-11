def node_features(G, path=PATH):
    PR = nx.pagerank(G)
    DC = nx.degree_centrality(G)
    CC = nx.closeness_centrality(G)
    BC = nx.betweenness_centrality(G)
    C = nx.clustering(nx.Graph(G))

    louvain = {}
    for c, cluster in enumerate(algorithms.louvain(G).communities):
        for i in cluster:
            louvain[i] = c

    infomap = {}
    for c, cluster in enumerate(algorithms.infomap(G).communities):
        for i in cluster:
            infomap[i] = c

    sbm = {}
    for c, cluster in enumerate(algorithms.sbm_dl(G).communities):
        for i in cluster:
            sbm[i] = c

    with open(path + "/" + name + ".nodes.features.tab", 'w') as file:
        file.write(
            "mS#id\tmS#node\tC#degree\tC#pagerank\tC#closeness\tC#betweenness\tC#clustering\tD#louvain\tD#infomap\tD#sbm\tcD#class\n")
        for i, node in G.nodes(data=True):
            file.write(
                "{:s}\t{:s}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:d}\t{:d}\t{:d}\t{:d}\n".format(i, node['label'], DC[i],
                                                                                            PR[i], CC[i], BC[i], C[i],
                                                                                            louvain[i], infomap[i],
                                                                                            sbm[i], node['cluster']))
