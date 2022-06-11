def edge_features(G, edges, path=PATH):
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

    with open(path + "/" + name + ".edges.features.tab", 'w') as file:
        file.write(
            "mS#id1\tmS#id2\tmD#data\tC#preferential\tC#jaccard\tC#adamic_adar\tD#louvain\tD#infomap\tD#sbm\tcD#class\n");
        for data in edges:
            for c in edges[data]:
                for i, j in edges[data][c]:
                    file.write(i + "\t" + j + "\t" + data + "\t" + str(preferential(G, i, j)) + "\t" + str(
                        jaccard(G, i, j)) + "\t" + str(adamic_adar(G, i, j)) + "\t" + (
                                   "1" if louvain[i] == louvain[j] else "0") + "\t" + (
                                   "1" if infomap[i] == infomap[j] else "0") + "\t" + (
                                   "1" if sbm[i] == sbm[j] else "0") + "\t" + str(c) + "\n")
