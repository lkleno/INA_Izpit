def edge_embeddings(G, edges, dims=32, p=1, q=1, walks=32, length=32, path=PATH):
    n2v = Node2Vec(G, dimensions=dims, p=p, q=q, num_walks=walks, walk_length=length, workers=8, quiet=True).fit().wv
    e2v = HadamardEmbedder(n2v, quiet=True)

    with open(path + "/" + name + ".edges.node2vec.tab", 'w') as file:
        file.write("mS#id1\tmS#id2\tmD#data\t" + "\t".join(["C#e2v-" + str(i) for i in range(dims)]) + "\tcD#class\n");
        for data in edges:
            for c in edges[data]:
                for i, j in edges[data][c]:
                    file.write(
                        i + "\t" + j + "\t" + data + "\t" + "\t".join([str(v) for v in e2v[(i, j)]]) + "\t" + str(
                            c) + "\n")
