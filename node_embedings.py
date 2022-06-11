def node_embeddings(G, dims = 32, p = 1, q = 1, walks = 32, length = 32, path = PATH):
  n2v = Node2Vec(G, dimensions = dims, p = p, q = q, num_walks = walks, walk_length = length, workers = 8, quiet = True).fit().wv

  with open(path + "/" + name + ".nodes.node2vec.tab", 'w') as file:
    file.write("mS#id\tmS#node\t" + "\t".join(["C#n2v-" + str(i) for i in range(dims)]) + "\tcD#class\n");
    for i, node in G.nodes(data = True):
      file.write(i + "\t" + node['label'] + "\t" + "\t".join([str(v) for v in n2v[i]]) + "\t" + str(node['cluster']) + "\n")
