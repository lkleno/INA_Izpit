
import networkx as nx


def read(name, path=PATH):
    G = nx.MultiGraph(name=name)
    with open(path + "/" + name + ".net", 'r') as file:
        file.readline()

        for line in file:
            if line.startswith("*"):
                break
            else:
                G.add_node(int(line.split(" ")[0]) - 1)

        for line in file:
            i, j = (int(x) - 1 for x in line.split()[:2])
            if i != j:
                G.add_edge(i, j)

    return G


def read(name, path=PATH):
    G = nx.MultiGraph(name=name)
    with open(path + "/" + name + ".net", 'r') as file:
        file.readline()

        for line in file:
            if line.startswith("*"):
                break
            else:
                node = line.split("\"")
                G.add_node(node[0].strip(), label=node[1],
                           cluster=int(node[2]) if len(node) > 2 and len(node[2].strip()) > 0 else 0)

        for line in file:
            G.add_edge(*line.split()[:2], weight=1)

    return G
