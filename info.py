import time
import networkx as nx

def info(G, fast=True):
    tic = time.time()
    print("{:>12s} | '{:s}'".format('Graph', G.name))

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print("{:>12s} | {:,d}".format('Nodes', n))
    print("{:>12s} | {:,d}".format('Edges', m))
    print("{:>12s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, max([k for _, k in G.degree()])))

    if not fast:
        C = list(nx.connected_components(G))

        print("{:>12s} | {:.1f}% ({:,d})".format('LCC', 100 * max(len(c) for c in C) / n, len(C)))

        # C = algorithms.label_propagation(G)
        C = fast_label_propagation(G)

        print("{:>12s} | {:.3f} ({:,d})".format('Q', C.newman_girvan_modularity().score, len(C.communities)))

    print("{:>12s} | {:.1f} s".format('Time', time.time() - tic))
    print()


def info(G):
    print("{:>10s} | '{:s}'".format('Graph', G.name))

    tic = time.time()
    n = G.number_of_nodes()
    n0, n1, delta = 0, 0, 0
    for i in G.nodes():
        if isolated(G, i):
            n0 += 1
        elif G.degree(i) == 1:
            n1 += 1
        if G.degree(i) > delta:
            delta = G.degree(i)

    print("{:>10s} | {:,d} ({:,d}, {:,d})".format('Nodes', n, n0, n1))

    m = G.number_of_edges()
    m0 = nx.number_of_selfloops(G)

    print("{:>10s} | {:,d} ({:,d})".format('Edges', m, m0))
    print("{:>10s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, delta))
    print("{:>10s} | {:.2e}".format('Density', 2 * m / n / (n - 1)))

    C = components(G)

    print("{:>10s} | {:.1f}% ({:,d})".format('LCC', 100 * max(len(c) for c in C) / n, len(C)))

    D = distances(G, 100)
    D = [i for d in D for i in d]

    print("{:>10s} | {:.2f} ({:,d})".format('Distance', sum(D) / len(D), max(D)))

    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    print("{:>10s} | {:.4f}".format('Clustering', nx.average_clustering(G)))

    print("{:>10s} | {:.1f} s".format('Time', time.time() - tic))
    print()
