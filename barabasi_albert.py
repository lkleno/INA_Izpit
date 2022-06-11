def barabasi_albert(n, c):
    return nx.MultiGraph(price(n, c, c), name="barabasi_albert")
