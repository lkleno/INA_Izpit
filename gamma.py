for gamma in [2.01, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0]:
  G = price(n, c, c * (gamma - 2))
  G.name += "_" + str(gamma)

  info(G, kmin = 25)
  plot(G)
