def dists(Gs, measure):
    Ds = [[0] * len(Gs) for _ in range(len(Gs))]

    for i in range(len(Gs)):
        for j in range(i, len(Gs)):
            Ds[i][j] = measure(Gs[i], Gs[j])
            Ds[j][i] = Ds[i][j]

    return Ds
