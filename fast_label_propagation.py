import time
import random

import numpy as np
import networkx as nx
from cdlib.classes import *
from cdlib import algorithms

from collections import deque

def fast_label_propagation(G):
    N = list(G.nodes())
    random.shuffle(N)

    Q = deque(N)
    S = [True] * len(G)

    C = [i for i in range(len(G))]

    while Q:
        i = Q.popleft()
        S[i] = False

        if len(G[i]) > 0:
            N = {}
            for j in G[i]:
                if C[j] in N:
                    N[C[j]] += 1  # len(G[i][j])
                else:
                    N[C[j]] = 1  # len(G[i][j])

            maxn = max(N.values())
            c = random.choice([c for c in N if N[c] == maxn])

            if C[i] != c:
                C[i] = c
                for j in G[i]:
                    if C[j] != c and not S[j]:
                        Q.append(j)
                        S[j] = True

    L = {}
    for i, c in enumerate(C):
        if c in L:
            L[c].append(i)
        else:
            L[c] = [i]

    return NodeClustering(list(L.values()), G, 'FLPA')
