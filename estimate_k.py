import time
import random

import numpy as np
import networkx as nx
from cdlib.classes import *
from cdlib import algorithms

from collections import dequeimport time

def estimate_k(G, size=0.15):
    g = [[] for _ in range(len(G))]
    for i in G.nodes():
        for j in G[i]:
            g[i].extend([j] * len(G[i][j]))

    i = random.randint(0, len(g) - 1)
    sumk, sumk_1 = len(g[i]), 1 / len(g[i])
    s = 1

    while s < size * len(g):
        i = random.choice(g[i])
        sumk += len(g[i])
        sumk_1 += 1 / len(g[i])
        s += 1

    return sumk / s, s / sumk_1