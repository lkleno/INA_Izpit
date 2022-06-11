import time
import random

import numpy as np
import networkx as nx
from cdlib.classes import *
from cdlib import algorithms

from collections import deque

def lcc(G):
    return nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)))
