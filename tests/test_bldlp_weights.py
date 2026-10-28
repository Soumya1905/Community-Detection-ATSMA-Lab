"""Check the closed-form BLDLP weights against a literal port of the notebook's loop."""
import math
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bldlp  # noqa: E402


def notebook_weights(G, alpha):
    out = {}
    for v, u in G.edges:
        edges_neigh_v = []
        for n in G.neighbors(v):
            edges_neigh_v.extend(list(G.edges(n)))
        e = [tuple(x) for x in edges_neigh_v if u not in x]
        out[(v, u)] = math.floor(alpha / (len(set(e)) + 1))
    return out


def test_weights_match():
    G = nx.gnm_random_graph(60, 220, seed=3)
    G = nx.Graph(G)  # nodes 0..59 inserted ascending
    adj = [sorted(G.neighbors(i)) for i in range(60)]
    w = bldlp.compute_weights(adj, 15)
    ref = notebook_weights(G, 15)
    assert ref and all(w[v][u] == val for (v, u), val in ref.items())


if __name__ == "__main__":
    test_weights_match()
    print("ok")
