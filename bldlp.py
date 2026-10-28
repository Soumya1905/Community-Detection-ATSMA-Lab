"""Balanced Link Density-based Label Propagation (BLDLP).

SOURCE (the only accessible reference): notebook `sna_partb.ipynb`, section 2.3, of
https://github.com/SophiaVei/Community-Detection-in-Social-Networks . The repo and notebook cite
no paper for BLDLP, so the equations below are reproduced from that code, not from a paper.

1. Edge weights (alpha = 15, the notebook's value):
     for each undirected edge (v, u), with v the endpoint visited first by networkx `G.edges`
     (= smaller node id, as nodes are inserted in ascending id order here):
       E_vu      = set of adjacency tuples (n, x), n in N(v) minus {u}, x in N(n), x != u
       strength  = |E_vu| + 1
       weight    = floor(alpha / strength)
   Equivalent closed form (verified against the literal notebook code in tests/):
       |E_vu| = sum_{n in N(v)} deg(n) - deg(u) - |N(v) & N(u)|
   Note: the weight is computed from v's neighbourhood only and is thus not symmetric in v,u;
   the notebook stores one value per undirected edge, which we reproduce.
2. Asynchronous label propagation (the notebook's modified networkx `asyn_lpa_communities`):
     * labels initialised: every node its own label (label = node index).
     * each iteration visits all nodes in a fresh random order (in-place/asynchronous updates);
       isolated nodes are skipped.
     * a node counts neighbour labels (unweighted frequency); the most frequent label wins.
     * TIE-BREAK 1: if several labels tie, recount using the edge weights above (sum of weights).
     * TIE-BREAK 2: if still tied, pick uniformly at random among the tied labels.
     * `cont` stays True for the iteration if any node still had >1 tied label after weighting.
     * STOP: when an iteration finishes with no remaining ties, or after `max_iter` iterations.
   The notebook gives no definition of "balance" beyond this weighting, which counter-balances
   plain frequency voting with link density around an edge.
max_iter default 10000 = notebook value. Because random tie-breaks can keep `cont` True indefinitely, convergence is not guaranteed and is
reported honestly.

Run alone:  python bldlp.py [--seed 42] [--alpha 15] [--max-iter 10000]
"""
import argparse
import math
import random
import time
from collections import Counter

import common

NAME = "BLDLP"


def compute_weights(adj, alpha):
    """Return list-of-dicts w[v][u]: one value per undirected edge, as in the notebook."""
    deg = [len(a) for a in adj]
    nset = [set(a) for a in adj]
    sum_deg = [sum(deg[n] for n in a) for a in adj]
    w = [dict() for _ in adj]
    for v in range(len(adj)):
        for u in adj[v]:
            if v < u:  # v = endpoint visited first by G.edges
                strength = sum_deg[v] - deg[u] - len(nset[v] & nset[u]) + 1
                val = math.floor(alpha / strength)
                w[v][u] = val
                w[u][v] = val
    return w


def propagate(adj, w, max_iter, rng):
    labels = list(range(len(adj)))
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        cont = False
        nodes = list(range(len(adj)))
        rng.shuffle(nodes)
        for node in nodes:
            nb = adj[node]
            if not nb:
                continue
            freq = Counter(labels[v] for v in nb)
            mx = max(freq.values())
            best = [l for l, f in freq.items() if f == mx]
            if len(best) > 1:
                wf = Counter()
                for v in nb:
                    wf[labels[v]] += w[node][v]
                mx = max(wf.values())
                best = [l for l, f in wf.items() if f == mx]
            labels[node] = rng.choice(best)
            cont = cont or len(best) > 1
        if not cont:
            converged = True
            break
    return labels, converged, it


def run(ds, seed=42, alpha=15, max_iter=10000, log=None):
    log = log or common.get_logger(NAME, common.RESULTS_DIR / NAME)
    rng = random.Random(seed)
    common.seed_everything(seed)
    adj = [sorted(a) for a in ds.graph.get_adjlist()]  # ascending index == ascending node id
    params = {"alpha": alpha, "max_iter": max_iter, "weight_formula": "floor(alpha / (|E_vu| + 1))",
              "update_order": "asynchronous, random permutation every iteration",
              "tie_break": "unweighted frequency -> link-density weight sum -> random choice",
              "initialisation": "unique label per node", "source": "SophiaVei notebook sna_partb.ipynb sec 2.3"}
    t0 = time.perf_counter()
    w = compute_weights(adj, alpha)
    t_w = time.perf_counter() - t0
    labels, converged, iters = propagate(adj, w, max_iter, rng)
    runtime = time.perf_counter() - t0
    log.info("BLDLP: weights %.2fs; %d iterations; converged=%s", t_w, iters, converged)
    if not converged:
        log.warning("Stopped at max_iter=%d with ties still unresolved (notebook stopping rule).", max_iter)
    return common.save_results(
        ds, NAME, labels, runtime, seed, params,
        {"converged": converged, "iterations": iters, "max_iter": max_iter},
        {"weight_computation_seconds": t_w}, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=10000)
    a = ap.parse_args()
    log = common.get_logger(NAME, common.RESULTS_DIR / NAME)
    run(common.load_dataset(log=log), a.seed, a.alpha, a.max_iter, log)


if __name__ == "__main__":
    main()
