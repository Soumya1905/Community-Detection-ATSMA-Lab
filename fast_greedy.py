"""Fast Greedy (Clauset-Newman-Moore) community detection via igraph.

Agglomerative: every node starts alone and the pair of communities giving the largest
modularity gain is merged repeatedly, producing a dendrogram. The partition is selected by
cutting the dendrogram at the level of MAXIMUM modularity (igraph's `as_clustering()`
default, `optimal_count`). The count is not forced to match any other algorithm.
Deterministic (no randomness); isolated nodes stay singleton communities, and
disconnected components are never merged (igraph handles an incomplete dendrogram).

Run alone:  python fast_greedy.py
"""
import argparse
import time

import common

NAME = "Fast_Greedy"


def run(ds, log=None):
    log = log or common.get_logger(NAME, common.RESULTS_DIR / NAME)
    params = {"implementation": "igraph.Graph.community_fastgreedy + VertexDendrogram.as_clustering()",
              "weights": None,
              "cut_selection": "maximum-modularity level of the dendrogram (optimal_count)"}
    t0 = time.perf_counter()
    dendro = ds.graph.community_fastgreedy(weights=None)
    clustering = dendro.as_clustering()
    runtime = time.perf_counter() - t0
    return common.save_results(ds, NAME, clustering.membership, runtime, None, params,
                               {"applicable": False}, {"deterministic": True}, log)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    log = common.get_logger(NAME, common.RESULTS_DIR / NAME)
    run(common.load_dataset(log=log), log)


if __name__ == "__main__":
    main()
