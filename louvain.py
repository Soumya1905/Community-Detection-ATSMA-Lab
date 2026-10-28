"""Louvain community detection (igraph multilevel implementation).

Run alone:  python louvain.py [--seed 42]
"""
import argparse
import time

import common

NAME = "Louvain"


def run(ds, seed=42, log=None):
    log = log or common.get_logger(NAME, common.RESULTS_DIR / NAME)
    common.seed_everything(seed)  # igraph draws its random vertex order from Python's RNG
    params = {"implementation": "igraph.Graph.community_multilevel", "resolution": 1.0,
              "weights": None, "return_levels": False,
              "note": "Number of communities is chosen automatically by greedy modularity optimisation."}
    t0 = time.perf_counter()
    clustering = ds.graph.community_multilevel(weights=None, return_levels=False, resolution=1.0)
    runtime = time.perf_counter() - t0
    return common.save_results(ds, NAME, clustering.membership, runtime, seed, params,
                               {"applicable": False}, {}, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    log = common.get_logger(NAME, common.RESULTS_DIR / NAME)
    run(common.load_dataset(log=log), a.seed, log)


if __name__ == "__main__":
    main()
