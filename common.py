"""Shared utilities: dataset loading/validation, metrics, plotting, result writing.

Graph policy (applied identically for every algorithm):
  * The graph is undirected and simple: self-loops and duplicate edges
    (including A-B / B-A pairs) are dropped; counts are reported.
  * Modularity is Newman's unweighted, undirected modularity with resolution 1,
    computed by igraph on this single graph for all three algorithms.
"""
import json
import logging
import platform
import random
import sys
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "lasftm_asia"
RESULTS_DIR = ROOT / "results"
EDGES_FILE = "lastfm_asia_edges.csv"
TARGET_FILE = "lastfm_asia_target.csv"
EXPECTED_NODES, EXPECTED_EDGES = 7624, 27806
LAYOUT_SEED = 42
ALGO_DIRS = {"Louvain": "Louvain", "Fast_Greedy": "Fast_Greedy", "BLDLP": "BLDLP"}


class Dataset:
    """Prepared graph. igraph vertex i corresponds to original node id node_ids[i]."""

    def __init__(self, graph, node_ids, edges, target, prep):
        self.graph, self.node_ids, self.edges = graph, node_ids, edges
        self.target, self.prep = target, prep  # target: dict id->class


def get_logger(name, outdir=None):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(outdir) / f"{name}_run_log.txt", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def load_dataset(data_dir=DATA_DIR, log=None):
    """Load, validate and prepare the graph once. Original files are only read."""
    log = log or logging.getLogger("data")
    data_dir = Path(data_dir)
    ep, tp = data_dir / EDGES_FILE, data_dir / TARGET_FILE
    for p in (ep, tp):
        if not p.exists():
            raise FileNotFoundError(f"Required dataset file missing: {p}")
    edges = pd.read_csv(ep)
    if list(edges.columns) != ["node_1", "node_2"]:
        raise ValueError(f"Unexpected edge-file columns {list(edges.columns)}")
    tgt = pd.read_csv(tp)
    if list(tgt.columns) != ["id", "target"]:
        raise ValueError(f"Unexpected target-file columns {list(tgt.columns)}")
    if edges.isna().any().any():
        raise ValueError("Edge file contains missing values")

    raw = len(edges)
    a, b = edges["node_1"].to_numpy(), edges["node_2"].to_numpy()
    self_loops = int((a == b).sum())
    keep = a != b
    lo, hi = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
    pairs = pd.DataFrame({"u": lo, "v": hi})
    dup = int(pairs.duplicated().sum())
    pairs = pairs.drop_duplicates()
    edge_list = list(zip(pairs["u"].tolist(), pairs["v"].tolist()))

    node_ids = sorted(set(tgt["id"].tolist()) | set(pairs["u"]) | set(pairs["v"]))
    in_target = set(tgt["id"].tolist())
    missing_target = [n for n in node_ids if n not in in_target]
    index = {n: i for i, n in enumerate(node_ids)}
    g = ig.Graph(n=len(node_ids), edges=[(index[u], index[v]) for u, v in edge_list], directed=False)
    g.vs["orig_id"] = node_ids
    comps = len(g.connected_components())
    prep = {
        "raw_edge_rows": raw, "self_loops_removed": self_loops,
        "duplicate_edges_removed": dup, "edges_after_cleaning": g.ecount(),
        "nodes": g.vcount(), "isolated_nodes": int(sum(d == 0 for d in g.degree())),
        "nodes_without_target_label": len(missing_target),
        "connected_components": comps,
        "directed": False,
        "policy": "Undirected simple graph; self-loops and duplicate (incl. reversed) edges dropped; "
                  "no nodes removed; whole graph (all components) used.",
    }
    log.info("Loaded %s: %d nodes, %d edges (raw rows %d, self-loops %d, duplicates %d), %d components",
             EDGES_FILE, g.vcount(), g.ecount(), raw, self_loops, dup, comps)
    if (g.vcount(), g.ecount()) != (EXPECTED_NODES, EXPECTED_EDGES):
        log.warning("DISCREPANCY vs expected %d nodes / %d edges", EXPECTED_NODES, EXPECTED_EDGES)
    else:
        log.info("Matches expected %d nodes / %d edges", EXPECTED_NODES, EXPECTED_EDGES)
    return Dataset(g, node_ids, edge_list, dict(zip(tgt["id"], tgt["target"])), prep)


def versions():
    import matplotlib as m, networkx as nx, sklearn
    return {"python": platform.python_version(), "igraph": ig.__version__, "networkx": nx.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": m.__version__,
            "scikit-learn": sklearn.__version__}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def relabel_by_size(membership):
    """Community IDs are arbitrary; relabel 0..k-1 by descending size (ties by first member)."""
    first = {}
    for i, c in enumerate(membership):
        first.setdefault(c, i)
    sizes = pd.Series(membership).value_counts()
    order = sorted(sizes.index, key=lambda c: (-sizes[c], first[c]))
    m = {c: k for k, c in enumerate(order)}
    return [m[c] for c in membership]


def modularity(ds, membership):
    return float(ds.graph.modularity(membership))


# ---------------------------------------------------------------- top-3 metrics
def community_stats(g, nodes, target_of):
    """Structural metrics of the subgraph induced by `nodes` (igraph indices).

    Path length / diameter are only defined for connected graphs. If the community is
    disconnected they are computed on its largest connected component (scope is recorded).
    """
    sub = g.induced_subgraph(nodes)
    n, m = sub.vcount(), sub.ecount()
    density = 2 * m / (n * (n - 1)) if n > 1 else None
    clustering = float(np.mean(sub.transitivity_local_undirected(mode="zero"))) if n else None
    comps = sub.connected_components()
    connected = len(comps) == 1
    notes = []
    if connected:
        core, scope = sub, "whole community"
    else:
        big = max(comps, key=len)
        core = sub.induced_subgraph(big)
        scope = f"largest connected component ({len(big)} of {n} nodes)"
        notes.append(f"community disconnected ({len(comps)} components); path metrics on largest component only")
    if core.vcount() > 1:
        avg_path = float(core.average_path_length(directed=False, unconn=False))
        diam = int(core.diameter(directed=False, unconn=False))
    else:
        avg_path = diam = None
        notes.append("path metrics undefined for a single node")
    if density is None:
        notes.append("density undefined for <2 nodes")
    labs = pd.Series([target_of.get(g.vs[i]["orig_id"]) for i in nodes]).dropna()
    dom, share = (int(labs.value_counts().index[0]), float(labs.value_counts(normalize=True).iloc[0])) \
        if len(labs) else (None, None)
    return {"num_nodes": n, "internal_edges": m, "internal_density": density,
            "avg_clustering_coefficient": clustering, "is_connected": connected,
            "num_connected_components": len(comps), "path_metrics_scope": scope,
            "path_metrics_nodes": core.vcount(), "avg_shortest_path_length": avg_path, "diameter": diam,
            "dominant_country_target_id": dom, "dominant_country_share": share,
            "notes": "; ".join(notes)}


# ---------------------------------------------------------------- layout & plots
def get_layout(ds, log):
    """Compute node layout once (fixed seed) and cache it for all algorithms."""
    path = RESULTS_DIR / f"layout_coordinates_seed{LAYOUT_SEED}.csv"
    if path.exists():
        df = pd.read_csv(path)
        if df["node_id"].tolist() == ds.node_ids:
            return df[["x", "y"]].to_numpy()
    log.info("Computing shared layout (DrL, seed=%d)", LAYOUT_SEED)
    random.seed(LAYOUT_SEED)
    np.random.seed(LAYOUT_SEED)
    xy = np.array(ds.graph.layout_drl().coords)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"node_id": ds.node_ids, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(path, index=False)
    return xy


def _palette(k):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(k)]


def plot_community_graph(ds, membership, xy, name, outdir, n_colored=12, sample_nodes=1500):
    g = ds.graph
    sizes = pd.Series(membership).value_counts()  # ids already ranked by size
    top = list(sizes.index[:n_colored])
    pal = dict(zip(top, _palette(len(top))))
    cols = np.array([pal.get(c, (0.72, 0.72, 0.72, 1.0)) for c in membership])
    ed = np.array(g.get_edgelist())

    def draw(ax, idx, s, lw, alpha, title):
        mask = np.zeros(g.vcount(), bool)
        mask[idx] = True
        sel = ed[mask[ed[:, 0]] & mask[ed[:, 1]]]
        ax.add_collection(LineCollection(xy[sel], colors="#999999", linewidths=lw, alpha=alpha, zorder=1))
        other = [i for i in idx if membership[i] not in pal]
        if other:
            ax.scatter(xy[other, 0], xy[other, 1], s=s, c=[cols[i] for i in other], linewidths=0, zorder=2)
        for c in top:
            ii = [i for i in idx if membership[i] == c]
            if ii:
                ax.scatter(xy[ii, 0], xy[ii, 1], s=s, color=pal[c], linewidths=0, zorder=3)
        ax.set_xlim(xy[:, 0].min() - 1, xy[:, 0].max() + 1)
        ax.set_ylim(xy[:, 1].min() - 1, xy[:, 1].max() + 1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=13)
        handles = [plt.Line2D([], [], marker="o", ls="", color=pal[c], label=f"C{c} (n={sizes[c]})") for c in top]
        if len(sizes) > len(top):
            handles.append(plt.Line2D([], [], marker="o", ls="", color=(0.72, 0.72, 0.72),
                                      label=f"other ({len(sizes) - len(top)} communities)"))
        ax.legend(handles=handles, loc="upper left", fontsize=7, title="Community (by size rank)",
                  title_fontsize=8, framealpha=0.85)

    allidx = list(range(g.vcount()))
    fig, ax = plt.subplots(figsize=(11, 11))
    draw(ax, allidx, 3, 0.15, 0.12,
         f"{name.replace('_', ' ')}: {len(sizes)} communities, full network "
         f"({g.vcount()} nodes, {g.ecount()} edges)")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"{name}_community_graph.png", dpi=150)
    plt.close(fig)

    deg = np.array(g.degree())
    idx = list(np.argsort(-deg, kind="stable")[:sample_nodes])
    fig, ax = plt.subplots(figsize=(11, 11))
    draw(ax, idx, 14, 0.3, 0.25,
         f"{name.replace('_', ' ')}: induced subgraph of {sample_nodes} highest-degree nodes (same layout)")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"{name}_community_graph_sample.png", dpi=150)
    plt.close(fig)


def plot_community_sizes(sizes, name, outdir):
    sizes = sorted(sizes, reverse=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].bar(range(1, len(sizes) + 1), sizes, color="#4c78a8", width=1.0)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Community rank (1 = largest)")
    axs[0].set_ylabel("Community size (number of nodes, log scale)")
    axs[0].set_title("Community sizes by rank")
    bins = np.unique(np.logspace(0, np.log10(max(sizes)) + 0.01, 20).astype(int))
    axs[1].hist(sizes, bins=bins if len(bins) > 1 else 5, color="#f58518", edgecolor="white")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Community size (number of nodes, log scale)")
    axs[1].set_ylabel("Number of communities")
    axs[1].set_title("Size distribution")
    fig.suptitle(f"{name.replace('_', ' ')}: {len(sizes)} communities")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"{name}_community_sizes.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- result writing
def save_results(ds, name, membership, runtime, seed, params, convergence, extra, log, with_plots=True):
    """Write the standard output files for one algorithm. `membership` is aligned to ds.graph."""
    outdir = RESULTS_DIR / ALGO_DIRS[name]
    outdir.mkdir(parents=True, exist_ok=True)
    g = ds.graph
    if len(membership) != g.vcount():
        raise ValueError("membership must cover every node")
    membership = relabel_by_size(list(membership))
    sizes_s = pd.Series(membership).value_counts().sort_index()
    sizes = sizes_s.tolist()
    q = modularity(ds, membership)

    df = pd.DataFrame({"node_id": ds.node_ids, "community_id": membership})
    df["community_size"] = df["community_id"].map(sizes_s)
    df["country_target_id"] = [ds.target.get(n) for n in ds.node_ids]
    df.to_csv(outdir / f"{name}_node_communities.csv", index=False)

    rows = []
    for rank in range(min(3, len(sizes))):
        nodes = [i for i, c in enumerate(membership) if c == rank]  # id `rank` = rank-th largest
        st = community_stats(g, nodes, ds.target)
        rows.append({"rank": rank + 1, "community_id": rank, **st})
    pd.DataFrame(rows).to_csv(outdir / f"{name}_top3_communities.csv", index=False)

    metrics = {
        "algorithm": name, "dataset_file": EDGES_FILE,
        "num_nodes": g.vcount(), "num_edges": g.ecount(),
        "num_connected_components": ds.prep["connected_components"],
        "num_communities": len(sizes), "overall_modularity": q,
        "modularity_definition": "igraph unweighted undirected modularity, resolution=1, original simple graph",
        "execution_time_seconds": runtime,
        "largest_community_size": max(sizes), "smallest_community_size": min(sizes),
        "average_community_size": float(np.mean(sizes)),
        "random_seed": seed, "parameters": params, "convergence": convergence,
        "software_versions": versions(), "preprocessing": ds.prep,
        "target_label_note": "country_target_id is the dataset's integer country-derived class; "
                             "the README does not give the id->country mapping.",
        **extra,
    }
    with open(outdir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if with_plots:
        plot_community_sizes(sizes, name, outdir)
        plot_community_graph(ds, membership, get_layout(ds, log), name, outdir)
    log.info("%s: %d communities, modularity %.4f, time %.3fs; outputs in %s",
             name, len(sizes), q, runtime, outdir)
    return metrics
