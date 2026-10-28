"""Compare saved results of Louvain, Fast Greedy and BLDLP. Never re-runs an algorithm.

Run:  python compare_results.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import common

ALGOS = ["Louvain", "Fast_Greedy", "BLDLP"]
COLORS = {"Louvain": "#4c78a8", "Fast_Greedy": "#f58518", "BLDLP": "#54a24b"}
OUT = common.RESULTS_DIR / "comparison"


def load():
    metrics, parts, top3, missing = {}, {}, {}, []
    for a in ALGOS:
        d = common.RESULTS_DIR / a
        files = [d / f"{a}_metrics.json", d / f"{a}_node_communities.csv", d / f"{a}_top3_communities.csv"]
        if not all(f.exists() for f in files):
            missing.append(a)
            continue
        metrics[a] = json.load(open(files[0], encoding="utf-8"))
        parts[a] = pd.read_csv(files[1]).set_index("node_id")["community_id"]
        top3[a] = pd.read_csv(files[2])
    return metrics, parts, top3, missing


def bar(ax_vals, ylabel, title, fname, log=False):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = list(ax_vals)
    bars = ax.bar([n.replace("_", " ") for n in names], list(ax_vals.values()),
                  color=[COLORS[n] for n in names])
    ax.bar_label(bars, fmt="%.4g", padding=2)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    metrics, parts, top3, missing = load()
    if missing:
        print(f"WARNING: no saved results for {missing}; run those algorithms first. "
              "Comparing the available ones.", file=sys.stderr)
    if len(metrics) < 2:
        sys.exit("Need results from at least two algorithms to compare.")
    OUT.mkdir(parents=True, exist_ok=True)
    algos = list(metrics)

    rows = []
    for a in algos:
        m = metrics[a]
        rows.append({"algorithm": a, "num_communities": m["num_communities"],
                     "overall_modularity": m["overall_modularity"],
                     "execution_time_seconds": m["execution_time_seconds"],
                     "largest_community_size": m["largest_community_size"],
                     "smallest_community_size": m["smallest_community_size"],
                     "average_community_size": m["average_community_size"],
                     "random_seed": m["random_seed"], "converged": m["convergence"].get("converged")})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "comparison_summary.csv", index=False)

    # partition similarity (community IDs are arbitrary labels -> only partition-level scores)
    sim = []
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            idx = parts[a].index.intersection(parts[b].index)
            sim.append({"algorithm_a": a, "algorithm_b": b,
                        "ARI": float(adjusted_rand_score(parts[a][idx], parts[b][idx])),
                        "NMI": float(normalized_mutual_info_score(parts[a][idx], parts[b][idx]))})
    t3 = pd.concat([t.assign(algorithm=a) for a, t in top3.items()])
    t3.to_csv(OUT / "comparison_top3_communities.csv", index=False)
    with open(OUT / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": rows, "partition_similarity": sim,
                   "top3": json.loads(t3.to_json(orient="records"))}, f, indent=2)

    bar(dict(zip(algos, summ.overall_modularity)), "Overall modularity (whole partition)",
        "Overall modularity", "comparison_modularity.png")
    bar(dict(zip(algos, summ.num_communities)), "Number of communities",
        "Detected communities", "comparison_community_counts.png")
    bar(dict(zip(algos, summ.execution_time_seconds)), "Execution time (s, log scale)",
        "Execution time of the algorithm core", "comparison_execution_time.png", log=True)

    # size distributions
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for a in algos:
        s = np.sort(parts[a].value_counts().to_numpy())[::-1]
        axs[0].plot(range(1, len(s) + 1), s, marker="o", ms=3, label=a.replace("_", " "), color=COLORS[a])
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("Community rank (log scale)"); axs[0].set_ylabel("Community size (nodes, log scale)")
    axs[0].set_title("Rank-size curves"); axs[0].legend()
    data = [parts[a].value_counts().to_numpy() for a in algos]
    axs[1].boxplot(data, tick_labels=[a.replace("_", " ") for a in algos])
    axs[1].set_yscale("log"); axs[1].set_ylabel("Community size (nodes, log scale)")
    axs[1].set_title("Size distribution per algorithm")
    fig.tight_layout(); fig.savefig(OUT / "comparison_community_sizes.png", dpi=150); plt.close(fig)

    # top-3 structural comparison
    props = [("num_nodes", "Nodes"), ("internal_density", "Internal density"),
             ("avg_clustering_coefficient", "Avg clustering coeff."),
             ("avg_shortest_path_length", "Avg shortest path length"), ("diameter", "Diameter")]
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.2))
    w = 0.27
    for ax, (col, title) in zip(axs, props):
        for k, a in enumerate(algos):
            v = pd.to_numeric(top3[a][col], errors="coerce").to_numpy(dtype=float)
            ax.bar(np.arange(len(v)) + (k - 1) * w, v, w, label=a.replace("_", " "), color=COLORS[a])
        ax.set_xticks(range(3)); ax.set_xticklabels(["1st", "2nd", "3rd"])
        ax.set_xlabel("Rank by size within algorithm"); ax.set_title(title)
    axs[0].legend(); fig.suptitle("Three largest communities per algorithm (community-level properties; "
                                  "not related to overall modularity)")
    fig.tight_layout(); fig.savefig(OUT / "comparison_top3_properties.png", dpi=150); plt.close(fig)

    write_report(algos, summ, sim, t3, metrics, missing)
    print(summ.to_string(index=False))
    print(pd.DataFrame(sim).to_string(index=False))
    print(f"Comparison outputs written to {OUT}")


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                     for v in r) + " |")
    return "\n".join(out)


def write_report(algos, summ, sim, t3, metrics, missing):
    L = ["# Community detection comparison: LastFM Asia", "",
         "All figures below are read from the saved result files of the actual runs.", ""]
    if missing:
        L += [f"**Not compared (no saved results): {', '.join(missing)}**", ""]
    L += ["## Summary", "", md_table(summ), ""]
    b = metrics.get("BLDLP")
    if b:
        c = b["convergence"]
        L += [f"BLDLP convergence: converged = **{c.get('converged')}** after {c.get('iterations')} iterations "
              f"(max_iter = {c.get('max_iter')}). Seed {b['random_seed']}; a single run only, so BLDLP's "
              "result is one sample of a randomised process.", ""]
    L += ["## Partition similarity", "",
          "Community IDs are arbitrary labels: ID 0 in one algorithm has no relation to ID 0 in another, "
          "so IDs are never compared. Similarity is measured on the partitions themselves with the "
          "Adjusted Rand Index (1 = identical, ~0 = chance) and Normalized Mutual Information (0-1).", "",
          md_table(pd.DataFrame(sim)), ""]
    L += ["## Three largest communities of each algorithm", "",
          "Properties are per community (induced subgraph). Where a community is disconnected, path length "
          "and diameter are computed on its largest connected component (see `path_metrics_scope`). "
          "These are not shares of the overall modularity, which is a single number for the whole partition.", "",
          md_table(t3[["algorithm", "rank", "num_nodes", "internal_edges", "internal_density",
                       "avg_clustering_coefficient", "avg_shortest_path_length", "diameter",
                       "is_connected", "path_metrics_scope"]]), "",
          "![top3](comparison_top3_properties.png)", ""]
    L += ["## Charts", "", "- comparison_modularity.png", "- comparison_community_counts.png",
          "- comparison_execution_time.png", "- comparison_community_sizes.png", "",
          "## Interpretation and trade-offs", "",
          "- **Modularity is not a universal ranking.** Louvain and Fast Greedy explicitly optimise "
          "modularity (Louvain by local moving plus aggregation, Fast Greedy by greedy pairwise merging), so "
          "a high score is expected for them. BLDLP is a label-propagation method that does not optimise "
          "modularity; a lower score does not by itself mean a worse partition for other purposes.",
          "- Modularity also has a known resolution limit, and different community counts "
          "reflect different granularity, not necessarily different quality.",
          "- Execution times measure only the algorithm core (BLDLP includes edge-weight computation); "
          "BLDLP is a pure-Python implementation while Louvain and Fast Greedy use compiled igraph code, "
          "so the timing gap partly reflects implementation, not just algorithmic complexity.",
          "- Louvain and BLDLP are randomised (seeded); Fast Greedy is deterministic. Other seeds could "
          "give different partitions; only one seed was run.",
          "- No ground-truth communities exist here. The dataset's country-derived target labels are "
          "available but were not used as ground truth in these comparisons.", ""]
    (OUT / "comparison_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
