# Comparative Community Detection on the LastFM Asia Network

Compares **Louvain**, **Fast Greedy** and **BLDLP** on the SNAP LastFM Asia social network.
All plots use Matplotlib.

## Dataset
Source: https://snap.stanford.edu/data/feather-lastfm-social.html (files in `data/lasftm_asia/`, unmodified; note the folder name's spelling is from the download).

| File | Format |
|---|---|
| `lastfm_asia_edges.csv` | header `node_1,node_2`, comma separated, one undirected mutual-follower edge per row |
| `lastfm_asia_target.csv` | header `id,target`; integer class (0-17) derived from the user's country. The README does not give the id-to-country mapping, so it is reported as `country_target_id` only |
| `lastfm_asia_features.json` | liked-artist lists per node (not used) |

Verified: 7,624 nodes, 27,806 edges (matches the SNAP description), no self-loops, no duplicate edges, no missing values, 1 connected component, undirected. Policy (in `common.py`): undirected simple graph; self-loops/duplicates would be dropped (none found); no nodes removed.

## Structure
```
data/  common.py  louvain.py  fast_greedy.py  bldlp.py  compare_results.py  main.py
requirements.txt  README.md  tests/test_bldlp_weights.py
results/{Louvain,Fast_Greedy,BLDLP,comparison}/   results/layout_coordinates_seed42.csv
```

## Install
```
pip install -r requirements.txt
```

## Run
```
python louvain.py [--seed 42]
python fast_greedy.py
python bldlp.py [--seed 42] [--alpha 15] [--max-iter 10000]
python main.py                      # everything: load graph once, run 3 algorithms, compare
python main.py --algorithms louvain bldlp --skip-existing
python compare_results.py           # comparison only, from saved results (reruns nothing)
python tests/test_bldlp_weights.py  # BLDLP weight check against the notebook's literal code
```

## Algorithms
* **Louvain** (igraph `community_multilevel`, resolution 1, seed 42): greedily moves nodes to the neighbouring community with best modularity gain, then collapses communities into super-nodes and repeats. Number of communities is automatic.
* **Fast Greedy** (igraph `community_fastgreedy`): starts with singletons, repeatedly merges the pair with the largest modularity gain; the dendrogram is cut at the maximum-modularity level. Deterministic.
* **BLDLP**: label propagation with link-density edge weights used for tie-breaking. See below.

### BLDLP reference and implementation
The only accessible source is the notebook `sna_partb.ipynb` (section 2.3) in https://github.com/SophiaVei/Community-Detection-in-Social-Networks. Neither the repository nor the notebook cites a paper for BLDLP, so **no paper equations could be checked**; the implementation reproduces the notebook:
* Edge weight for edge (v,u): `floor(alpha / (|E_vu| + 1))`, alpha = 15, where `E_vu` = adjacency tuples `(n,x)` of neighbours n of v (excluding u), with x != u. It is computed from v's side only (v = smaller node id here), as in the notebook. `bldlp.py` uses an equivalent closed form, verified identical to the notebook loop on a random graph (`tests/`).
* Labels start unique; nodes are updated asynchronously in a fresh random order each iteration; majority neighbour label wins; ties are broken by summed edge weight, then randomly.
* The notebook's stopping rule: stop when an iteration leaves no tied nodes, else at `max_iter` (10000, the notebook value).
* Not Standard LP: the weight tie-break is the BLDLP part. Because random tie-breaks can recur indefinitely, convergence is not guaranteed.

## Outputs
Per algorithm `results/<Name>/` (`<Name>` = `Louvain`, `Fast_Greedy`, `BLDLP`):
* `<Name>_community_graph.png`: full network, shared layout (DrL, seed 42, cached in `results/layout_coordinates_seed42.csv`); the 12 largest communities are coloured, the rest grey.
* `<Name>_community_graph_sample.png` (extra): induced subgraph of the 1,500 highest-degree nodes on the same layout (hub-biased, documented; all metrics use the full graph).
* `<Name>_community_sizes.png`: size by rank, and size histogram.
* `<Name>_metrics.json`: algorithm, dataset, nodes, edges, components, communities, modularity, time, min/max/mean size, seed, parameters, convergence, library versions, preprocessing.
* `<Name>_node_communities.csv`: `node_id, community_id, community_size, country_target_id`. Community IDs are arbitrary, assigned 0..k-1 by descending size.
* `<Name>_top3_communities.csv`: rank, community_id, num_nodes, internal_edges, internal_density, avg_clustering_coefficient, is_connected, num_connected_components, path_metrics_scope, path_metrics_nodes, avg_shortest_path_length, diameter, dominant_country_target_id/share, notes.
* `<Name>_run_log.txt`.

`results/comparison/`: `comparison_summary.csv/.json`, `comparison_modularity.png`, `comparison_community_counts.png`, `comparison_execution_time.png`, `comparison_community_sizes.png`, `comparison_report.md`, plus `comparison_top3_communities.csv` and `comparison_top3_properties.png`. The report includes ARI/NMI between partitions; community IDs are never compared across algorithms.

## Metric definitions
* **Modularity**: fraction of edges inside communities minus that expected under a random graph with the same degrees (igraph, unweighted, undirected, resolution 1, same graph for all algorithms). It is a single score for a whole partition, not for individual communities.
* **Community count / size**: number of groups and nodes per group.
* **Density**: internal edges / possible edges, `2m / (n(n-1))`.
* **Average clustering coefficient**: mean over nodes of local triangle closure; nodes with degree < 2 count as 0.
* **Average shortest path length / diameter**: mean and maximum hop distance on the induced community subgraph. Defined only for connected graphs; for a disconnected community they are computed on its largest connected component and flagged in `path_metrics_scope`/`notes`. (All current top-3 communities are connected.)

## Limitations
* BLDLP lacks a citable paper (see above); `balance` has no definition beyond the notebook's weighting. In the actual run it did not converge in 10,000 iterations.
* Louvain and BLDLP are randomised; one seed (42) was run, so results are single samples.
* BLDLP is pure Python, so its timing is not comparable with compiled igraph code.
* Execution time excludes data loading and plotting.
* No ground-truth communities; country labels are not used for scoring.
