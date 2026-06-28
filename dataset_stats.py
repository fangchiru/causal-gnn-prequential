"""
dataset_stats.py
================
Computes per-dataset graph statistics to support the Chapter 4 interpretation:
average #nodes, #edges, density, degree, diameter/radius, and node/edge feature dims.

These statistics support claims such as "PROTEINS is local and correlation-driven"
(small, dense graphs) vs. SPMotif being synthetic and simple.
"""
import sys
from pathlib import Path
import numpy as np
import networkx as nx

import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_scripts"))

from data_preparation.spmotif_dataset import SPMotif

DATA_ROOT = ROOT / "data"

# Sample a subset for the expensive diameter computation (full pass for the rest)
DIAMETER_SAMPLE = 200


class AddOnesFeatures(T.BaseTransform):
    def forward(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        return data


def load_tu(name):
    needs_ones = name in ["IMDB-MULTI"]
    transform = AddOnesFeatures() if needs_ones else None
    ds = TUDataset(
        root=str(DATA_ROOT / "TUDataset"),
        name=name,
        use_node_attr=True,
        use_edge_attr=True,
        **({"transform": transform} if transform else {}),
    )
    return list(ds)


def load_spmotif(bias):
    root = str(DATA_ROOT / f"SPMotif_{bias}")
    graphs = []
    for mode in ["train", "val", "test"]:
        graphs.extend(list(SPMotif(root=root, mode=mode)))
    return graphs[:2000]  # match the subsample used in experiments


def graph_stats(graphs, name):
    n_nodes = np.array([g.num_nodes for g in graphs])
    # edge_index counts each undirected edge twice in PyG
    n_edges = np.array([g.num_edges // 2 for g in graphs])

    # density = 2E / (N(N-1))
    densities = []
    avg_degrees = []
    for nn, ee in zip(n_nodes, n_edges):
        if nn > 1:
            densities.append(2 * ee / (nn * (nn - 1)))
            avg_degrees.append(2 * ee / nn)
    densities = np.array(densities)
    avg_degrees = np.array(avg_degrees)

    # diameter on a sample (largest connected component)
    diameters = []
    sample = graphs[:DIAMETER_SAMPLE]
    for g in sample:
        G = to_networkx(g, to_undirected=True)
        if G.number_of_nodes() == 0:
            continue
        if nx.is_connected(G):
            diameters.append(nx.diameter(G))
        else:
            comps = (G.subgraph(c) for c in nx.connected_components(G))
            largest = max(comps, key=lambda c: c.number_of_nodes())
            if largest.number_of_nodes() > 1:
                diameters.append(nx.diameter(largest))
    diameters = np.array(diameters) if diameters else np.array([0])

    g0 = graphs[0]
    node_dim = g0.x.shape[1] if getattr(g0, "x", None) is not None else 0
    edge_dim = g0.edge_attr.shape[1] if getattr(g0, "edge_attr", None) is not None else 0

    print(f"\n{'='*60}\n  {name}   ({len(graphs)} graphs)\n{'='*60}")
    print(f"  Nodes / graph     : {n_nodes.mean():6.1f}  (min {n_nodes.min()}, max {n_nodes.max()})")
    print(f"  Edges / graph     : {n_edges.mean():6.1f}  (min {n_edges.min()}, max {n_edges.max()})")
    print(f"  Avg node degree   : {avg_degrees.mean():6.2f}")
    print(f"  Density           : {densities.mean():6.3f}")
    print(f"  Diameter (n={len(diameters)} sample): {diameters.mean():6.2f}  (max {diameters.max()})")
    print(f"  Node feat dim     : {node_dim}")
    print(f"  Edge feat dim     : {edge_dim}")

    return {
        "name": name,
        "graphs": len(graphs),
        "nodes_mean": n_nodes.mean(),
        "edges_mean": n_edges.mean(),
        "degree_mean": avg_degrees.mean(),
        "density_mean": densities.mean(),
        "diameter_mean": diameters.mean(),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
    }


def main():
    results = []
    results.append(graph_stats(load_tu("MUTAG"), "MUTAG"))
    results.append(graph_stats(load_tu("PROTEINS"), "PROTEINS"))
    results.append(graph_stats(load_tu("IMDB-MULTI"), "IMDB-MULTI"))
    results.append(graph_stats(load_spmotif(0.5), "SPMotif (b=0.5)"))

    # LaTeX-ready summary
    print(f"\n\n{'#'*60}\n  LaTeX table rows\n{'#'*60}")
    print(r"Dataset & Graphs & Nodes & Edges & Degree & Density & Diam. \\")
    for r in results:
        print(f"{r['name']} & {r['graphs']} & {r['nodes_mean']:.1f} & "
              f"{r['edges_mean']:.1f} & {r['degree_mean']:.2f} & "
              f"{r['density_mean']:.3f} & {r['diameter_mean']:.1f} \\\\")


if __name__ == "__main__":
    main()
