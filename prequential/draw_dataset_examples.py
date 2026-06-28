"""
draw_dataset_examples.py
========================
Draws one representative example graph from each of the four datasets
(MUTAG, PROTEINS, IMDB-MULTI, SPMotif) in a 2x2 panel, to illustrate the
structural contrasts described in the Data chapter.

Output: ../paper/dataset_examples.png  (and .pdf)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
OUT = ROOT / "results" / "dataset_examples"

# colour per dataset (matches a clean academic palette)
COLORS = {
    "MUTAG":      "#2E86AB",
    "PROTEINS":   "#27500A",
    "IMDB-MULTI": "#085041",
    "SPMotif":    "#712B13",
}


class AddOnes(T.BaseTransform):
    def forward(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        return data


def tu(name, transform=None):
    return TUDataset(root=str(DATA_ROOT / "TUDataset"), name=name,
                     use_node_attr=True, use_edge_attr=True,
                     **({"transform": transform} if transform else {}))


def spmotif(bias=0.5):
    return list(SPMotif(root=str(DATA_ROOT / f"SPMotif_{bias}"), mode="train"))


def pick_representative(graphs, target_nodes):
    """Pick a graph whose node count is closest to target (a 'typical' example)."""
    best, bestdiff = None, 1e9
    for g in graphs[:500]:
        n = g.num_nodes
        if n < 4:
            continue
        diff = abs(n - target_nodes)
        if diff < bestdiff:
            bestdiff, best = diff, g
    return best


def draw(ax, data, title, color, stats):
    G = to_networkx(data, to_undirected=True)
    # spring layout with a fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=7, k=0.6)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", width=0.7, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color,
                           node_size=32, linewidths=0.3, edgecolors="white")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=6)
    ax.margins(0.08)
    ax.axis("off")


def main():
    # representative graphs near each dataset's average node count
    g_mutag = pick_representative(list(tu("MUTAG")), 18)
    g_prot  = pick_representative(list(tu("PROTEINS")), 39)
    g_imdb  = pick_representative(list(tu("IMDB-MULTI", AddOnes())), 13)
    g_spm   = pick_representative(spmotif(0.5), 18)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    draw(axes[0], g_mutag, "MUTAG", COLORS["MUTAG"], "")
    draw(axes[1], g_prot, "PROTEINS", COLORS["PROTEINS"], "")
    draw(axes[2], g_imdb, "IMDB-MULTI", COLORS["IMDB-MULTI"], "")
    draw(axes[3], g_spm, "SPMotif", COLORS["SPMotif"], "")

    plt.tight_layout(pad=1.5)
    fig.savefig(str(OUT) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    print(f"Saved: {OUT}.png and {OUT}.pdf")
    print(f"Node counts -> MUTAG {g_mutag.num_nodes}, PROTEINS {g_prot.num_nodes}, "
          f"IMDB {g_imdb.num_nodes}, SPMotif {g_spm.num_nodes}")


if __name__ == "__main__":
    main()
