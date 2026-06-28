"""
run_prequential.py
==================
Runs prequential encoding over all 5 datasets × 3 models × 2 backbones × 5 seeds.

Results are saved to:
    ../results/{dataset}/prequential_{dataset}_{model}_{backbone}.json

Each JSON contains:
    compression_ratio_mean, compression_ratio_std, total_bits_mean, total_bits_std,
    uniform_bits, per_seed dict.
"""

import sys
import os
import json
import math
import logging
import random
import numpy as np
from pathlib import Path

import torch
import torch_geometric.transforms as T

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_scripts"))

# ── model imports ─────────────────────────────────────────────────────────────
from src.baselines.GNN.gnn import GNN
from src.baselines.CAL.cal import CAL
from src.baselines.CAL.cel import CEL
from src.baselines.ICL.icl import ICL

# ── backbone imports ──────────────────────────────────────────────────────────
from src.backbones.GNNs.gin import GIN_encoder
from src.backbones.GNNs.graph_GPS import GraphGPS_encoder

# ── hyper-parameters ──────────────────────────────────────────────────────────
from src.hyper_parameters import (
    model_class_hparams,
    GraphGPS_encoder_hparams,
)

# ── prequential encoder ───────────────────────────────────────────────────────
from prequential_encoder import PrequentialEncoder, exponential_timesteps, uniform_codelength

# ── dataset loaders ───────────────────────────────────────────────────────────
from torch_geometric.datasets import TUDataset
from data_preparation.spmotif_dataset import SPMotif

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Config
# ==============================================================================

DATA_ROOT = ROOT / "data"

SEEDS = [28, 1999, 1130]  # 3 seeds

MODEL_CLASSES = [GNN, CAL, CEL, ICL]

BACKBONES = {
    "GIN_encoder":      GIN_encoder,
    "GraphGPS_encoder": GraphGPS_encoder,
}

# Optimizer defaults (used for all models/datasets in prequential encoding)
OPTIMIZER_HPARAMS = {"lr": 1e-3, "wd": 1e-4}

# ==============================================================================
# Pre-transform (adds RWSE for GraphGPS, also used for GIN — ignored if absent)
# ==============================================================================
pre_transform = T.AddRandomWalkPE(
    walk_length=GraphGPS_encoder_hparams["rwse_walk_length"],
    attr_name="RWSE",
)


# ==============================================================================
# Dataset loading helpers
# ==============================================================================

class AddOnesFeatures(T.BaseTransform):
    """Adds a column of 1s as node features for datasets without node attributes."""
    def forward(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        return data


def load_tu_dataset(name: str):
    """Load a TUDataset and return (all_graphs, in_channels, in_channels_e, num_classes)."""
    needs_ones = name in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]
    transform = AddOnesFeatures() if needs_ones else None

    ds = TUDataset(
        root=str(DATA_ROOT / "TUDataset"),
        name=name,
        use_node_attr=True,
        use_edge_attr=True,
        pre_transform=pre_transform,
        **({"transform": transform} if transform else {}),
    )
    g0 = ds[0]
    in_channels   = g0.x.shape[1]
    in_channels_e = g0.edge_attr.shape[1] if getattr(g0, "edge_attr", None) is not None else None

    num_classes = {"MUTAG": 2, "IMDB-MULTI": 3}.get(name, len(set(int(g.y) for g in ds)))
    return list(ds), in_channels, in_channels_e, num_classes


SPMOTIF_SANITY_SIZE = 2000  # subsample for local sanity check; set to None for full run on server

def load_spmotif(bias: float):
    """Combine train/val/test splits of SPMotif and return (all_graphs, in_channels, in_channels_e, num_classes)."""
    root = str(DATA_ROOT / f"SPMotif_{bias}")
    splits = [
        SPMotif(root=root, mode=mode, pre_transform=pre_transform)
        for mode in ["train", "val", "test"]
    ]
    all_graphs = []
    for split in splits:
        all_graphs.extend(list(split))

    # Subsample for local sanity check (full dataset = 12,000 graphs, too slow on CPU)
    if SPMOTIF_SANITY_SIZE is not None:
        all_graphs = all_graphs[:SPMOTIF_SANITY_SIZE]

    g0 = all_graphs[0]
    in_channels   = g0.x.shape[1]                                           # 4
    in_channels_e = g0.edge_attr.shape[1] if getattr(g0, "edge_attr", None) is not None else None  # 1
    return all_graphs, in_channels, in_channels_e, 3


DATASET_LOADERS = {
    # "MUTAG":       lambda: load_tu_dataset("MUTAG"),
    # "IMDB-MULTI":  lambda: load_tu_dataset("IMDB-MULTI"),
    # "SPMotif_0.5": lambda: load_spmotif(0.5),
    # "SPMotif_0.7": lambda: load_spmotif(0.7),
    "SPMotif_0.9": lambda: load_spmotif(0.9),
    # "PROTEINS":    lambda: load_tu_dataset("PROTEINS"),
}


# ==============================================================================
# Build model kwargs per (model_class, backbone, dataset)
# ==============================================================================

def build_model_kwargs(model_class, backbone_cls, in_channels, in_channels_e, num_classes):
    """Return the kwargs dict to pass to PrequentialEncoder(model_kwargs=...)."""
    hparams = model_class_hparams[model_class.__name__]
    return dict(
        gnn_backbone=backbone_cls,
        in_channels=in_channels,
        in_channels_e=in_channels_e,
        num_classes=num_classes,
        model_hparams=hparams,
        optimizer_hparams=OPTIMIZER_HPARAMS,
    )


# ==============================================================================
# Main experiment loop
# ==============================================================================

def main():
    for dataset_name, load_fn in DATASET_LOADERS.items():
        print(f"\n{'#'*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'#'*70}")

        # Load dataset once (pre_transform result is cached on disk by PyG)
        all_graphs, in_channels, in_channels_e, num_classes = load_fn()
        print(f"  Loaded {len(all_graphs)} graphs  |  "
              f"in_channels={in_channels}  in_channels_e={in_channels_e}  "
              f"num_classes={num_classes}")

        results_dir = ROOT / "results" / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)

        for model_class in MODEL_CLASSES:
            for backbone_name, backbone_cls in BACKBONES.items():
                tag = f"{dataset_name}_{model_class.__name__}_{backbone_name}"
                out_path = results_dir / f"prequential_{model_class.__name__}_{backbone_name}.json"

                print(f"\n{'='*70}")
                print(f"  {tag}")
                print(f"{'='*70}")

                model_kwargs = build_model_kwargs(
                    model_class, backbone_cls, in_channels, in_channels_e, num_classes
                )

                seed_results = []

                for seed in SEEDS:
                    print(f"\n  -- seed {seed} --")
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    # Shuffle dataset independently per seed
                    shuffled = list(all_graphs)
                    random.shuffle(shuffled)
                    n = len(shuffled)
                    timesteps = exponential_timesteps(n, base=2, start=16)

                    print(f"  n={n}  timesteps={timesteps}")
                    print(f"  uniform baseline: {uniform_codelength(n, num_classes):.1f} bits")

                    encoder = PrequentialEncoder(
                        model_class=model_class,
                        model_kwargs=model_kwargs,
                        num_classes=num_classes,
                        timesteps=timesteps,
                        max_epochs=100,
                        batch_size=32,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        retrain_from_scratch=True,
                        patience=20,
                        min_delta=1e-4,
                    )

                    result = encoder.encode(shuffled)
                    seed_results.append(result)

                    print(f"  compression_ratio = {result.compression_ratio:.4f}  "
                          f"total_bits = {result.total_bits:.1f}")

                # ── Aggregate across seeds ──────────────────────────────────
                ratios = [r.compression_ratio for r in seed_results]
                bits   = [r.total_bits        for r in seed_results]

                summary = {
                    "dataset":                  dataset_name,
                    "model":                    model_class.__name__,
                    "backbone":                 backbone_name,
                    "compression_ratio_mean":   float(np.mean(ratios)),
                    "compression_ratio_std":    float(np.std(ratios)),
                    "total_bits_mean":          float(np.mean(bits)),
                    "total_bits_std":           float(np.std(bits)),
                    "uniform_bits":             float(seed_results[0].uniform_bits),
                    "num_seeds":                len(SEEDS),
                    "per_seed": {
                        str(s): {
                            "compression_ratio": r.compression_ratio,
                            "total_bits":        r.total_bits,
                            "bits_per_sample":   r.bits_per_sample,
                            "block_nll_nats":    r.block_nll_nats,
                            "cumulative_bits":   r.cumulative_bits,
                            "timesteps":         r.timesteps,
                        }
                        for s, r in zip(SEEDS, seed_results)
                    },
                }

                with open(out_path, "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"\n  [{tag}]")
                print(f"  compression_ratio = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
                print(f"  total_bits        = {np.mean(bits):.1f} ± {np.std(bits):.1f}")
                print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
