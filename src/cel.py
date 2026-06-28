from src.baselines.CAL.causal_x_gnn import XGNNCausal
import torch.nn as nn
from typing import Dict, Any, Type


class CEL(XGNNCausal):
    """
    XGNN architecture with correlational (cross-entropy) training on combined_logits.

    Uses the same causal/spurious branch split as CAL and ICL, but is trained
    with a plain cross-entropy loss on combined_logits (no causal intervention).
    This serves as an ablation: XGNN architecture without causal training objective.
    """
    def __init__(
        self,
        gnn_backbone: Type[nn.Module],
        in_channels: int,
        in_channels_e: int,
        num_classes: int,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any]
    ):
        super().__init__(
            gnn_backbone,
            in_channels,
            in_channels_e,
            num_classes,
            optimizer_hparams
        )
        self.xgnn_config.update(model_hparams)
    # Training uses XGNNCausal default: compute_loss = CEL on combined_logits
