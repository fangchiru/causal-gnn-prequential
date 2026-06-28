from src.baselines.CAL.causal_x_gnn import XGNNCausal
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Type


class CAL(XGNNCausal):
    """CAL: https://github.com/yongduosui/CAL"""
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
    
    # CAL Loss
    def compute_loss(self, c_logits, s_logits, combined_logits, labels):
        # Log_probs
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        combined_log_probs = F.log_softmax(combined_logits, dim=-1)

        # Loss computation
        uniform_target = torch.ones_like(s_log_probs, dtype=torch.float, device=s_log_probs.device) / self.num_classes
        c_loss = F.nll_loss(c_log_probs, labels)
        s_loss = F.kl_div(s_log_probs, uniform_target, reduction='batchmean')
        combined_loss = F.nll_loss(combined_log_probs, labels)

        total_loss = (
            self.xgnn_config['c'] * c_loss +
            self.xgnn_config['s'] * s_loss +
            self.xgnn_config['combined'] * combined_loss
        )
        return total_loss
    
    # Training step
    def training_step(self, batch, batch_idx):
        # Logits
        c_logits, s_logits, combined_logits = self(
            data=batch, 
            eval_random=self.xgnn_config['with_random']
        )
        
        # Labels
        labels = batch.y.long()

        # Loss
        total_loss = self.compute_loss(
            c_logits=c_logits,
            s_logits=s_logits, 
            combined_logits=combined_logits, 
            labels=labels
        )
        
        # Logging
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return total_loss
    
    # Validation step
    def validation_step(self, batch, batch_idx):
        # Logits
        c_logits, s_logits, combined_logits = self(
            data=batch, 
            eval_random=self.xgnn_config['eval_random']
        )
        
        # Labels
        labels = batch.y.long()

        # Loss
        total_loss = self.compute_loss(
            c_logits=c_logits, 
            s_logits=s_logits, 
            combined_logits=combined_logits, 
            labels=labels
        )
        
        # Logging
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        # Predictions
        probs = self.predict_probabilities(logits=combined_logits)

        # Classification metrics
        self.val_metrics.update(probs, labels)

        return total_loss
    