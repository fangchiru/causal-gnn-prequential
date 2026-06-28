import math
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Type, Dict, Any, Optional
 
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
 
@dataclass
class PrequentialResult:
    """All numbers that matter from a prequential encoding run."""
 
    # Total bits to encode all labels (including the uniform prefix)
    total_bits: float
 
    # Bits per sample (total_bits / n)
    bits_per_sample: float
 
    # Compression ratio vs uniform baseline (lower is better; <1 means learning)
    compression_ratio: float
 
    # Uniform baseline: n * log2(K)  bits
    uniform_bits: float
 
    # Per-block log-losses (in nats) on the held-out block
    block_nll_nats: List[float] = field(default_factory=list)
 
    # Cumulative bits after each block (for plotting)
    cumulative_bits: List[float] = field(default_factory=list)
 
    # Number of samples seen at each checkpoint
    timesteps: List[int] = field(default_factory=list)
 
    # Test accuracy of the final fully-trained model (optional, set externally)
    final_test_accuracy: Optional[float] = None
 
    def __repr__(self):
        return (
            f"PrequentialResult(\n"
            f"  total_bits        = {self.total_bits:.1f}\n"
            f"  uniform_bits      = {self.uniform_bits:.1f}\n"
            f"  compression_ratio = {self.compression_ratio:.4f}\n"
            f"  bits_per_sample   = {self.bits_per_sample:.4f}\n"
            f"  final_test_acc    = {self.final_test_accuracy}\n"
            f")"
        )
 
 
# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------
 
class PrequentialEncoder:
    """
    Implements Algorithm 2 (Appendix of Blier & Ollivier 2018) for
    PyTorch Geometric graph datasets and XGNN-style models.
 
    Parameters
    ----------
    model_class : Type
        The XGNN model class to instantiate at each step (e.g. CAL).
    model_kwargs : dict
        Keyword arguments passed to model_class(...).
    num_classes : int
        K — the number of output classes.
    timesteps : List[int]
        Sorted list of sample indices [t_0, t_1, ..., t_S] where
          t_0 = 0  (or a small seed, often skipped to 1st real timestep)
          t_S = len(dataset)
        The first t_1 samples are encoded uniformly; each subsequent block
        [t_s, t_{s+1}) is encoded using the model trained on [0, t_s).
    max_epochs : int
        How many epochs to train the model at each timestep.
    batch_size : int
        Batch size for DataLoader.
    device : str
        "cuda" or "cpu".
    retrain_from_scratch : bool
        If True (default), reset model weights at every timestep.
        If False, continue fine-tuning from the previous checkpoint
        (faster but slightly less faithful to the paper).
    dataloader_kwargs : dict
        Extra kwargs forwarded to DataLoader (e.g. num_workers).
    """
 
    def __init__(
        self,
        model_class: Type,
        model_kwargs: Dict[str, Any],
        num_classes: int,
        timesteps: List[int],
        max_epochs: int = 200,
        batch_size: int = 32,
        device: str = "cuda",
        retrain_from_scratch: bool = True,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        patience: int = 20,
        min_delta: float = 1e-4,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.num_classes = num_classes
        self.timesteps = sorted(timesteps)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.retrain_from_scratch = retrain_from_scratch
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.patience = patience
        self.min_delta = min_delta
 
        # Validate timestep list
        assert len(self.timesteps) >= 2, "Need at least [t_0, t_1]"
        assert self.timesteps[0] >= 0, "First timestep must be >= 0"
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def encode(self, dataset) -> PrequentialResult:
        """
        Run the full prequential encoding over `dataset`.
 
        Parameters
        ----------
        dataset : list-like of PyG Data objects
            The full graph dataset. Graphs are assumed to be in a fixed
            order; shuffling should be done *before* calling this method
            (use the same random seed for reproducibility, as required
            by the paper so Alice and Bob agree on the same ordering).
 
        Returns
        -------
        PrequentialResult
        """
        n = len(dataset)
        K = self.num_classes
        timesteps = self.timesteps
 
        # Validate that the last timestep covers the full dataset
        if timesteps[-1] != n:
            logger.warning(
                f"Last timestep {timesteps[-1]} != dataset size {n}. "
                "Appending n to timesteps."
            )
            timesteps = timesteps + [n]
 
        uniform_bits = n * math.log2(K)
 
        # ------------------------------------------------------------------
        # Block 0: uniform encoding
        # Cost = t_1 * log2(K)  (the first block is free of model cost)
        # ------------------------------------------------------------------
        t1 = timesteps[1]  # end of the first uniformly-encoded block
        uniform_block_bits = t1 * math.log2(K)
 
        total_bits = uniform_block_bits
        cumulative_bits = [uniform_block_bits]
        block_nll_nats = []
 
        logger.info(
            f"Block 0 (uniform): indices [0, {t1}), cost = {uniform_block_bits:.2f} bits"
        )
 
        # ------------------------------------------------------------------
        # Blocks 1 … S-1: train on [0, t_s), evaluate on [t_s, t_{s+1})
        # ------------------------------------------------------------------
        model = None  # will be created/reset at each step
 
        for s in range(1, len(timesteps) - 1):
            t_train_end = timesteps[s]       # train on [0, t_train_end)
            t_eval_start = timesteps[s]      # evaluate on [t_eval_start, t_eval_end)
            t_eval_end = timesteps[s + 1]
 
            logger.info(
                f"\n--- Block {s} ---\n"
                f"  Train on indices [0, {t_train_end})\n"
                f"  Evaluate on indices [{t_eval_start}, {t_eval_end})"
            )
 
            # 1. (Re)train the model on data [0, t_train_end)
            model = self._train(dataset, end_idx=t_train_end, prev_model=model)
 
            # 2. Evaluate the model on the held-out block [t_eval_start, t_eval_end)
            nll_nats = self._evaluate_nll(dataset, model, t_eval_start, t_eval_end)
            block_nll_nats.append(nll_nats)
 
            # Convert nats → bits  (1 nat = log2(e) bits ≈ 1.4427 bits)
            block_bits = nll_nats * math.log2(math.e)
            total_bits += block_bits
            cumulative_bits.append(total_bits)
 
            logger.info(
                f"  NLL = {nll_nats:.4f} nats | block cost = {block_bits:.2f} bits | "
                f"cumulative = {total_bits:.2f} bits | "
                f"ratio = {total_bits / uniform_bits:.4f}"
            )
 
        result = PrequentialResult(
            total_bits=total_bits,
            bits_per_sample=total_bits / n,
            compression_ratio=total_bits / uniform_bits,
            uniform_bits=uniform_bits,
            block_nll_nats=block_nll_nats,
            cumulative_bits=cumulative_bits,
            timesteps=timesteps[1:],  # one entry per block boundary
        )
 
        logger.info(f"\nFinal result:\n{result}")
        return result
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _build_model(self) -> torch.nn.Module:
        """Instantiate a fresh model and move it to the target device."""
        model = self.model_class(**self.model_kwargs)
        model.to(self.device)
        return model
 
    def _make_loader(self, dataset, start_idx: int, end_idx: int, shuffle: bool = True):
        """Return a DataLoader over dataset[start_idx:end_idx]."""
        subset = Subset(dataset, list(range(start_idx, end_idx)))
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **self.dataloader_kwargs,
        )
 
    def _train(self, dataset, end_idx: int, prev_model) -> torch.nn.Module:
        """
        Train a model on dataset[0:end_idx].
 
        If retrain_from_scratch is True, a new model is built every call.
        Otherwise, the previous model's weights are used as a starting point.
        """
        if self.retrain_from_scratch or prev_model is None:
            model = self._build_model()
        else:
            # Continue from previous weights (faster, slightly less faithful)
            model = copy.deepcopy(prev_model)
 
        model.train()
 
        # Build optimizer — use the model's own configure_optimizers if available
        # (Lightning-style models define this method), otherwise fall back to Adam.
        if hasattr(model, "configure_optimizers"):
            optimizer_config = model.configure_optimizers()
            # configure_optimizers may return an optimizer or a dict/list
            if isinstance(optimizer_config, (list, tuple)):
                optimizer = optimizer_config[0]
            elif isinstance(optimizer_config, dict):
                optimizer = optimizer_config["optimizer"]
            else:
                optimizer = optimizer_config
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
 
        loader = self._make_loader(dataset, start_idx=0, end_idx=end_idx, shuffle=True)

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = self._compute_training_loss(model, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"  [train] epoch {epoch+1}/{self.max_epochs}  "
                    f"loss = {avg_loss:.4f}"
                )

            # Early stopping: stop if loss hasn't improved by min_delta for `patience` epochs
            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        f"  [early stopping] epoch {epoch+1}  "
                        f"best_loss = {best_loss:.4f}"
                    )
                    break

        return model
 
    def _forward(self, model, batch):
        """
        Call the model forward pass and return (logits, is_probs).

        Handles three model types:
          - CAL  : forward(data, eval_random) → (c_logits, s_logits, combined_logits)
                   use combined_logits (index 2), which are raw logits.
          - ICL  : forward(data) → (logits_main, logits_c, logits_s)
                   use logits_main (index 0), which are raw logits.
          - GNN  : forward(data) → probabilities (Softmax applied inside model)
                   returns probs; flag is_probs=True so callers use log+NLL instead.
        """
        model_name = type(model).__name__

        if model_name in ("CAL", "CEL"):
            output = model(data=batch, eval_random=False)
            return output[2], False          # combined_logits, is_probs=False

        elif model_name == "ICL":
            output = model(batch)
            return output[0], False          # logits_main, is_probs=False

        else:                                # GNN (and any future plain classifier)
            output = model(batch)
            output = output[0] if isinstance(output, tuple) else output
            return output, False             # logits, is_probs=False

    def _compute_training_loss(self, model, batch) -> torch.Tensor:
        """Compute the training loss for a batch, handling all model types."""
        labels = batch.y.long()
        model_name = type(model).__name__

        if model_name == "CAL":
            # Replicate CAL's own training loss: c + KL(s, uniform) + combined
            c_logits, s_logits, combined_logits = model(data=batch, eval_random=False)
            c_log_probs       = F.log_softmax(c_logits,       dim=-1)
            s_log_probs       = F.log_softmax(s_logits,       dim=-1)
            combined_log_probs = F.log_softmax(combined_logits, dim=-1)
            uniform_target    = torch.ones_like(s_log_probs) / self.num_classes
            cfg   = getattr(model, "xgnn_config", {})
            w_c   = cfg.get("c",        1.0)
            w_s   = cfg.get("s",        1.0)
            w_com = cfg.get("combined", 1.0)
            return (w_c   * F.nll_loss(c_log_probs,       labels)
                  + w_s   * F.kl_div(s_log_probs, uniform_target, reduction="batchmean")
                  + w_com * F.nll_loss(combined_log_probs, labels))

        elif model_name == "CEL":
            # XGNN architecture with plain cross-entropy on combined_logits
            _, _, combined_logits = model(data=batch, eval_random=False)
            return F.cross_entropy(combined_logits, labels)

        elif model_name == "ICL":
            # Replicate ICL's training loss: main + causal + KL(spurious, uniform)
            logits_main, logits_c, logits_s = model(batch)
            uniform_target = torch.ones_like(logits_s) / self.num_classes
            loss_main = F.cross_entropy(logits_main, labels)
            loss_c    = F.cross_entropy(logits_c,    labels)
            loss_s    = F.kl_div(F.log_softmax(logits_s, dim=-1), uniform_target, reduction="batchmean")
            return loss_main + loss_c + loss_s

        else:
            # GNN: output is logits (no Softmax in supervisor's MLP)
            logits = model(batch)
            logits = logits[0] if isinstance(logits, tuple) else logits
            return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def _evaluate_nll(
        self, dataset, model: torch.nn.Module, start_idx: int, end_idx: int
    ) -> float:
        """
        Compute the total negative log-likelihood (in nats) of the model
        on dataset[start_idx:end_idx].

        Returns
        -------
        float
            Sum of -log p(yᵢ | xᵢ, θ) over the block, in nats.
        """
        model.eval()
        loader = self._make_loader(dataset, start_idx, end_idx, shuffle=False)
        total_nll = 0.0

        for batch in loader:
            batch  = batch.to(self.device)
            logits, is_probs = self._forward(model, batch)
            labels = batch.y.long()

            if is_probs:
                # GNN output: probabilities → use NLL loss
                nll = F.nll_loss(torch.log(logits + 1e-10), labels, reduction="sum")
            else:
                # CAL / ICL output: logits → use cross entropy
                nll = F.cross_entropy(logits, labels, reduction="sum")

            total_nll += nll.item()

        return total_nll
 
 
# ---------------------------------------------------------------------------
# Helper: build default exponential timesteps (as used in the paper)
# ---------------------------------------------------------------------------
 
def exponential_timesteps(n: int, base: int = 2, start: int = 8) -> List[int]:
    """
    Generate doubling timesteps [start, 2*start, 4*start, …, n].
 
    These are the timesteps used in the paper for MNIST and CIFAR.
    For graph datasets with n samples you can use this directly or
    pass your own list to PrequentialEncoder.
 
    Example
    -------
        exponential_timesteps(1024, start=8)
        → [8, 16, 32, 64, 128, 256, 512, 1024]
    """
    steps = []
    t = start
    while t < n:
        steps.append(t)
        t *= base
    steps.append(n)
    return steps
 
 
# ---------------------------------------------------------------------------
# Convenience: compute uniform baseline (bits)
# ---------------------------------------------------------------------------
 
def uniform_codelength(n: int, num_classes: int) -> float:
    """n * log2(K) bits — the cost of encoding labels with no model."""
    return n * math.log2(num_classes)
 