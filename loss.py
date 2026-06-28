import torch
import torch.nn as nn

def correlational_loss(model, batch):
    """
    Calculates standard Binary Cross Entropy loss.
    """
    # 1. Forward pass
    # The model returns probabilities (0-1) because of the Sigmoid in the MLP
    y_pred = model(batch).squeeze(-1) 
    
    # 2. Prepare targets
    y_true = batch.y.float()
    
    # 3. Calculate Loss
    criterion = nn.BCELoss()
    return criterion(y_pred, y_true)