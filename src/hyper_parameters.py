from torch.nn import ReLU
from torch_geometric.nn import global_add_pool 

# Trainer hyperparameters
trainer_hparams = {
    'patience' : 64, # Patience
    'patience_hparam_tun': 32, # Patience during h param tunning
    'batch_size' : { # Batch Size
        "MUTAG": 16,
        "PROTEINS": 16,
        "IMDB-BINARY": 16,
        "IMDB-MULTI": 16,
        "NCI1": 64,
        "COLLAB": 64,
        "SPMotif_b_05": 128,
        "SPMotif_b_07": 128,
        "SPMotif_b_09": 128,
        "Graph_SST2": 128,
        "Molhiv": 128,
        "MNIST_75sp": 128,
        "MNIST_SUPERPIXELS": 128,
        "SYN_binary": 128,
        "SYN_multi": 64,
    },
    'epochs_hparam_tun': 100, # Maximum number of epochs during h param tunning
    'epochs' : 200, # Maximum number of epochs
    'max_norm' : 1, # Maximum norm for gradient clipping
    'train_frac': 0.60, # Train split
    'val_frac' : 0.20, # Validation split
    'test_frac' : 0.20 # Test split
}

# Model hyperparameters
DIM_EMBEDDING_SPACE = 32
HIDDEN_CHANNELS_MLP = 16
HIDDEN_CHANNELS_GNN = 16
N_SAMPLES = 16
DROPOUT = 0.1
N_HEADS = 4
N_LAYERS = 2
CAT_OR_ADD = 'cat'
POOLING = global_add_pool # We use add_pool as default following related work (CAL, ICL)

# Hyper-parameters of the graph encoders
# MPNNs (GCN, GAT, GIN)
MPNN_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'global_node_att': False,
    'global_edge_att': False
}
# GTs
# GraphGPS
GraphGPS_encoder_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'rwse_walk_length' : 16, # Random walk length for the SE
    'pe_dim' : 8, # PE dim
    'num_layers': N_LAYERS, # Number of GPS layers
    'attn_type' : 'performer', # Linear Global attention mechanisim
    'attn_heads': N_HEADS, # Number of attention heads
    'attn_kwargs' : {'dropout': DROPOUT},
    'global_node_att': True,
    'global_edge_att': False
}

# GrokFormer
GrokFormer_encoder_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'num_layers': N_LAYERS, # Number of GrokFormer layers
    'k': 8, # Number of eigenvalues/eigenvectors to keep from the Laplacian decomposition
    'nheads': N_HEADS, # Number of attention heads
    'sine_dim': 16, # Eigenvalue encoding dim
    'tran_dropout': DROPOUT,
    'feat_dropout': DROPOUT,
    'prop_dropout': DROPOUT,
    'global_node_att': True,
    'global_edge_att': False
}

# DualFormer
DualFormer_encoder_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'activation': ReLU(),
    'num_gnns': N_LAYERS, 
    'num_sa': 1,
    'num_heads': N_HEADS,
    'dropout': DROPOUT,
    'dropout_sa': DROPOUT,
    'alpha': 0.1,
    'lammda': 0.1,
    'GraphConv': 'sgc',
    'use_bn': True,
    'global_node_att': True,
    'global_edge_att': False
}

# Graph Neural Network encoders hyperparameters        
GNN_encoders_hparams = {
    'GCN_encoder': MPNN_hparams,
    'GAT_encoder': MPNN_hparams,
    'GIN_encoder': MPNN_hparams,
    'GraphGPS_encoder': GraphGPS_encoder_hparams,
    'GrokFormer_encoder': GrokFormer_encoder_hparams,
    'DualFormer_encoder': DualFormer_encoder_hparams
}
# ------------------------------------ GNNs -----------------------------------
GNN_hparams = {
    'hidden_channels_mlp': HIDDEN_CHANNELS_MLP, # Hidden dimension of MLPs
}

# ----------------------------------- XGNNs -----------------------------------
# ------------ Non-Causal XGNNs -----------------------------------
# SUNNY-GNN
SUNNY_SUPPORTED_BACKBONES = ["GCN_encoder", "GAT_encoder"]
SUNNYGNN_hparams = {
    'max_topk': 0.5,
    'min_topk': 0.1,
    'n_pos': N_SAMPLES,
    'n_neg': N_SAMPLES,
    'k': 0.1,
    'temp': 1,           
    'tau': 0.1,           
    'cts_coef': 0.01,    
    'dropout': DROPOUT,
    'sparsity_mask_coef': 1e-4,
    'sparsity_ent_coef': 1e-2, 
}

# ------------ Causal XGNNs -----------------------------------
Causal_XGNN_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'cat_or_add': CAT_OR_ADD,
    'with_random': True,
    'eval_random': False,
    'dropout': DROPOUT,
    'hidden_channels_mlp': HIDDEN_CHANNELS_MLP, # Hidden dimension of MLPs
}
# CAL
CAL_hparams = {
    's': 0.5,
    'c': 1.0,
    'combined': 0.5,
    'layers': N_LAYERS
}
# ICL
ICL_hparams = {
    'mgda_model': 'loss+',  # gradient normalization: 'loss+', 'loss', 'l2', 'none'
}


# ----------------------------------- Models -----------------------------------
# The code here is: model_class_hparams[f'{model_class.__name__}'],
model_class_hparams = {
    "GNN": GNN_hparams, # Vanilla GNN
    # Causal XGNNs:
    "CAL": CAL_hparams,
    "ICL": ICL_hparams,
    "CEL": {},          # XGNN architecture + plain cross-entropy on combined_logits (ablation)
    # Non Causal XGNNs:
    "SUNNY_GNN": SUNNYGNN_hparams,
}


