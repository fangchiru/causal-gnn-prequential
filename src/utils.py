# This file contains 3 core util functions:
#   1. A function to integrate the results of the experiments over the 5 runs
#   2. A function to generate the LaTex code for the results tables
#   3. A helper function to get GNN kwargs
# Imports:
import os
import json
import pandas as pd

# ---------------------------------------- Integrate results over runs ----------------------------------------

def integrate_results(base_dir, integrated_dir):
    """Auxiliary function to integrate results into a single file"""
    runs = [
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not runs:
        raise ValueError(f"No run_* directories found in {base_dir}")

    example_run_metrics_dir = os.path.join(base_dir, runs[0], "test_metrics")
    metric_files = [
        f for f in os.listdir(example_run_metrics_dir)
        if f.endswith(".json")
    ]

    os.makedirs(integrated_dir, exist_ok=True)

    for metric_file in metric_files:
        metrics_list = []

        for run in runs:
            file_path = os.path.join(base_dir, run, "test_metrics", metric_file)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected JSON structure in {file_path}: {type(data)}")

            metrics_list.append(df)

        if not metrics_list:
            continue

        combined_metrics = pd.concat(metrics_list, axis=0, ignore_index=True)

        agg_data = {}
        for col in combined_metrics.columns:
            agg_data[f"av_{col}"] = [combined_metrics[col].mean()]
            agg_data[f"stdev_{col}"] = [combined_metrics[col].std()]

        aggregated_df = pd.DataFrame(agg_data)

        output_file = os.path.join(
            integrated_dir,
            metric_file.replace(".json", "_integrated.csv")
        )
        aggregated_df.to_csv(output_file, index=False)


# ---------------------------------------- Get GNN kwargs ----------------------------------------
def get_gnn_kwargs(in_channels, in_channels_e, model_hparams, gnn_backbone_name):
    """Auxiliary function to set the GNN kwargs"""
    # GCN, GIN
    gnn_kwargs = {
        'in_channels': in_channels,
        'hidden_channels' : model_hparams["hidden_channels_gnn"], 
        'out_channels': model_hparams["gnn_out_channels"],
    }

    # GAT
    if gnn_backbone_name == "GAT_encoder":
        gnn_kwargs["in_channels_e"] = in_channels_e
    
    # GraphGPS
    elif gnn_backbone_name == "GraphGPS_encoder":
        gnn_kwargs["in_channels_e"] = in_channels_e
        gnn_kwargs["num_layers"] =  model_hparams["num_layers"]
        gnn_kwargs["rwse_dim"] = model_hparams["rwse_walk_length"]
        gnn_kwargs["pe_dim"] = model_hparams["pe_dim"]
        gnn_kwargs["attn_type"] = model_hparams["attn_type"]
        gnn_kwargs["attn_heads"] = model_hparams["attn_heads"]
        gnn_kwargs["attn_kwargs"] = model_hparams["attn_kwargs"]
    
    # GrokFormer
    elif gnn_backbone_name == 'GrokFormer_encoder':
        gnn_kwargs['in_channels_e'] = in_channels_e
        gnn_kwargs['num_layers'] = model_hparams['num_layers']
        gnn_kwargs['k'] = model_hparams['k']
        gnn_kwargs['nheads'] = model_hparams['nheads']
        gnn_kwargs['sine_dim'] = model_hparams['sine_dim']
        gnn_kwargs['tran_dropout'] = model_hparams['tran_dropout']
        gnn_kwargs['feat_dropout'] = model_hparams['feat_dropout']
        gnn_kwargs['prop_dropout'] = model_hparams['prop_dropout']
    
    # DualFormer
    elif gnn_backbone_name == 'DualFormer_encoder':
        gnn_kwargs['activation'] = model_hparams['activation']
        gnn_kwargs['num_gnns'] = model_hparams['num_gnns']
        gnn_kwargs['num_trans'] = model_hparams['num_sa']
        gnn_kwargs['num_heads'] = model_hparams['num_heads']
        gnn_kwargs['dropout_trans'] = model_hparams['dropout_sa']
        gnn_kwargs['dropout'] = model_hparams['dropout']
        gnn_kwargs['alpha'] = model_hparams['alpha']
        gnn_kwargs['lammda'] = model_hparams['lammda']
        gnn_kwargs['GraphConv'] = model_hparams['GraphConv']
        gnn_kwargs['use_bn'] = model_hparams['use_bn']
    
    return gnn_kwargs