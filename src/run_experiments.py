# Imports
import os
import sys
import torch
import json
from pathlib import Path
# Data imports
# Add paths so data_scripts files are findable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent)+"/data_scripts")
from data_scripts.dataset_names import DATASET_NAMES, GET_DATASET, DATASET_N_CLASSES
# Hyperparameter imports
from src.hyper_parameters import trainer_hparams, model_class_hparams
# Backbones
# GNN backbones
from backbones.GNNs.gcn import GCN_encoder
from backbones.GNNs.gat import GAT_encoder
from backbones.GNNs.gin import GIN_encoder
# GT backbones
from backbones.GNNs.graph_GPS import GraphGPS_encoder

# Baseline and model imports (multi-class)
from baselines.GNN.gnn import GNN
from baselines.ICL.icl import ICL
from baselines.CAL.cal import CAL

# Baseline and model imports (binary)
#from baselines.GNN.gnn import GNN

# Pytorch lightning imports 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# Optuna imports
import optuna
#from optuna.integration import PyTorchLightningPruningCallback
import warnings
warnings.filterwarnings("ignore")


# Experimental setup
MODEL_CLASSES = [GNN, CAL, ICL]
BACKBONES = [GIN_encoder, GraphGPS_encoder]
SEEDS = [28, 1999, 1130, 5898, 820]
# Set up accelerator 
if torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"


# Objective function for optuna hyperparameter tuning
def objective_wrapper(model_class, gnn_backbone, input_channels, num_classes, loader, directory_name):
    def objective(trial):
        # For simplicity we go for full discrete grid
        lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3])
        wd = trial.suggest_categorical("wd", [1e-4, 5e-4, 1e-3, 5e-3])

        optimizer_hparams = {"lr": lr, "wd": wd}

        # Build model
        model = model_class(
            gnn_backbone=gnn_backbone,
            in_channels=input_channels['in_channels'],
            in_channels_e=input_channels['in_channels_e'],
            num_classes=num_classes,
            model_hparams=model_class_hparams[f"{model_class.__name__}"],
            optimizer_hparams=optimizer_hparams
        )

        # Callbacks (Early stopping not to waste compute resources + checkpoint the best performing epoch)
        early_stop_callback = EarlyStopping(
            monitor="val_auroc", #"val_auprc",
            mode="max",
            patience=trainer_hparams["patience"]
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_auroc", #"val_auprc"
            mode="max",
            save_top_k=1,
            dirpath=f"{directory_name}/hparam_tuning/checkpoints/{model_class.__name__}_{gnn_backbone.__name__}",
            filename=f"lr_{lr}_wd_{wd}"
        )

        # Save optuna logs
        log_dir = (
            f"{directory_name}/hparam_tuning/optuna_logs/"
            f"{model_class.__name__}_{gnn_backbone.__name__}"
        )

        logger = TensorBoardLogger(
            save_dir=log_dir,
            name=f"lr_{lr}_wd_{wd}"
        )

        # Trainer
        trainer = pl.Trainer(
            accelerator=accelerator, 
            devices=1,
            max_epochs=trainer_hparams['epochs'],
            gradient_clip_val=trainer_hparams['max_norm'],
            callbacks=[checkpoint_callback, early_stop_callback],  
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True
        )

        # Train
        trainer.fit(
            model=model,
            train_dataloaders=loader['train'],
            val_dataloaders=loader['val']
        )

        # Return the best AUPRC returned
        best_val_auprc = float(checkpoint_callback.best_model_score.item())
        return best_val_auprc

    return objective


# Run hyperparameter tuning
def run_hparam_tuning(model_class, gnn_backbone, input_channels, num_classes, loader, directory_name):

    # Search space
    search_space = {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3],
        "wd": [1e-4, 5e-4, 1e-3, 5e-3],
    }
    # Seed the sampler so the hyperparameter tunning is reproducible
    sampler = optuna.samplers.GridSampler(search_space, seed=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    objective = objective_wrapper(
        model_class, 
        gnn_backbone, 
        input_channels, 
        num_classes, 
        loader, 
        directory_name
    )

    study.optimize(objective)

    # Save best hyperparameters
    best_trial = study.best_trial

    save_dir = f"{directory_name}/hparam_tuning/hparams_best"
    os.makedirs(save_dir, exist_ok=True)

    best_params_path = (
        f"{save_dir}/{model_class.__name__}_{gnn_backbone.__name__}.json"
    )

    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)

    return best_trial

 
# Train model 
def train_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, directory_name, run_index, loader):
    # Getting the best hyperparameters
    with open(best_params_file_name, "r") as f:
        best_params = json.load(f)

    # Creating the model with the best hyperparameters
    model = model_class(
        gnn_backbone=gnn_backbone,
        in_channels=input_channels['in_channels'],
        in_channels_e=input_channels['in_channels_e'],
        num_classes=num_classes,
        model_hparams=model_class_hparams[f'{model_class.__name__}'],
        optimizer_hparams=best_params
    )

    # Creating trainer instance
    # Add checkpoint to save the best model
    early_stop_callback = EarlyStopping(
            monitor="val_auroc", #"val_auprc"
            mode="max",
            patience=trainer_hparams["patience"]
        )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc", #"val_auprc"
        mode="max",
        save_top_k=1,
        dirpath=directory_name + f"/run_{run_index + 1}/checkpoints",
        filename=f"{model_class.__name__}_{gnn_backbone.__name__}"
    )

    # Save the lightning logs
    logger = TensorBoardLogger(
        save_dir=f"{directory_name}/lightning_logs/{model_class.__name__}_{gnn_backbone.__name__}",
        name=f"run_{run_index + 1}" 
    ) 
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=trainer_hparams['epochs'],
        gradient_clip_val=trainer_hparams['max_norm'],
        deterministic=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )
    
    # Training the model with the best hyperparameters
    trainer.fit(model=model, train_dataloaders=loader['train'], val_dataloaders=loader['val'])


# Test model 
def test_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, checkpoint_path, metrics_file_name, loader):
    # Load best hyperparameters
    with open(best_params_file_name, "r") as f:
        best_params = json.load(f)

    # Load model from checkpoint
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        gnn_backbone=gnn_backbone,
        in_channels=input_channels['in_channels'],
        in_channels_e=input_channels['in_channels_e'],
        num_classes=num_classes,
        model_hparams=model_class_hparams[f'{model_class.__name__}'],
        optimizer_hparams=best_params
    )

    trainer = pl.Trainer(accelerator=accelerator, deterministic=True, logger=False, enable_checkpointing=False)

    # Test with loaded model
    results = trainer.test(model, dataloaders=loader["test"])
    
    # Save results into JSON file
    with open(metrics_file_name, "w") as f:
        json.dump(results, f, indent=4)


# Run the experiments
def main():
    for dataset_name in ["MNIST_75sp"]:#DATASET_NAMES:
        for model_class in MODEL_CLASSES:
            for gnn_backbone in BACKBONES:
                # Create directories to save best hyperparameters
                directory_name = f"../results/{dataset_name}"
                os.makedirs(directory_name, exist_ok=True)
                
                # Set seed
                pl.seed_everything(0, workers=True)

                # Fetch dataset
                input_channels, loader = GET_DATASET[dataset_name]
                num_classes = DATASET_N_CLASSES[dataset_name]
                
                # Run hyperparameter tunning
                run_hparam_tuning(model_class, gnn_backbone, input_channels, num_classes, loader, directory_name)
                
                # Train and test the model 5 times with the best found hyperparameter configuration
                for run_index, seed in enumerate(SEEDS):
                    # Set the seed
                    pl.seed_everything(seed, workers=True)

                    # Train
                    best_params_file_name = directory_name + f"/hparam_tuning/hparams_best/{model_class.__name__}_{gnn_backbone.__name__}.json"
                    train_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, directory_name, run_index, loader)

                    # Test
                    checkpoint_path = directory_name + f"/run_{run_index + 1}/checkpoints/{model_class.__name__}_{gnn_backbone.__name__}.ckpt"
                    metrics_file_name = directory_name + f"/run_{run_index + 1}/test_metrics/{model_class.__name__}_{gnn_backbone.__name__}.json"
                    os.makedirs(directory_name + f"/run_{run_index + 1}/test_metrics", exist_ok=True)
                    test_model(model_class, gnn_backbone, input_channels, num_classes, best_params_file_name, checkpoint_path, metrics_file_name, loader)
                    #sys.exit()
                   

if __name__ == "__main__":  
    main()