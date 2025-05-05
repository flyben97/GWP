import datetime
import json
import numpy as np
import pandas as pd
import torch
import optuna
import Result_log
import Model_performance
import Data_process
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from Get_graphs import collate_molgraphs,batch_smiles_to_graph
from datetime import datetime
from Device import set_seed
from Model_evaluate import train
from Initialize_model import GATv2_initialize_model

log_filename = 'GATv2 Regression.log'
model_name = "GATv2 Predictor"

# Read data
#excel_file_path = '~/machine_learning_space/GWP/Dataset/gas-gwp.csv'
excel_file_path ='~/machine_learning_space/S04/Dataset/gwp_data_combined_0714.csv' 
dataset = pd.read_csv(excel_file_path)

gwp_smi = dataset['smiles']
gwp = dataset['loggwp500']

seed_number = 0  # set a seed number
num_epochs = 8000  # number of iterations
batch_size = 128  # for the dataloader module
num_workers = 4  # number of workers for the dataloader
atom_bond_type = 'AttentiveFP' #'AttentiveFP' or 'Canonical' (default)
num_trial = 5
cv_fold = 5


bg, atom_feat_size, bond_feat_size = batch_smiles_to_graph(gwp_smi, atom_bond_type)
X_train, X_test, graphs_train, graphs_test, y_train, y_test, train_dataloader, test_dataloader = Data_process.split(gwp_smi, bg, gwp,
                                                                                                                    test_size=0.1, 
                                                                                                                    num_workers=num_workers, 
                                                                                                                    batch_size=batch_size)

print(atom_feat_size)

# Check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))  # Assumes only one GPU is available, modify as necessary

set_seed(seed=seed_number)

def get_best_loss(study):
    try:
        return study.best_trial.value
    except ValueError:
        return np.Inf

def objective(trial):
    
    lowest_loss_cv = get_best_loss(study)

    params = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    'num_gatv2_layers': trial.suggest_int('num_gatv2_layers', 2, 8, step=1),
    'hidden_feats_i': trial.suggest_categorical('hidden_feats_i', [32, 64, 128, 256]),
    'num_heads_i': trial.suggest_categorical('num_heads_i', [4, 6, 8]),
    'feat_drops_i': trial.suggest_categorical('feat_drops_i', [0, 0.2, 0.4, 0.5, 0.6]),
    'attn_drops_i': trial.suggest_categorical('attn_drops_i', [0, 0.2, 0.4, 0.5, 0.6]),
    'alphas_i': trial.suggest_categorical('alphas_i', [0.05, 0.1, 0.2, 0.3]),
    'residuals_i': trial.suggest_categorical('residuals_i', [True, False]),
    'share_weights_i': trial.suggest_categorical('share_weights_i', [True, False]),
    'agg_modes_i': trial.suggest_categorical('agg_modes_i', ['mean', 'flatten']),
     }
        
    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=42)


    train_loss_cv, valid_loss_cv = 0, 0
    for train_index, valid_index in kf.split(X_train):       
        X_train_cv, X_valid_cv = X_train.iloc[train_index].values, X_train.iloc[valid_index].values
        graphs_train_cv = [graphs_train[i] for i in train_index]
        graphs_valid_cv = [graphs_train[i] for i in valid_index]
        y_train_cv = y_train.iloc[train_index].values
        y_valid_cv = y_train.iloc[valid_index].values

        y_train_cv = [torch.tensor(label, dtype=torch.float32) for label in y_train_cv]
        y_valid_cv = [torch.tensor(label, dtype=torch.float32) for label in y_valid_cv]

        train_dataset_cv = list(zip(X_train_cv, graphs_train_cv, y_train_cv))
        valid_dataset_cv = list(zip(X_valid_cv, graphs_valid_cv, y_valid_cv))

        train_dataloader_cv = DataLoader(
            dataset=train_dataset_cv, 
            num_workers=num_workers, 
            batch_size=batch_size, 
            collate_fn=collate_molgraphs
        )
        valid_dataloader_cv = DataLoader(
            dataset=valid_dataset_cv, 
            num_workers=num_workers, 
            batch_size=batch_size, 
            collate_fn=collate_molgraphs
        )
        
        train_dataloader_cv = [(smiles, bg.to(device), labels.to(device)) for smiles, bg, labels in train_dataloader_cv]
        valid_dataloader_cv = [(smiles, bg.to(device), labels.to(device)) for smiles, bg, labels in valid_dataloader_cv]
        
        model, train_loss, valid_loss = train(train_dataloader_cv, valid_dataloader_cv, num_epochs, model_name, patience=200, atom_feat_size=atom_feat_size, bond_feat_size=None, **params)

        train_loss_cv += train_loss        
        valid_loss_cv += valid_loss

    train_loss_cv /= cv_fold
    valid_loss_cv /= cv_fold

    if valid_loss_cv < lowest_loss_cv:
        lowest_loss_cv = valid_loss_cv
        print(f'Train loss:', train_loss_cv,'Valid loss:', valid_loss_cv)
        torch.save(model.state_dict(), 'best_GATv2_model_trial.pth')
        with open('best_GATv2_hyperparameters_trial.json', 'w') as f:
            json.dump(trial.params, f)

    return valid_loss_cv

train_dataloader = [(smiles, bg.to(device), labels.to(device)) for smiles, bg, labels in train_dataloader]
test_dataloader = [(smiles, bg.to(device), labels.to(device)) for smiles, bg, labels in test_dataloader]

training_start_time = datetime.now()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=num_trial)

training_end_time = datetime.now()
training_time = (training_end_time - training_start_time).total_seconds()


best_trial = study.best_trial

# Load the best hyperparameters
with open('best_GATv2_hyperparameters_trial.json', 'r') as f:
    best_hyperparameters = json.load(f)

best_model = GATv2_initialize_model(atom_feat_size=atom_feat_size,
                              num_gatv2_layers=best_hyperparameters['num_gatv2_layers'],
                              hidden_feats_i=best_hyperparameters['hidden_feats_i'],
                              num_heads_i=best_hyperparameters['num_heads_i'],
                              feat_drops_i=best_hyperparameters['feat_drops_i'],
                              attn_drops_i=best_hyperparameters['attn_drops_i'],
                              alphas_i=best_hyperparameters['alphas_i'],
                              residuals_i=best_hyperparameters['residuals_i'],
                              share_weights_i=best_hyperparameters['share_weights_i'],
                              agg_modes_i=best_hyperparameters['agg_modes_i'])

best_model.load_state_dict(torch.load('best_GATv2_model_trial.pth'))
best_model.to(device)
best_model.eval()

r2_train, rmse_train, mae_train, r2_test, rmse_test, mae_test = Model_performance.calculator_result(best_model, 
                                                                                                   train_dataloader, 
                                                                                                   test_dataloader) 

Result_log.setup_logger(log_filename)
Result_log.log_results_with_time_and_date(model_name, best_trial, 
                                          y_train, r2_train, rmse_train, mae_train, 
                                          y_test, r2_test, rmse_test, mae_test, 
                                          training_start_time, training_end_time, training_time)

print(f'GATv2 model training is complete')


