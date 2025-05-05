import datetime
import numpy as np
import pandas as pd
import torch
import optuna
import Data_process
import json
import Model_performance
import Result_log
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from dgllife.model import AttentiveFPPredictor
from torch import nn
from Get_graphs import collate_molgraphs
from Get_graphs import batch_smiles_to_graph
from Early_stopping import EarlyStopping
from Device import set_seed
from Model_evaluate import train
from datetime import datetime
from Initialize_model import AFP_initialize_model

# Read data
excel_file_path ='~/machine_learning_space/S04/Dataset/gwp_data_combined_0714.csv' 
dataset = pd.read_csv(excel_file_path)

gwp_smi = dataset['smiles']
gwp_20 = dataset['loggwp500']

log_filename = 'AFP Regression.log'
model_name = "AFP Predictor"

# Define constants
num_epochs = 8000 # number of iterations
batch_size = 128  # for the dataloader module
seed_number = 0  # set a seed number
num_workers = 4  # number of workers for the dataloader
num_trial = 5
cv_fold = 5
# Featurizers
atom_bond_type = 'AttentiveFp' #'AttentiveFP' or 'Canonical' (default)

bg, atom_feat_size, bond_feat_size = batch_smiles_to_graph(gwp_smi, atom_bond_type)
X_train, X_test, graphs_train, graphs_test, y_train, y_test, train_dataloader, test_dataloader = Data_process.split(gwp_smi, bg, gwp_20,
                                                                                                                    test_size=0.1, 
                                                                                                                    num_workers=num_workers, 
                                                                                                                    batch_size=batch_size)

 # Check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Print the GPU model
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))  # Assumes only one GPU is available, modify as necessary

set_seed(seed=seed_number)

def get_best_loss(study):
    try:
        # 如果有完成的试验，返回最优的验证损失
        return study.best_trial.value
    except ValueError:
        # 如果没有完成的试验，返回无穷大
        return np.Inf

def objective(trial):
    
    lowest_loss_cv = get_best_loss(study)

    params = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),  # similar to L2 regularization weight
    'num_layers': trial.suggest_int('num_layers', 1, 6, step=1),  # number of GNN layers for atom embedding (Attentive FP radius)
    'num_timesteps': trial.suggest_int('num_timesteps', 1, 6, step=1),  # times of updating the molecular graph representations with GRU
    'graph_feat_size': trial.suggest_int('graph_feat_size', 30, 300, step=1),  # size for the learned graph representations (Attentive FP dimension)
    'dropout': trial.suggest_categorical('dropout', [0, 0.2, 0.4, 0.5, 0.6]), # probability for performing the dropout
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
        
        model, train_loss, valid_loss = train(train_dataloader_cv, valid_dataloader_cv, num_epochs, model_name, patience=200, atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size, **params)

        train_loss_cv += train_loss        
        valid_loss_cv += valid_loss

    train_loss_cv /= cv_fold
    valid_loss_cv /= cv_fold

    if valid_loss_cv < lowest_loss_cv:
        lowest_loss_cv = valid_loss_cv
        print(f'Train loss:', train_loss_cv,'Valid loss:', valid_loss_cv)
        torch.save(model.state_dict(), 'best_AFP_model_trial.pth')
        with open('best_AFP_hyperparameters_trial.json', 'w') as f:
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
with open('best_AFP_hyperparameters_trial.json', 'r') as f:
    best_hyperparameters = json.load(f)

best_model = AFP_initialize_model(atom_feat_size=atom_feat_size,
                                  bond_feat_size=bond_feat_size,
                                  num_layers=best_hyperparameters['num_layers'],
                                  num_timesteps=best_hyperparameters['num_timesteps'],
                                  graph_feat_size=best_hyperparameters['graph_feat_size'],
                                  dropout=best_hyperparameters['dropout']
                                 )

best_model.load_state_dict(torch.load('best_AFP_model_trial.pth'))
best_model.to(device)
best_model.eval()

r2_train, rmse_train, mae_train, r2_test, rmse_test, mae_test = Model_performance.calculator_result2(best_model, 
                                                                                                     train_dataloader, 
                                                                                                     test_dataloader)


Result_log.setup_logger(log_filename)
Result_log.log_results_with_time_and_date(model_name, best_trial, 
                                          y_train, r2_train, rmse_train, mae_train, 
                                          y_test, r2_test, rmse_test, mae_test, 
                                          training_start_time, training_end_time, training_time)

print(f'AFP model training is complete')



