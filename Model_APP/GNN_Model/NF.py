import pandas as pd
import numpy as np
import torch
import optuna
import datetime
import Data_process
import json
import Model_performance
import Result_log
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from dgllife.model import NFPredictor
from torch import nn
from Get_graphs import collate_molgraphs
from Get_graphs import batch_smiles_to_graph
from Early_stopping import EarlyStopping
from Device import set_seed
from Model_evaluate import train, evaluate
from Initialize_model import NF_initialize_model
from datetime import datetime

# Read data
excel_file_path = '~/machine_learning_space/S04/Dataset/gwp_data_combined_0714.csv'
dataset = pd.read_csv(excel_file_path)

gwp_smi = dataset['smiles']
gwp_20 = dataset['loggwp500']

log_filename = 'NF Regression.log'
model_name = "NF Predictor"

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
        return study.best_trial.value
    except ValueError:
        return np.Inf

def objective(trial):
    
    lowest_loss_cv = get_best_loss(study)

    params = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),  # similar to L2 regularization weight
    'max_degree':trial.suggest_int('max_degree', 2, 15, step=1),  # the maximum node degree to consider when updating weights
    'predictor_hidden_size':trial.suggest_int('predictor_hidden_size', 30, 200, step=1),  # size for hidden representations in the output MLP predictor
    'predictor_batchnorm':trial.suggest_categorical('predictor_batchnorm', [True, False]), # whether to apply batch normalization in the output MLP predictor
    'predictor_dropout':trial.suggest_categorical('predictor_dropout', [0, 0.2, 0.4, 0.5, 0.6])  # the dropout probability in the output MLP predictor
     }

    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=42)
    train_loss_cv, valid_loss_cv = 0.0 , 0.0
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
        torch.save(model.state_dict(), 'best_NF_model_trial.pth')
        with open('best_NF_hyperparameters_trial.json', 'w') as f:
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
with open('best_NF_hyperparameters_trial.json', 'r') as f:
    best_hyperparameters = json.load(f)

best_model = NF_initialize_model(atom_feat_size=atom_feat_size,
                                 max_degree=best_hyperparameters['max_degree'],
                                 predictor_hidden_size=best_hyperparameters['predictor_hidden_size'],
                                 predictor_batchnorm=best_hyperparameters['predictor_batchnorm'],
                                 predictor_dropout=best_hyperparameters['predictor_dropout']
                                )

best_model.load_state_dict(torch.load('best_NF_model_trial.pth'))
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

print(f'NF model training is complete')