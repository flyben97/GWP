import pandas as pd
import numpy as np
import torch
import optuna
import Data_process
import json
import Model_performance
import Result_log
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from dgllife.model import MPNNPredictor
from torch import nn
from Get_graphs import collate_molgraphs
from Get_graphs import batch_smiles_to_graph
from Early_stopping import EarlyStopping
from Device import set_seed
from Model_evaluate import train
from datetime import datetime
from Initialize_model import GATv2_initialize_model,GCN_initialize_model,NF_initialize_model,AFP_initialize_model,MPNN_initialize_model
# Read data
excel_file_path ='~/machine_learning_space/GWP/Dataset/gwp_data_combined_0714.csv' 
dataset = pd.read_csv(excel_file_path)

gwp_smi = dataset['smiles']
gwp_20 = dataset['loggwp20']

log_filename = 'MPNN Regression.log'
model_name = "MPNN Predictor"

# Define constants
num_epochs = 8000 # number of iterations
batch_size = 32  # for the dataloader module
seed_number = 0  # set a seed number
num_workers = 4  # number of workers for the dataloader
num_trial = 2
cv_fold = 10
# Featurizers
atom_bond_type = 'AttentiveFP' #'AttentiveFP' or 'Canonical' (default)

bg, atom_feat_size, bond_feat_size = batch_smiles_to_graph(gwp_smi, atom_bond_type)
X_train, X_test, graphs_train, graphs_test, y_train, y_test, train_dataloader, test_dataloader =  Data_process.split(
    gwp_smi, bg, gwp_20, test_size=0.1, num_workers=num_workers, batch_size=batch_size
)

 # Check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))  # Assumes only one GPU is available, modify as necessary

print(atom_feat_size)


# Load the best hyperparameters
with open('gwp20_best_MPNN_hyperparameters_trial.json', 'r') as f:
    best_hyperparameters = json.load(f)

train_dataloader = [(smiles, bg.to(device), labels.to(device)) for smiles, bg, labels in train_dataloader]
test_dataloader = [(smiles, bg.to(device), labels.to(device)) for smiles, bg, labels in test_dataloader]

def train(train_dataloader, num_epochs, model_name, patience=100, atom_feat_size=None, bond_feat_size=None, **params):
    
    early_stopping = EarlyStopping(patience=patience)

    if atom_feat_size is None:
        raise ValueError("atom_feat_size must be provided.")

    model_params = {key: params[key] for key in params if key not in ['learning_rate', 'weight_decay']}

    # Initialize the model based on the model_name
    if model_name == 'GATv2 Predictor':
        model = GATv2_initialize_model(atom_feat_size=atom_feat_size, **model_params)
    elif model_name == 'GCN Predictor':
        model = GCN_initialize_model(atom_feat_size=atom_feat_size, **model_params)
    elif model_name == 'NF Predictor':
        model = NF_initialize_model(atom_feat_size=atom_feat_size, **model_params)
    elif model_name == 'AFP Predictor':
        model = AFP_initialize_model(atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size, **model_params)
    elif model_name == 'MPNN Predictor':
        model = MPNN_initialize_model(atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size, **model_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_dataloader:
            smiles, bg, labels = batch
            features_node = bg.ndata['atom_feat'].to(device).float()
            labels = labels.float().to(device)

            if bond_feat_size is not None:
                features_edge = bg.edata['bond_feat'].to(device).float()
                predictions = model(bg, features_node, features_edge).squeeze().float()
            else:
                predictions = model(bg, features_node).squeeze().float()

            loss = loss_fn(predictions, labels)
            #loss = loss_fn(predictions, labels)
            #loss = loss.mean()  # 或者 loss.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)

        if (epoch + 1) % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    return model, avg_loss

# Evaluation function
def evaluate(model, dataloader, loss_fn, bond_feat_size=None):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            smiles, bg, labels = batch
            bg = bg.to(device)
            features_node = bg.ndata['atom_feat'].to(device)
            labels = labels.float().to(device)

            if bond_feat_size is not None:
                features_edge = bg.edata['bond_feat'].to(device)
                predictions = model(bg, features_node, features_edge).squeeze().float()
            else:
                predictions = model(bg, features_node).squeeze().float()

            # Calculate loss
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_predictions

#best_model, avg_loss = train(train_dataloader, num_epochs, model_name, patience=200, atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size, **best_hyperparameters)
#'''
best_model = MPNN_initialize_model(atom_feat_size=atom_feat_size,
                              bond_feat_size=bond_feat_size,
                              node_out_feats=best_hyperparameters['node_out_feats'],
                              edge_hidden_feats=best_hyperparameters['edge_hidden_feats'],
                              num_step_message_passing=best_hyperparameters['num_step_message_passing'],
                              num_step_set2set=best_hyperparameters['num_step_set2set'],
                              num_layer_set2set=best_hyperparameters['num_layer_set2set']
                              )

best_model.load_state_dict(torch.load('gwp20_best_MPNN_model_trial.pth'))
best_model.to(device)
best_model.eval()
#'''
# Define a loss function, e.g., mean squared error
loss_fn = torch.nn.MSELoss()

# Perform evaluation on the training set
train_loss, y_train, y_train_pred = evaluate(best_model, train_dataloader, loss_fn, bond_feat_size=bond_feat_size)

# Perform evaluation on the testing set
test_loss, y_test, y_test_pred = evaluate(best_model, test_dataloader, loss_fn, bond_feat_size=bond_feat_size)

# Convert to pandas DataFrame for easy export
train_results = pd.DataFrame({'y_train': y_train, 'y_train_pred': y_train_pred})
test_results = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred})

r2_train, rmse_train, mae_train, r2_test, rmse_test, mae_test = Model_performance.calculator_result2(best_model, 
                                                                                                     train_dataloader, 
                                                                                                     test_dataloader)
print(r2_test)
print(r2_train)
# Export to CSV files
train_results.to_csv('gwp20_MPNN_train_predictions.csv', index=False)
test_results.to_csv('gwp20_MPNN_test_predictions.csv', index=False)

print("Predictions saved to train_predictions.csv and test_predictions.csv")