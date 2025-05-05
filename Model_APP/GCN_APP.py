import torch
import json
import os
from dgllife.utils import AttentiveFPAtomFeaturizer
from GNN_Model.Get_graphs import batch_smiles_to_graph
from GNN_Model.Initialize_model import GCN_initialize_model

# Initialize featurizer
atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')
atom_feat_size = atom_featurizer.feat_size()

# Set device to CPU
device = torch.device('cpu')
print(f'Using device: {device}')

# Define model parameters directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_parameters_dir = os.path.join(current_dir, 'model_parameters')
json_file_path = os.path.join(model_parameters_dir, 'gwp500_best_GCN_hyperparameters_trial.json')
model_weights_path = os.path.join(model_parameters_dir, 'gwp500_best_GCN_model_trial.pth')

# Load hyperparameters
try:
    with open(json_file_path, 'r') as f:
        best_hyperparameters = json.load(f)
    print("Best hyperparameters loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {json_file_path} was not found.")
    raise
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file {json_file_path}.")
    raise

# Initialize model
best_model = GCN_initialize_model(
    atom_feat_size=atom_feat_size,
    predictor_hidden_feats=best_hyperparameters['predictor_hidden_feats'],
    predictor_dropout=best_hyperparameters['predictor_dropout'],
    num_gcn_layers=best_hyperparameters['num_gcn_layers'],
    hidden_feats_i=best_hyperparameters['hidden_feats_i'],
    gnn_norm_i=best_hyperparameters['gnn_norm_i'],
    residual_i=best_hyperparameters['residual_i'],
    batchnorm_i=best_hyperparameters['batchnorm_i'],
    dropout_i=best_hyperparameters['dropout_i']
)

# Load pre-trained model weights
best_model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
best_model.to(device)
best_model.eval()

def predict_smiles(smiles, atom_bond_type='AttentiveFP'):
    """
    Predict the GWP500 value for a given SMILES string.
    
    Args:
        smiles (str): The SMILES string to predict.
        atom_bond_type (str): The type of atom and bond features to use.
    
    Returns:
        float: The predicted value.
    """
    try:
        bg_list, atom_feat_size, _ = batch_smiles_to_graph([smiles], atom_bond_type)
        bg = bg_list[0].to(device)
        features_node = bg.ndata['atom_feat'].to(device).float()
        
        # GCN does not use edge features
        prediction = best_model(bg, features_node).squeeze().float()
        predicted_value_log10 = 10 ** prediction.item()
        return predicted_value_log10
    except Exception as e:
        raise ValueError(f"Prediction failed for SMILES {smiles}: {str(e)}")