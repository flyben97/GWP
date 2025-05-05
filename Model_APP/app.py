import torch
import json
import os
from flask import Flask, request, jsonify, render_template
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, CanonicalAtomFeaturizer
from GNN_Model.Get_graphs import batch_smiles_to_graph
from GNN_Model.Initialize_model import MPNN_initialize_model, NF_initialize_model, GCN_initialize_model

app = Flask(__name__)

# Set device to CPU
device = torch.device('cpu')
print(f'Using device: {device}')

# Initialize model parameters directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_parameters_dir = os.path.join(current_dir, 'model_parameters')

# Initialize featurizers
mpnn_atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')
mpnn_bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='bond_feat', self_loop=True)
mpnn_atom_feat_size, mpnn_bond_feat_size = mpnn_atom_featurizer.feat_size(), mpnn_bond_featurizer.feat_size()

nf_atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='atom_feat')
nf_atom_feat_size = nf_atom_featurizer.feat_size()

gcn_atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')
gcn_atom_feat_size = gcn_atom_featurizer.feat_size()

# Load MPNN (GWP20) model
mpnn_json_path = os.path.join(model_parameters_dir, 'gwp20_best_MPNN_hyperparameters_trial.json')
mpnn_weights_path = os.path.join(model_parameters_dir, 'gwp20_best_MPNN_model_trial.pth')
try:
    with open(mpnn_json_path, 'r') as f:
        mpnn_hyperparameters = json.load(f)
    mpnn_model = MPNN_initialize_model(
        atom_feat_size=mpnn_atom_feat_size,
        bond_feat_size=mpnn_bond_feat_size,
        node_out_feats=mpnn_hyperparameters['node_out_feats'],
        edge_hidden_feats=mpnn_hyperparameters['edge_hidden_feats'],
        num_step_message_passing=mpnn_hyperparameters['num_step_message_passing'],
        num_step_set2set=mpnn_hyperparameters['num_step_set2set'],
        num_layer_set2set=mpnn_hyperparameters['num_layer_set2set']
    )
    mpnn_model.load_state_dict(torch.load(mpnn_weights_path, map_location=device, weights_only=True))
    mpnn_model.to(device)
    mpnn_model.eval()
    print("MPNN model (GWP20) loaded successfully.")
except Exception as e:
    print(f"Error loading MPNN model: {e}")
    raise

# Load NF (GWP100) model
nf_json_path = os.path.join(model_parameters_dir, 'gwp100_best_NF_hyperparameters_trial.json')
nf_weights_path = os.path.join(model_parameters_dir, 'gwp100_best_NF_model_trial.pth')
try:
    with open(nf_json_path, 'r') as f:
        nf_hyperparameters = json.load(f)
    nf_model = NF_initialize_model(
        atom_feat_size=nf_atom_feat_size,
        max_degree=nf_hyperparameters['max_degree'],
        predictor_hidden_size=nf_hyperparameters['predictor_hidden_size'],
        predictor_batchnorm=nf_hyperparameters['predictor_batchnorm'],
        predictor_dropout=nf_hyperparameters['predictor_dropout']
    )
    nf_model.load_state_dict(torch.load(nf_weights_path, map_location=device, weights_only=True))
    nf_model.to(device)
    nf_model.eval()
    print("NF model (GWP100) loaded successfully.")
except Exception as e:
    print(f"Error loading NF model: {e}")
    raise

# Load GCN (GWP500) model
gcn_json_path = os.path.join(model_parameters_dir, 'gwp500_best_GCN_hyperparameters_trial.json')
gcn_weights_path = os.path.join(model_parameters_dir, 'gwp500_best_GCN_model_trial.pth')
try:
    with open(gcn_json_path, 'r') as f:
        gcn_hyperparameters = json.load(f)
    gcn_model = GCN_initialize_model(
        atom_feat_size=gcn_atom_feat_size,
        predictor_hidden_feats=gcn_hyperparameters['predictor_hidden_feats'],
        predictor_dropout=gcn_hyperparameters['predictor_dropout'],
        num_gcn_layers=gcn_hyperparameters['num_gcn_layers'],
        hidden_feats_i=gcn_hyperparameters['hidden_feats_i'],
        gnn_norm_i=gcn_hyperparameters['gnn_norm_i'],
        residual_i=gcn_hyperparameters['residual_i'],
        batchnorm_i=gcn_hyperparameters['batchnorm_i'],
        dropout_i=gcn_hyperparameters['dropout_i']
    )
    gcn_model.load_state_dict(torch.load(gcn_weights_path, map_location=device, weights_only=True))
    gcn_model.to(device)
    gcn_model.eval()
    print("GCN model (GWP500) loaded successfully.")
except Exception as e:
    print(f"Error loading GCN model: {e}")
    raise

def predict_smiles(smiles, model_type='GWP20'):
    """
    Predict the GWP value for a given SMILES string using the specified model.
    
    Args:
        smiles (str): The SMILES string to predict.
        model_type (str): The model to use ('GWP20', 'GWP100', 'GWP500').
    
    Returns:
        int: The predicted value rounded to the nearest integer.
    """
    try:
        if model_type == 'GWP20':
            model = mpnn_model
            atom_bond_type = 'AttentiveFP'
            bg_list, _, bond_feat_size = batch_smiles_to_graph([smiles], atom_bond_type)
            bg = bg_list[0].to(device)
            features_node = bg.ndata['atom_feat'].to(device).float()
            features_edge = bg.edata['bond_feat'].to(device).float()
            prediction = model(bg, features_node, features_edge).squeeze().float()
        elif model_type == 'GWP100':
            model = nf_model
            atom_bond_type = 'Canonical'
            bg_list, _, _ = batch_smiles_to_graph([smiles], atom_bond_type)
            bg = bg_list[0].to(device)
            features_node = bg.ndata['atom_feat'].to(device).float()
            prediction = model(bg, features_node).squeeze().float()
        elif model_type == 'GWP500':
            model = gcn_model
            atom_bond_type = 'AttentiveFP'
            bg_list, _, _ = batch_smiles_to_graph([smiles], atom_bond_type)
            bg = bg_list[0].to(device)
            features_node = bg.ndata['atom_feat'].to(device).float()
            prediction = model(bg, features_node).squeeze().float()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        predicted_value_log10 = round(10 ** prediction.item())
        return predicted_value_log10
    except Exception as e:
        raise ValueError(f"Prediction failed for SMILES {smiles}: {str(e)}")

@app.route('/')
def serve_index():
    return render_template('index5.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        model_type = data.get('model')  # e.g., 'GWP20', 'GWP100', 'GWP500'

        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400

        predicted_value = predict_smiles(smiles, model_type)
        
        return jsonify({
            'prediction': f"Predicted GWP ({model_type}): {predicted_value}",
            'smiles': smiles,
            'model': model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)