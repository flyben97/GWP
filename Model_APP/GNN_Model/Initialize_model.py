from dgllife.model.model_zoo.gatv2_predictor import GATv2Predictor
from dgllife.model import GCNPredictor
from dgllife.model import NFPredictor
from dgllife.model import AttentiveFPPredictor
from dgllife.model import MPNNPredictor
from torch import nn


def GATv2_initialize_model(atom_feat_size, num_gatv2_layers, hidden_feats_i, 
                     num_heads_i, feat_drops_i, attn_drops_i, alphas_i, 
                     residuals_i, share_weights_i, agg_modes_i):
    
    hidden_feats = [hidden_feats_i] * num_gatv2_layers
    num_heads = [num_heads_i] * num_gatv2_layers
    feat_drops = [feat_drops_i] * num_gatv2_layers
    attn_drops = [attn_drops_i] * num_gatv2_layers
    alphas = [alphas_i] * num_gatv2_layers
    residuals = [residuals_i] * num_gatv2_layers
    share_weights = [share_weights_i] * num_gatv2_layers
    agg_modes = [agg_modes_i] * num_gatv2_layers

    model = GATv2Predictor(in_feats=atom_feat_size,
                           hidden_feats=hidden_feats,
                           num_heads=num_heads,
                           feat_drops=feat_drops,
                           attn_drops=attn_drops,
                           alphas=alphas,
                           residuals=residuals,
                           share_weights=share_weights,
                           agg_modes=agg_modes,
                           n_tasks=1)  # Assuming n_tasks is required
    return model

def GCN_initialize_model(atom_feat_size, predictor_hidden_feats, predictor_dropout,
                     num_gcn_layers, hidden_feats_i, gnn_norm_i, residual_i,
                     batchnorm_i, dropout_i):
    
    hidden_feats = [hidden_feats_i] * num_gcn_layers
    activation = [nn.ReLU()] * num_gcn_layers
    gnn_norm = [gnn_norm_i] * num_gcn_layers
    residual = [residual_i] * num_gcn_layers
    batchnorm = [batchnorm_i] * num_gcn_layers
    dropout = [dropout_i] * num_gcn_layers

    model = GCNPredictor(in_feats=atom_feat_size,
                         predictor_hidden_feats=predictor_hidden_feats,
                         predictor_dropout=predictor_dropout,
                         gnn_norm=gnn_norm,
                         hidden_feats=hidden_feats,
                         activation=activation,
                         residual=residual,
                         batchnorm=batchnorm,
                         dropout=dropout,
                         n_tasks=1)
    return model

def NF_initialize_model(atom_feat_size, max_degree, predictor_hidden_size, predictor_batchnorm,
                     predictor_dropout):

    model = NFPredictor(in_feats=atom_feat_size, n_tasks=1, max_degree=max_degree,
                        predictor_hidden_size=predictor_hidden_size, predictor_batchnorm=predictor_batchnorm,
                        predictor_dropout=predictor_dropout)

    return model

def AFP_initialize_model(atom_feat_size, bond_feat_size, num_layers, 
                     num_timesteps, graph_feat_size, dropout):

    params = {'num_layers':num_layers,'num_timesteps':num_timesteps, 
              'graph_feat_size':graph_feat_size, 'dropout':dropout}

    model = AttentiveFPPredictor(node_feat_size=atom_feat_size, 
                                 edge_feat_size=bond_feat_size, 
                                 n_tasks=1, **params)

    return model

def MPNN_initialize_model(atom_feat_size, bond_feat_size, node_out_feats, edge_hidden_feats,
                     num_step_message_passing, num_step_set2set, num_layer_set2set):

    model = MPNNPredictor(node_in_feats=atom_feat_size, edge_in_feats=bond_feat_size, 
                          node_out_feats=node_out_feats,edge_hidden_feats=edge_hidden_feats,
                          n_tasks=1, num_step_message_passing=num_step_message_passing,
                          num_step_set2set=num_step_set2set, num_layer_set2set=num_layer_set2set)

    return model