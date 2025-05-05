import torch
import torch.optim as optim
from torch import nn
import numpy as np
from Early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from Get_graphs import collate_molgraphs
from Initialize_model import GATv2_initialize_model,GCN_initialize_model,NF_initialize_model,AFP_initialize_model,MPNN_initialize_model
from datetime import datetime
from sklearn.metrics import r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_dataloader, valid_dataloader_cv, num_epochs, model_name, patience=100, atom_feat_size=None, bond_feat_size=None, **params):
    
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        valid_loss = evaluate(model, valid_dataloader_cv, loss_fn, bond_feat_size=bond_feat_size)

        if (epoch + 1) % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {valid_loss:.4f}")

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    return model, avg_loss, valid_loss

def evaluate(model, dataloader, loss_fn, bond_feat_size=None):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            smiles, bg, labels = batch
            bg = bg.to(device)  # 将图移动到设备
            features_node = bg.ndata['atom_feat'].to(device)
            labels = labels.float().to(device)

            if bond_feat_size is not None:
                features_edge = bg.edata['bond_feat'].to(device)
                predictions = model(bg, features_node, features_edge).squeeze().float()
            else:
                predictions = model(bg, features_node).squeeze().float()

             # 计算损失
            loss = loss_fn(predictions, labels)
            
            # 对损失进行归约，确保是标量
            #loss = loss.mean()  # 或者 loss.sum()
            total_loss += loss.item()

            #all_predictions.extend(predictions.cpu().numpy())
            #all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    #r2 = r2_score(all_labels, all_predictions)
    return avg_loss

def cross_validate(X_train, y_train, graphs_train, kf, num_epochs, model_name, atom_feat_size=None, bond_feat_size=None, num_workers=4, batch_size=128, **params):
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
        
        model, train_loss, valid_loss = train(
            train_dataloader_cv, valid_dataloader_cv, num_epochs, model_name, patience=200, 
            atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size, **params
        )
        
        train_loss_cv += train_loss
        valid_loss_cv += valid_loss
    
    avg_train_loss_cv = train_loss_cv / kf.get_n_splits()
    avg_valid_loss_cv = valid_loss_cv / kf.get_n_splits()
    
    return avg_train_loss_cv, avg_valid_loss_cv, model

