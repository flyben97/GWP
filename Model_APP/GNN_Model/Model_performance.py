import torch
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metrics(model, dataloader):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            smiles, bg, labels = batch
            features = bg.ndata['atom_feat'].to(device)
            labels = labels.float().to(device)

            predictions = model(bg, features).squeeze()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    r2 = r2_score(all_labels, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
    mae = mean_absolute_error(all_labels, all_predictions)

    return r2, rmse, mae

def calculator_result(model, train_dataloader, test_dataloader):

    r2_train, rmse_train, mae_train = calculate_metrics(model, train_dataloader)
    r2_test, rmse_test, mae_test = calculate_metrics(model, test_dataloader)

    return r2_train, rmse_train, mae_train,r2_test, rmse_test, mae_test

def calculate_metrics2(model, dataloader):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            smiles, bg, labels = batch
            features_node = bg.ndata['atom_feat'].to(device)
            features_edge = bg.edata['bond_feat'].to(device)
            labels = labels.float().to(device)

            predictions = model(bg, features_node, features_edge).squeeze()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    r2 = r2_score(all_labels, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
    mae = mean_absolute_error(all_labels, all_predictions)

    return r2, rmse, mae

def calculator_result2(model, train_dataloader, test_dataloader):

    r2_train, rmse_train, mae_train = calculate_metrics2(model, train_dataloader)
    r2_test, rmse_test, mae_test = calculate_metrics2(model, test_dataloader)

    return r2_train, rmse_train, mae_train,r2_test, rmse_test, mae_test