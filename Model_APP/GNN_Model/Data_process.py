import torch
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from Get_graphs import collate_molgraphs

def split(smiles, graphs, labels, test_size, num_workers, batch_size):

    X_train, X_test, graphs_train, graphs_test, y_train, y_test = train_test_split(smiles, graphs, labels, test_size=test_size, random_state=10)

    train_dataloader = DataLoader(dataset=list(zip(X_train, graphs_train, y_train)), 
                                num_workers=num_workers, 
                                batch_size=batch_size, 
                                collate_fn=collate_molgraphs)


    test_dataloader = DataLoader(dataset=list(zip(X_test, graphs_test, y_test)), 
                                num_workers=num_workers, 
                                batch_size=batch_size, 
                                collate_fn=collate_molgraphs)
    
    return X_train, X_test, graphs_train, graphs_test, y_train, y_test, train_dataloader, test_dataloader

def split_train_valid_test(smiles, graphs, labels, valid_size, test_size, num_workers, batch_size):
    # First split: train + validation and test
    X_temp, X_test, graphs_temp, graphs_test, y_temp, y_test = train_test_split(smiles, graphs, labels, test_size=test_size, random_state=42)
    
    # Second split: train and validation
    X_train, X_valid, graphs_train, graphs_valid, y_train, y_valid = train_test_split(X_temp, graphs_temp, y_temp, test_size=valid_size, random_state=42)
    
    train_dataloader = DataLoader(dataset=list(zip(X_train, graphs_train, y_train)), 
                                  num_workers=num_workers, 
                                  batch_size=batch_size, 
                                  collate_fn=collate_molgraphs)

    valid_dataloader = DataLoader(dataset=list(zip(X_valid, graphs_valid, y_valid)), 
                                  num_workers=num_workers, 
                                  batch_size=batch_size, 
                                  collate_fn=collate_molgraphs)

    test_dataloader = DataLoader(dataset=list(zip(X_test, graphs_test, y_test)), 
                                 num_workers=num_workers, 
                                 batch_size=batch_size, 
                                 collate_fn=collate_molgraphs)
    
    return X_train, X_valid, X_test, graphs_train, graphs_valid, graphs_test, y_train, y_valid, y_test, train_dataloader, valid_dataloader, test_dataloader





def split2(smiles, graphs, labels, test_size, num_workers, batch_size):
    # 初步划分数据集
    X_train, X_test, graphs_train, graphs_test, y_train, y_test = train_test_split(
        smiles, graphs, labels, test_size=test_size, random_state=42)

    # 转换为列表形式便于操作
    X_train, X_test = list(X_train), list(X_test)
    graphs_train, graphs_test = list(graphs_train), list(graphs_test)
    y_train, y_test = list(y_train), list(y_test)

    # 将 y 大于 5000 的值从测试集移动到训练集
    indices_to_move = [i for i, y in enumerate(y_test) if y > 5000]

    # 移动数据
    for i in sorted(indices_to_move, reverse=True):
        X_train.append(X_test.pop(i))
        graphs_train.append(graphs_test.pop(i))
        y_train.append(y_test.pop(i))

    # 将列表转换回 pandas DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # 创建 DataLoader
    train_dataloader = DataLoader(dataset=list(zip(X_train.values, graphs_train, y_train.values)), 
                                  num_workers=num_workers, 
                                  batch_size=batch_size, 
                                  collate_fn=collate_molgraphs)

    test_dataloader = DataLoader(dataset=list(zip(X_test.values, graphs_test, y_test.values)), 
                                 num_workers=num_workers, 
                                 batch_size=batch_size, 
                                 collate_fn=collate_molgraphs)

    # 打印 y_train 和 y_test 的大小
    #print(f'y_train size: {len(y_train)}')
    #print(f'y_test size: {len(y_test)}')

    return X_train, X_test, graphs_train, graphs_test, y_train, y_test, train_dataloader, test_dataloader

def split3(smiles, labels, test_size):
    # 初步划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        smiles, labels, test_size=test_size, random_state=42)

    # 转换为列表形式便于操作
    X_train, X_test = list(X_train), list(X_test)
    y_train, y_test = list(y_train), list(y_test)

    # 将 y 大于 5000 的值从测试集移动到训练集
    indices_to_move = [i for i, y in enumerate(y_test) if y > 5000]

    # 移动数据
    for i in sorted(indices_to_move, reverse=True):
        X_train.append(X_test.pop(i))
        y_train.append(y_test.pop(i))

    # 将列表转换回 pandas DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)


    # 打印 y_train 和 y_test 的大小
    print(f'y_train size: {len(y_train)}')
    print(f'y_test size: {len(y_test)}')

    return X_train, X_test,  y_train, y_test

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def split_with_condition(smiles, graphs, labels, test_size, num_workers, batch_size):
    # 将数据分成 y > 10000 和 其他部分
    condition_indices = [i for i, y in enumerate(labels) if y > 10000]
    remaining_indices = [i for i, y in enumerate(labels) if y <= 10000]
    
    # 分割数据
    condition_smiles = [smiles[i] for i in condition_indices]
    condition_graphs = [graphs[i] for i in condition_indices]
    condition_labels = [labels[i] for i in condition_indices]
    
    remaining_smiles = [smiles[i] for i in remaining_indices]
    remaining_graphs = [graphs[i] for i in remaining_indices]
    remaining_labels = [labels[i] for i in remaining_indices]
    
    # 对剩余数据进行常规训练/测试划分
    X_train, X_test, graphs_train, graphs_test, y_train, y_test = train_test_split(
        remaining_smiles, remaining_graphs, remaining_labels, test_size=test_size, random_state=42)
    
    # 将 y > 10000 的样本添加到训练集中
    X_train.extend(condition_smiles)
    graphs_train.extend(condition_graphs)
    y_train.extend(condition_labels)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(
        dataset=list(zip(X_train, graphs_train, y_train)), 
        num_workers=num_workers, 
        batch_size=batch_size, 
        collate_fn=collate_molgraphs
    )

    test_dataloader = DataLoader(
        dataset=list(zip(X_test, graphs_test, y_test)), 
        num_workers=num_workers, 
        batch_size=batch_size, 
        collate_fn=collate_molgraphs
    )

    # 转换为 pandas DataFrame 或 Series 以便于后续索引操作
    X_train = pd.Series(X_train)
    X_test = pd.Series(X_test)
    graphs_train = pd.Series(graphs_train)
    graphs_test = pd.Series(graphs_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    

    
    return X_train, X_test, graphs_train, graphs_test, y_train, y_test, train_dataloader, test_dataloader


