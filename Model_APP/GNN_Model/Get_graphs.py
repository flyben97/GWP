import datetime
import os
import time
import numpy as np
import pandas as pd
import dgl
import torch
from dgllife.utils import (AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, 
                           CanonicalAtomFeaturizer, CanonicalBondFeaturizer, 
                           smiles_to_bigraph)
from dgl import save_graphs
from torch import nn

def view_bar(current, total):
    p = int(100 * current / total)
    a = "#" * (p // 2)
    b = "-" * (50 - len(a))
    print(f"\r|{a}{b}| {p:3d}% ({current}/{total})", end='', flush=True)

# Convert SMILES to graph
def smiles_to_graph(smiles,atom_bond_type):

    if atom_bond_type == 'AttentiveFP':
        atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')
        bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='bond_feat', self_loop=True)
    else:
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='atom_feat')
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='bond_feat', self_loop=True)
        atom_bond_type = 'Canonical'

    graph = smiles_to_bigraph(smiles, add_self_loop=True,
                              node_featurizer=atom_featurizer, 
                              edge_featurizer=bond_featurizer)
    
    atom_feat_size, bond_feat_size = atom_featurizer.feat_size(), bond_featurizer.feat_size()

    return graph, atom_feat_size, bond_feat_size

def batch_smiles_to_graph(smiles_list, atom_bond_type, output_file=None):
    graphs = []
    start_time = time.time()
    
    for i, smiles in enumerate(smiles_list):

        view_bar(i+1, len(smiles_list))

        graph, atom_feat_size, bond_feat_size = smiles_to_graph(smiles, atom_bond_type)
        graphs.append(graph)

    end_time = time.time()

    print(f"\nThe number of SMILES converted is {len(smiles_list)}"
          f"\nTotal time for processing batch: " 
          f"{datetime.timedelta(seconds=round(end_time - start_time, 0))}"
          )

    if output_file:
        save_graphs(output_file, graphs)
        print(f"Graphs saved to {output_file}")

    return graphs, atom_feat_size, bond_feat_size

def collate_molgraphs(data):
    smiles, graphs, labels = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    #labels = torch.stack(labels, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)  # Ensure labels are tensors
    return smiles, bg, labels


def collate_molgraphs2(data):
    smiles, graphs, labels = zip(*data)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)  # Ensure labels are tensors
    batched_graph = dgl.batch(graphs)
    return smiles, batched_graph, labels









if __name__ == '__main__':
    # Example usage
    smiles_list = ["CCO", "CC", "C1CCCCC1"]
    atom_bond_type = 'Canonical'
    output_file = ''

    batch_smiles_to_graph(smiles_list, atom_bond_type, output_file=False)