import json
import torch
import torch.nn.functional as F
from utils import load_model, init_featurizer
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np


def load(args):
    with open('configure.json', 'r') as f:
        args.update(json.load(f))

    args['device'] = torch.device('cpu')

    args = init_featurizer(args)
    model = load_model(args).to(args['device'])
    checkpoint = torch.load('gwp100_MPNN_9010.pkl', map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model


def get_graph(args, smiles):    
    bg = smiles_to_bigraph(smiles, add_self_loop=True, node_featurizer=args['node_featurizer'], edge_featurizer=args['edge_featurizer'])
    return bg


def visualization(graph, R, name, smi):
    nodes = []
    atom_weights = []
    save_weight = []
    for idx_atom in range(graph.nodes().shape[0]):
        weight = float(sum(F.sigmoid(R[idx_atom])))
        atom_weights.append(weight)
        nodes.append((idx_atom, {"weight": weight}))

        save_weight.append([idx_atom, weight])

    atom_weights = np.array(atom_weights)
    min_value = min(atom_weights)
    max_value = max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = matplotlib.colormaps.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(nodes))}

    mol = Chem.MolFromSmiles(smi)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(6)
    op = drawer.drawOptions()

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(len(nodes)),
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open(str(name)+'.svg', 'w') as f:
        f.write(svg)


if __name__ == "__main__":
    args = {}
    smi='FC1(F)OC(F)(F)C(F)(F)C1(F)F'
    model=load(args)
    bg=get_graph(args, smi)
    nt=bg.ndata['h']
    et=bg.edata['e']
    gt=model.gnn(bg, nt, et)
    visualization(bg, gt, 'test2', smi)
