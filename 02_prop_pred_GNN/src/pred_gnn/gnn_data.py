from typing import List, Union
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

import dgl
import dgllife
from dgllife.utils import CanonicalAtomFeaturizer, smiles_to_bigraph
from pred_gnn import utils


class GraphDataset(Dataset):
    

    def __init__(self,
                 smiles: List[str],
                 targets: List[Union[float, int]],
                 num_workers: int = 0,
                 **kwargs):
        
        
        self.smiles = np.array(smiles)
        self.targets = np.array(targets)
        self.num_workers = num_workers

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = OnehotBondFeaturizer()

    def __len__(self):
        
        return len(self.smiles)

    def __getitem__(self, idx: int):
        
        smi = self.smiles[idx]
        targ = self.targets[idx]

        graph = smiles_to_bigraph(
            smi,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )

        outdict = {"smi": smi, "graph": graph, "targ": targ}
        return outdict

    @classmethod
    def get_collate_fn(cls):
        
        return GraphDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        
        names = [j["smi"] for j in input_list]
        graphs = [j["graph"] for j in input_list]
        targs = [j["targ"] for j in input_list]

        graph_batch = dgl.batch(graphs)
        graph_batch.set_n_initializer(dgl.init.zero_initializer)
        graph_batch.set_e_initializer(dgl.init.zero_initializer)

        targs = torch.FloatTensor(targs)

        return_dict = {
            "graphs": graph_batch,
            "names": names,
            "targs": targs,
        }
        return return_dict


class MolDataset(Dataset):
    

    def __init__(self, smiles: List[str], num_workers: int = 0, **kwargs):
        

        self.smiles = np.array(smiles)
        self.num_workers = num_workers

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = OnehotBondFeaturizer()

    def __len__(self):
        
        return len(self.smiles)

    def __getitem__(self, idx: int):
        
        smi = self.smiles[idx]

        graph = smiles_to_bigraph(
            smi,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )
        if graph is None:
            return {}

        outdict = {
            "smi": smi,
            "graph": graph,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        
        graphs, names = zip(*[(i['graph'], i["smi"]) for i in input_list
                              if len(i) > 0])

        graph_batch = dgl.batch(graphs)
        graph_batch.set_n_initializer(dgl.init.zero_initializer)
        graph_batch.set_e_initializer(dgl.init.zero_initializer)

        return_dict = {
            "graphs": graph_batch,
            "names": names,
        }
        return return_dict



class OnehotBondFeaturizer(dgllife.utils.BaseBondFeaturizer):
    

    def __init__(self, bond_data_field="e", self_loop=False):
        super(OnehotBondFeaturizer, self).__init__(
            featurizer_funcs={
                bond_data_field: dgllife.utils.bond_type_one_hot
            },
            self_loop=False,
        )
