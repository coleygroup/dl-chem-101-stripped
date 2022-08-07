from typing import List, Union
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from pred_ffn import fingerprint, utils


class PredDataset(Dataset):
    

    def __init__(self,
                 smiles: List[str],
                 targets: List[Union[float, int]],
                 num_workers: int = 0,
                 **kwargs):
        
        
        self.smiles = np.array(smiles)
        self.targets = np.array(targets)
        self.num_workers = num_workers

        if self.num_workers == 0:
            self.fps = [fingerprint.get_morgan_fp_smi(i) for i in self.smiles]
        else:
            
            self.fps = utils.chunked_parallel(self.smiles,
                                              fingerprint.get_morgan_fp_smi,
                                              chunks=100,
                                              max_cpu=self.num_workers,
                                              timeout=4000,
                                              max_retries=3)

        
        self.fps = np.vstack(self.fps)

    def __len__(self):
        
        return len(self.smiles)

    def __getitem__(self, idx: int):
        
        smi = self.smiles[idx]
        fp = self.fps[idx]
        targ = self.targets[idx]
        outdict = {"smi": smi, "fp": fp, "targ": targ}
        return outdict

    @classmethod
    def get_collate_fn(cls):
        
        return PredDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        
        names = [j["smi"] for j in input_list]
        fp_ars = [j["fp"] for j in input_list]
        targs = [j["targ"] for j in input_list]

        fp_tensors = torch.stack([torch.tensor(fp) for fp in fp_ars])
        targs = torch.FloatTensor(targs)

        return_dict = {
            "fps": fp_tensors,
            "names": names,
            "targs": targs,
        }
        return return_dict


class MolDataset(Dataset):
    

    def __init__(self, smiles: List[str], num_workers: int = 0, **kwargs):
        

        self.smiles = np.array(smiles)
        self.num_workers = num_workers

        
        if self.num_workers == 0:
            self.fps = [fingerprint.get_morgan_fp_smi(i) for i in self.smiles]
        else:
            self.fps = utils.chunked_parallel(self.smiles,
                                              fingerprint.get_morgan_fp_smi,
                                              chunks=100,
                                              max_cpu=self.num_workers,
                                              timeout=4000,
                                              max_retries=3)
        self.fps, self.smiles = zip(*[(i, j)
                                      for (i, j) in zip(self.fps, self.smiles)
                                      if i is not None])

        
        self.fps = np.vstack(self.fps)

    def __len__(self):
        
        return len(self.smiles)

    def __getitem__(self, idx: int):
        
        smi = self.smiles[idx]
        fp = self.fps[idx]
        outdict = {
            "smi": smi,
            "fp": fp,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        
        names = [j["smi"] for j in input_list]
        fp_ars = [j["fp"] for j in input_list]

        
        fp_tensors = torch.stack([torch.tensor(fp) for fp in fp_ars])
        return_dict = {
            "fps": fp_tensors,
            "names": names,
        }
        return return_dict
