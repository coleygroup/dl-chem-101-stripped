
import pandas as pd
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier


def molecules(path : str) -> rdkit.Chem.rdmolfiles.SmilesMolSupplier:
    
    
    with open(path, "r", encoding="utf-8") as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    
    molecule_set = SmilesMolSupplier(path,
                                     sanitize=True,
                                     nameColumn=-1,
                                     titleLine=has_header)
    return molecule_set

def smiles(path : str) -> list:
    
    smiles_list = pd.read_csv(path, delimiter=" ")["SMILES"].to_list()
    return smiles_list
