
import csv
from builtins import ZeroDivisionError
from warnings import filterwarnings
from PIL import Image
import rdkit
from rdkit import RDLogger, Chem
from rdkit.Chem import Draw
import torch
import tqdm


def suppress_warnings(level : str="minor") -> None:
    
    if level == "minor":
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
        filterwarnings(action="ignore", category=UserWarning)
        filterwarnings(action="ignore", category=FutureWarning)
    elif level == "all":
        
        filterwarnings(action="ignore")
    else:
        raise ValueError(f"Not a valid `level`. Use 'minor' or 'all', not '{level}'.")

@staticmethod
def get_device() -> str:
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def draw_smiles(path : str, smiles_list : list) -> float:
    
    mols          = []
    smiles_legend = []
    n_valid       = 0
    for smiles in smiles_list:
        try:
            mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=True)
            rdkit.Chem.Kekulize(mol)
            if mol is not None:
                mols.append(mol)
                smiles_legend.append(smiles)
                n_valid += 1
        except:
            
            pass

    
    try:
        fraction_valid = n_valid/len(smiles_list)
    except ZeroDivisionError:
        fraction_valid = 0.0

    try:
        
        image = rdkit.Chem.Draw.MolsToGridImage(mols,
                                                molsPerRow=4,
                                                subImgSize=(300, 300),
                                                legends=smiles_legend,
                                                highlightAtomLists=None)
        image.save(path, format='png')
    except:
        
        image = Image.new('RGB', (300, 300))
        image.save(path, format='png')

    return fraction_valid

def progress_bar(iterable : iter, total : int, **kwargs) -> tqdm.tqdm:
    
    return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

def save_smiles(smiles : list, output_filename : str) -> None:
    
    with open(output_filename, "w", encoding="utf-8") as output_file:

        
        write = csv.writer(output_file, delimiter="\n")  
        write.writerow(smiles)
