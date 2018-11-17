
import numpy as np 

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem


class MACCSIndexer:



    def __init__(self):
        pass 

    def _index(self, mol):
        return np.array(list(map(int, MACCSkeys.GenMACCSKeys(mol))))

    def index(self, mol):
        return self._index(mol)


class AlkaneMACCSIndexer(MACCSIndexer):

    selector = np.zeros(167, dtype=bool)
    selector[np.array([
        11, 17, 19, 22, 66, 74, 96, 101, 105,
        108, 112, 114, 115, 116, 118, 128, 129, 
        141, 145, 147, 149, 155, 160, 163, 165
    ])] = True


    def __init__(self):
        pass
         

    def index(self, mol):
        return self._index(mol)[AlkaneMACCSIndexer.selector]
