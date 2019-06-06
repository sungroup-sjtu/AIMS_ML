import sys
import pathlib
import numpy as np
from pathlib import Path

from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect


class ECFP4Indexer:
    name = 'ecfp4'

    def __init__(self):
        self.n_bits = 1024

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        return np.array(
            list(map(int, Chem.GetMorganFingerprintAsBitVect(rdk_mol, radius=2, nBits=self.n_bits))))

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]


class MorganCountIndexer:
    name = 'morgan'

    def __init__(self):
        self.svg_dir: Path = None
        self.radius = 2
        self.fp_time_limit = 40

    def index(self, smiles):
        raise Exception('Use index_list() for this indexer')

    def index_list(self, smiles_list):
        fp_list = []
        identifiers = set()
        fpsvg_dict = {}
        fpsmi_dict = {}
        index_list = []

        print('Calculate with RDKit...')
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\t%i' % i)

            rdk_mol = Chem.MolFromSmiles(smiles)
            info = dict()
            fp: UIntSparseIntVect = Chem.GetMorganFingerprint(rdk_mol, radius=self.radius, bitInfo=info)
            fp_list.append(fp)
            identifiers.update(fp.GetNonzeroElements().keys())

            if self.svg_dir is not None:
                for idx in info.keys():
                    if idx not in fpsvg_dict:
                        fpsvg_dict[idx] = Draw.DrawMorganBit(rdk_mol, idx, info)

                        root, radius = info[idx][0]
                        if radius == 0:
                            id_bonds = []
                            id_atoms = [root]
                        else:
                            id_bonds = Chem.FindAtomEnvironmentOfRadiusN(rdk_mol, radius,
                                                                         root)  # args: mol, radius, atomId
                            id_atoms = set()
                            for bid in id_bonds:
                                id_atoms.add(rdk_mol.GetBondWithIdx(bid).GetBeginAtomIdx())
                                id_atoms.add(rdk_mol.GetBondWithIdx(bid).GetEndAtomIdx())
                            id_atoms = list(id_atoms)
                        smi = Chem.MolFragmentToSmiles(rdk_mol, atomsToUse=id_atoms, rootedAtAtom=root,
                                                       isomericSmiles=False)
                        fpsmi_dict[idx] = smi

        print('\nFilter identifiers...')
        print('%i identifiers total' % len(identifiers))
        identifiers = list(sorted(identifiers))
        identifier_times = dict([(id, 0) for id in identifiers])
        for fp in fp_list:
            for idx in fp.GetNonzeroElements().keys():
                identifier_times[idx] += 1
        identifiers = [idx for idx, times in identifier_times.items() if times >= self.fp_time_limit]
        print('%i identifiers saved' % len(identifiers))

        for fp in fp_list:
            index = [fp.GetNonzeroElements().get(idx, 0) for idx in identifiers]
            index_list.append(np.array(index))

        if self.svg_dir is not None:
            if not self.svg_dir.exists():
                self.svg_dir.mkdir()
            print('Save figures...')
            for idx in identifiers:
                figure = '%i-%i-%s.svg' % (identifier_times[idx], idx, fpsmi_dict[idx])
                with open(pathlib.Path(self.svg_dir, figure), 'w') as f:
                    f.write(fpsvg_dict[idx])

        return index_list


class Morgan1CountIndexer(MorganCountIndexer):
    name = 'morgan1'

    def __init__(self):
        super().__init__()
        self.radius = 1
