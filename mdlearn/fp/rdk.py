import sys
import pathlib
import numpy as np
import networkx as nx
from pathlib import Path

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

from . import Fingerprint
from .drawmorgan import DrawMorganBit


class ECFPIndexer(Fingerprint):
    name = 'ecfp'

    def __init__(self, arg=None):
        super().__init__()
        if arg is None:
            arg = [2, 1024]
        self.radius = arg[0]
        self.n_bits = arg[1]

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        return np.array(
            list(map(int, Chem.GetMorganFingerprintAsBitVect(rdk_mol, radius=self.radius, nBits=self.n_bits))))

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]


class MorganCountIndexer(Fingerprint):
    name = 'morgan'

    def __init__(self, arg=None):
        super().__init__()
        if arg is None:
            arg = [2, 200]
        self.radius = arg[0]
        self.fp_time_limit = arg[1]
        self.svg_dir: Path = None

    def index(self, smiles):
        raise Exception('Use index_list() for this indexer')

    def index_list(self, smiles_list):
        rdkfp_list = []
        identifiers = []
        fpsvg_dict = {}
        fpsmi_dict = {}
        bits_list = []

        print('Calculate with RDKit morgan fingerprints...')
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\t%i' % i)

            rdk_mol = Chem.MolFromSmiles(smiles)
            info = dict()
            rdkfp: UIntSparseIntVect = Chem.GetMorganFingerprint(rdk_mol, radius=self.radius, bitInfo=info)
            rdkfp_list.append(rdkfp)
            identifiers += list(rdkfp.GetNonzeroElements().keys())

            if self.svg_dir is not None:
                for idx in info.keys():
                    if idx not in fpsvg_dict:
                        fpsvg_dict[idx] = DrawMorganBit(rdk_mol, idx, info)

                        root, radius = info[idx][0]
                        if radius == 0:
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
        identifiers = set(identifiers)
        print('%i identifiers total' % len(identifiers))
        idx_times = dict([(id, 0) for id in identifiers])
        for rdkfp in rdkfp_list:
            for idx in rdkfp.GetNonzeroElements().keys():
                idx_times[idx] += 1
        self.bit_count = dict([(idx, 0) for idx, times in sorted(idx_times.items(), key=lambda x: x[1], reverse=True) if
                               times >= self.fp_time_limit])
        print('%i identifiers appears in more than %i molecules saved' % (len(self.idx_list), self.fp_time_limit))

        for rdkfp in rdkfp_list:
            bits = [rdkfp.GetNonzeroElements().get(idx, 0) for idx in self.idx_list]
            bits_list.append(np.array(bits))

        if self.svg_dir is not None:
            if not self.svg_dir.exists():
                self.svg_dir.mkdir()
            print('Save figures...')
            for idx in self.idx_list:
                figure = '%i-%i-%s.svg' % (idx_times[idx], idx, fpsmi_dict[idx])
                with open(pathlib.Path(self.svg_dir, figure), 'w') as f:
                    f.write(fpsvg_dict[idx])

        return bits_list


class Morgan1CountIndexer(MorganCountIndexer):
    name = 'morgan1'

    def __init__(self):
        super().__init__()
        self.radius = 1


class PredefinedMorganCountIndexer(Fingerprint):
    name = 'predefinedmorgan'

    def __init__(self, arg=None):
        super().__init__()
        if arg is None:
            arg = [2, 200]
        self.radius = arg[0]
        self.fp_time_limit = arg[1]
        self.use_pre_idx_list = 'morgan'
        self.pre_idx_list = []

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        fp: UIntSparseIntVect = Chem.GetMorganFingerprint(rdk_mol, radius=self.radius)
        return np.array([fp.GetNonzeroElements().get(int(idx), 0) for idx in self.pre_idx_list])

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]


class PredefinedMorgan1CountIndexer(PredefinedMorganCountIndexer):
    name = 'predefinedmorgan1'

    def __init__(self, fp_time_limit=200):
        super().__init__()
        self.radius = 1
        self.use_pre_idx_list = 'morgan1'


class TopologicalCountIndexer(Fingerprint):
    name = 'topological'

    def __init__(self, arg=None):
        super().__init__()
        if arg is None:
            arg = [1, 7, 200]
        self.minPath = arg[0]
        self.maxPath = arg[1]
        self.fp_time_limit = arg[2]
        self.svg_dir: Path = None

    def index(self, smiles):
        raise Exception('Use index_list() for this indexer')

    def index_list(self, smiles_list):
        rdkfp_list = []
        identifiers = []
        fpsvg_dict = {}
        fpsmi_dict = {}
        bits_list = []

        print('Calculate with RDKit topological fingerprints...')
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\t%i' % i)

            rdk_mol = Chem.MolFromSmiles(smiles)
            info = dict()
            rdkfp: UIntSparseIntVect = Chem.UnfoldedRDKFingerprintCountBased(rdk_mol, maxPath=self.maxPath)
            rdkfp_list.append(rdkfp)
            identifiers += list(rdkfp.GetNonzeroElements().keys())

            if self.svg_dir is not None:
                for idx in info.keys():
                    if idx not in fpsvg_dict:
                        fpsvg_dict[idx] = DrawMorganBit(rdk_mol, idx, info)

                        root, radius = info[idx][0]
                        if radius == 0:
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
        identifiers = set(identifiers)
        print('%i identifiers total' % len(identifiers))
        idx_times = dict([(id, 0) for id in identifiers])
        for rdkfp in rdkfp_list:
            for idx in rdkfp.GetNonzeroElements().keys():
                idx_times[idx] += 1
        self.bit_count = dict([(idx, 0) for idx, times in sorted(idx_times.items(), key=lambda x: x[1], reverse=True) if
                               times >= self.fp_time_limit])
        print('%i identifiers appears in more than %i molecules saved' % (len(self.idx_list), self.fp_time_limit))

        for rdkfp in rdkfp_list:
            bits = [rdkfp.GetNonzeroElements().get(idx, 0) for idx in self.idx_list]
            bits_list.append(np.array(bits))

        if self.svg_dir is not None:
            if not self.svg_dir.exists():
                self.svg_dir.mkdir()
            print('Save figures...')
            for idx in self.idx_list:
                figure = '%i-%i-%s.svg' % (idx_times[idx], idx, fpsmi_dict[idx])
                with open(pathlib.Path(self.svg_dir, figure), 'w') as f:
                    f.write(fpsvg_dict[idx])

        return bits_list


class PredefinedTopologicalCountIndexer(Fingerprint):
    name = 'predefinedtopological'

    def __init__(self, arg=None):
        super().__init__()
        if arg is None:
            arg = [1, 7, 200]
        self.minPath = arg[0]
        self.maxPath = arg[1]
        self.fp_time_limit = arg[2]
        self.use_pre_idx_list = 'topological'
        self.pre_idx_list = []

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        fp: UIntSparseIntVect = Chem.UnfoldedRDKFingerprintCountBased(rdk_mol)
        return np.array([fp.GetNonzeroElements().get(int(idx), 0) for idx in self.pre_idx_list])

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]
