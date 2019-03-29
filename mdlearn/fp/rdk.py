import sys
import pathlib
import numpy as np

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect


class ECFP4Indexer:
    name = 'ecfp4'

    def __init__(self):
        pass

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        return np.array(list(map(int, Chem.GetMorganFingerprintAsBitVect(rdk_mol, radius=2, nBits=2048))))

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]


class MorganCountIndexer:
    name = 'morgan'

    def __init__(self):
        self.svg_dir = None
        self.radius = 2

    def index(self, smiles):
        raise Exception('Use index_list() for this indexer')

    def index_list(self, smiles_list):
        fp_list = []
        identifiers = set()
        fpsvg_dict = {}
        fpsmi_dict = {}
        index_list = []

        print('\nCalculate with RDKit...')
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
                            fpsmi_dict[idx] = ''
                        else:
                            env = Chem.FindAtomEnvironmentOfRadiusN(rdk_mol, radius, root)  # args: mol, radius, atomId
                            amap = {}
                            submol = Chem.PathToSubmol(rdk_mol, env, atomMap=amap)
                            fpsmi_dict[idx] = Chem.MolToSmiles(submol, rootedAtAtom=amap[info[idx][0][0]],
                                                               isomericSmiles=False, canonical=False)

        print('\nFilter identifiers...')
        print('%i identifiers total' % len(identifiers))
        identifiers = list(sorted(identifiers))
        identifier_times = dict([(id, 0) for id in identifiers])
        for fp in fp_list:
            for idx in fp.GetNonzeroElements().keys():
                identifier_times[idx] += 1
        identifiers = [idx for idx, times in identifier_times.items() if times >= 40]
        print('%i identifiers saved' % len(identifiers))

        for fp in fp_list:
            index = [fp.GetNonzeroElements().get(idx, 0) for idx in identifiers]
            index_list.append(np.array(index))

        if self.svg_dir is not None:
            print('\nSave figures...')
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
