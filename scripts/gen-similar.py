#!/usr/bin/env python3


import os
import sys
import hashlib
import random
import pybel
from rdkit.Chem import AllChem as  Chem

sys.path.append('..')
from mdlearn.utils import ml_predict


def md5sum8(string):
    return hashlib.md5(string.encode()).hexdigest()[12:20]


def reduce_mol(smiles):
    s_list = [smiles]
    rdk_mol = Chem.MolFromSmiles(smiles)
    while True:
        rdk_mol = Chem.ReplaceSubstructs(rdk_mol, Chem.MolFromSmarts('[CH2][CH3]'), Chem.MolFromSmiles('C'), replaceAll=True)[0]
        s = Chem.MolToSmiles(rdk_mol)
        if s != s_list[-1]:
            s_list.append(s)
        else:
            break

    return s_list[-1]


def reduce_mol_ring(smiles):
    s_list = [smiles]
    rdk_mol = Chem.MolFromSmiles(smiles)
    while True:
        rdk_mol = Chem.DeleteSubstructs(rdk_mol, Chem.MolFromSmarts('[CH3]'))
        s = Chem.MolToSmiles(rdk_mol)
        if s != s_list[-1]:
            s_list.append(s)
        else:
            break

    return s_list[-1]


def enlarge_mol(smiles, substitutes=None, atom_limit=1000, max_sample_per_substitute=1000):
    if substitutes is None:
        substitutes = ['C']

    s_list = [smiles]
    rdk_mol = Chem.MolFromSmiles(smiles)
    for sub in substitutes:
        _set = set()
        _tuple = Chem.ReplaceSubstructs(rdk_mol, Chem.MolFromSmarts('[#6;H,H2,H3]'), Chem.MolFromSmiles('C' + sub), replaceAll=False)
        for mol in _tuple:
            if len(mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]'))) > atom_limit:
                continue
            s = Chem.MolToSmiles(mol)
            _set.add(s)
        if len(_set) > max_sample_per_substitute:
            _set = random.sample(_set, max_sample_per_substitute)
        s_list.extend(_set)

    return s_list


if __name__ == '__main__':
    with open(sys.argv[1])  as f:
        lines = f.read().splitlines()
    smiles_list = []
    for line in lines:
        if line.strip() == '':
            continue
        smiles = line.strip().split()[0]
        if smiles not in smiles_list:
            smiles_list.append(smiles)

    print('#No\tFormula\tCAS\tSMILES\tTfus\tTvap\tTc')
    smiles_dict = {}
    for smiles in smiles_list:
        smiles_base = reduce_mol(smiles)
        py_mol = pybel.readstring('smi', smiles_base)
        if len(py_mol.sssr) >= 2 and py_mol.OBMol.NumHvyAtoms() >= 16:
            smiles_base = reduce_mol_ring(smiles)

        s_list = enlarge_mol(smiles_base, substitutes=['CC', 'C(C)=C', 'C(C)CC', 'C1CCCC1'], atom_limit=19, max_sample_per_substitute=3)

        for i, s in enumerate(s_list):
            smiles_dict[f'{md5sum8(smiles)}-{i + 1}'] = pybel.readstring('smi', s)

    new_s_list = []
    for name, py_mol in smiles_dict.items():
        s = py_mol.write('can').strip()
        if s in new_s_list:
            continue
        new_s_list.append(s)
        tvap = ml_predict('../ml-models/CH-tvap', [s], [], [])[0][0]
        tc = ml_predict('../ml-models/CH-tc', [s], [], [])[0][0]
        print(f'0\t{py_mol.formula}\t{name}\t{s}\tNone\t{int(round(tvap, 0))}\t{int(round(tc, 0))}')
