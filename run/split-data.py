#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append('..')
from mdlearn import preprocessing

parser = argparse.ArgumentParser(description='Alkane property fitting demo')
parser.add_argument('-i', '--input', type=str, help='Data')
parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')

opt = parser.parse_args()

df = pd.read_csv(opt.input, sep='\s+', header=0)
smiles_array = df.SMILES.values
selector = preprocessing.Selector(smiles_array)
sel_mol = preprocessing.Selector(df.SMILES.unique())
fold = 5
sel_mol.kfold_partition(1.0, fold)
for n in range(fold):
    sel_mol.kfold_use(n)
    mol_train = sel_mol.training_set()
    mol_valid = sel_mol.validation_set()
    mol_testt = sel_mol.test_set()

    selector.training_index = np.array([m in mol_train for m in smiles_array], dtype=bool)
    selector.validation_index = np.array([m in mol_valid for m in smiles_array], dtype=bool)
    selector.test_index = np.array([m in mol_testt for m in smiles_array], dtype=bool)

    selector.save(opt.output + '/part-%i.txt' % (n + 1))

train_valid_x = selector.train_valid_set()
test_x = selector.test_set()
print(len(train_valid_x), len(test_x))
