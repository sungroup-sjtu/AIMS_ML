#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append('..')
from mdlearn import preprocessing

def main():
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-o', '--output', default='fp', type=str, help='Output directory')
    parser.add_argument('--fold', default=0, type=int, help='using n-fold partition as validation set')
    parser.add_argument('--similarity', default=-1.0, type=float, help='using similarity partition as validation set')

    opt = parser.parse_args()

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    smiles_list = []
    smiles_list_training = None
    input_list = opt.input.split(',')
    for file in input_list:
        df = pd.read_csv(file, sep='\s+', header=0)
        if 'train' in file:
            smiles_list_training = df.SMILES.unique().tolist()
        else:
            smiles_array = df.SMILES.values
    selector = preprocessing.Selector(smiles_array)
    sel_mol = preprocessing.Selector(smiles_array)

    if len(input_list) == 2 and smiles_list_training is not None:
        sel_mol.partition_smiles_list(smiles_list_training=smiles_list_training)
        mol_train = sel_mol.training_set()
        mol_valid = sel_mol.validation_set()

        mol_train_dict = dict([(s, 1) for s in mol_train])
        mol_valid_dict = dict([(s, 1) for s in mol_valid])

        selector.training_index = np.array([mol_train_dict.get(m, 0) for m in smiles_array], dtype=bool)
        selector.validation_index = np.array([mol_valid_dict.get(m, 0) for m in smiles_array], dtype=bool)
        selector.test_index = np.logical_not(np.logical_or(selector.training_index, selector.validation_index))

        selector.save(opt.output + '/part.txt')

    elif opt.similarity > 0.0:
        sel_mol.similarity_partition(cutoff=opt.similarity)
        mol_train = sel_mol.training_set()
        mol_valid = sel_mol.validation_set()

        mol_train_dict = dict([(s, 1) for s in mol_train])
        mol_valid_dict = dict([(s, 1) for s in mol_valid])

        selector.training_index = np.array([mol_train_dict.get(m, 0) for m in smiles_array], dtype=bool)
        selector.validation_index = np.array([mol_valid_dict.get(m, 0) for m in smiles_array], dtype=bool)
        selector.test_index = np.logical_not(np.logical_or(selector.training_index, selector.validation_index))

        selector.save(opt.output + '/part-similarity-%.2f.txt' % (opt.similarity))
    elif opt.fold != 0:
        fold = opt.fold
        sel_mol.kfold_partition(1.0, fold)
        for n in range(fold):
            sel_mol.kfold_use(n)
            mol_train = sel_mol.training_set()
            mol_valid = sel_mol.validation_set()

            mol_train_dict = dict([(s, 1) for s in mol_train])
            mol_valid_dict = dict([(s, 1) for s in mol_valid])

            selector.training_index = np.array([mol_train_dict.get(m, 0) for m in smiles_array], dtype=bool)
            selector.validation_index = np.array([mol_valid_dict.get(m, 0) for m in smiles_array], dtype=bool)
            selector.test_index = np.logical_not(np.logical_or(selector.training_index, selector.validation_index))

            selector.save(opt.output + '/part-%i.txt' % (n + 1))


if __name__ == '__main__':
    main()
