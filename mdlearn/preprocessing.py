""" Preprocessing
"""

import sklearn.model_selection
import sklearn.preprocessing
import numpy as np
import sys


def separate(array, frac):
    n = int(round(len(array) * frac))
    train_array = np.random.choice(array, n, replace=False)
    test_array = np.array([i for i in array if i not in train_array], dtype=int)
    return train_array, test_array


class Selector:
    """ OOP Selector
    """

    def __init__(self, *data):
        """ data: np.array/pandas.df
        """
        self.training_index = None
        self.validation_index = None
        self.test_index = None

        self.kfold = None
        self.kfold_train_indexes = None
        self.kfold_valid_indexes = None

        for d in data[1:]:
            assert len(d) == len(data[0]), 'Data does not have same length'

        self.data = data
        self.length = len(self.data[0])
        self.selected_num = np.arange(self.length)

    def partition(self, training_size, validation_size):
        """ Perform 3-partition (training + validating + testing)
        """

        self.training_index = np.zeros(self.length, dtype=bool)
        self.validation_index = np.zeros(self.length, dtype=bool)
        self.test_index = np.zeros(self.length, dtype=bool)

        train_valid_num, test_num = separate(self.selected_num, training_size + validation_size)
        train_num, valid_num = separate(train_valid_num, training_size / (training_size + validation_size))
        self.training_index[train_num] = True
        self.validation_index[valid_num] = True
        self.test_index[test_num] = True

    def partition_smiles_list(self, smiles_list_training):
        self.training_index = np.ones(self.length, dtype=bool)
        self.validation_index = np.zeros(self.length, dtype=bool)
        self.test_index = np.zeros(self.length, dtype=bool)
        for i, smiles in enumerate(self.data[0]):
            if smiles not in smiles_list_training:
                self.training_index[i] = False
                self.validation_index[i] = True

    def similarity_partition(self, cutoff=10.0):
        def similarity_comparison(smiles1, smiles2):
            from rdkit.Chem import AllChem as Chem
            from rdkit import DataStructs
            rdk_mol1 = Chem.MolFromSmiles(smiles1)
            fp1 = Chem.GetMorganFingerprintAsBitVect(rdk_mol1, 2)
            rdk_mol2 = Chem.MolFromSmiles(smiles2)
            fp2 = Chem.GetMorganFingerprintAsBitVect(rdk_mol2, 2)
            # print(smiles1, smiles2)
            return DataStructs.DiceSimilarity(fp1, fp2)
        # a switch function of similarity_comparison
        def similarity_score(smiles1, smiles2):
            a = similarity_comparison(smiles1, smiles2)
            if a < 0.6:
                return 0.
            else:
                return (a - 0.6) / 0.4
        def get_similarity_score(smiles, smiles_list):
            score = 0.
            for s in smiles_list:
                score += similarity_score(smiles, s)
            return score
        self.training_index = np.ones(self.length, dtype=bool)
        self.validation_index = np.zeros(self.length, dtype=bool)
        self.test_index = np.zeros(self.length, dtype=bool)

        smiles_list = []
        for i, smiles in enumerate(self.data[0]):
            sys.stdout.write('\r%i / %i' % (i, len(self.data[0])))
            if get_similarity_score(smiles, smiles_list) < cutoff:
                smiles_list.append(smiles)
            else:
                self.training_index[i] = False
                self.validation_index[i] = True
        print('\n%i molecules in training set' % len(smiles_list))

    def kfold_partition(self, train_valid_size, fold=5):
        self.kfold = fold
        self.kfold_train_indexes = [np.ones(self.length, dtype=bool) for i in range(fold)]
        self.kfold_valid_indexes = [np.zeros(self.length, dtype=bool) for i in range(fold)]
        self.test_index = np.zeros(self.length, dtype=bool)

        if train_valid_size < 1:
            train_valid_num, test_num = separate(self.selected_num, train_valid_size)
        else:
            train_valid_num = self.selected_num[:]
            test_num = np.array([], dtype=int)
        self.test_index[test_num] = True

        for k in range(fold):
            valid_num, train_valid_num = separate(train_valid_num, 1 / (fold - k))
            print(valid_num)
            self.kfold_valid_indexes[k][valid_num] = True
            self.kfold_train_indexes[k][valid_num] = False
            self.kfold_train_indexes[k][test_num] = False

    def kfold_use(self, n):
        self.training_index = self.kfold_train_indexes[n]
        self.validation_index = self.kfold_valid_indexes[n]

    def training_set(self):
        idx = self.training_index
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def validation_set(self):
        idx = self.validation_index
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def test_set(self):
        idx = self.test_index
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def train_valid_set(self):
        idx = np.logical_not(self.test_index)
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def load(self, filename):
        with open(filename, 'r') as f_cache:
            line1 = f_cache.readline().rstrip('\n')
            line2 = f_cache.readline().rstrip('\n')
            line3 = f_cache.readline().rstrip('\n')

            self.training_index = np.array(list(map(int, line1)), dtype=bool)
            self.validation_index = np.array(list(map(int, line2)), dtype=bool)
            self.test_index = np.array(list(map(int, line3)), dtype=bool)

    def save(self, filename):
        with open(filename, 'w') as f_cache:
            f_cache.write(''.join(map(str, self.training_index.astype(int))))
            f_cache.write('\n')
            f_cache.write(''.join(map(str, self.validation_index.astype(int))))
            f_cache.write('\n')
            f_cache.write(''.join(map(str, self.test_index.astype(int))))
            f_cache.write('\n')


class Scaler:
    """ StandardScaler that can save
    """

    def __init__(self, *args, **kwargs):
        self.scaler_ = sklearn.preprocessing.StandardScaler(*args, **kwargs)

    def fit(self, *args):
        return self.scaler_.fit(*args)

    def transform(self, *args):
        return self.scaler_.transform(*args)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(','.join(map(str, self.scaler_.mean_)) + '\n')
            f.write(','.join(map(str, self.scaler_.var_)) + '\n')
            f.write(','.join(map(str, self.scaler_.scale_)) + '\n')

    def load(self, filename):
        self.scaler_.scale_ = 1
        with open(filename, 'r') as f:
            line1 = f.readline().strip('\n')
            line2 = f.readline().strip('\n')
            line3 = f.readline().strip('\n')

            self.scaler_.mean_ = np.array(list(map(float, line1.split(','))))
            self.scaler_.var_ = np.array(list(map(float, line2.split(','))))
            self.scaler_.scale_ = np.array(list(map(float, line3.split(','))))
            assert len(self.scaler_.mean_) == len(self.scaler_.var_)
