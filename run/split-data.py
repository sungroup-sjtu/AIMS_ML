#!/usr/bin/env python3

import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from mdlearn import preprocessing, example

parser = argparse.ArgumentParser(description='Alkane property fitting demo')
parser.add_argument('-i', '--input', type=str, help='Data')
parser.add_argument('-f', '--fp', type=str, help='Fingerprints')
parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')
parser.add_argument('-t', '--target', default='raw_density', type=str, help='Fitting target')
parser.add_argument('--visual', default=0, type=int, help='Visualize data')

opt = parser.parse_args()

datax, _, names = example.load_altp(filename=opt.input, fps=opt.fp.split(','), target=opt.target, load_names=True)
mol_names = [n.split()[0] for n in names]

selector0 = preprocessing.Selector(np.array(list(set(mol_names))))
selector0.partition(0.8, 0.1)
mol_train = selector0.training_set()
mol_valid = selector0.validation_set()
mol_testt = selector0.test_set()

selector = preprocessing.Selector(datax)
selector.training_index = np.array([m in mol_train for m in mol_names], dtype=bool)
selector.validation_index = np.array([m in mol_valid for m in mol_names], dtype=bool)
selector.test_index = np.array([m in mol_testt for m in mol_names], dtype=bool)

selector.save(opt.output + '/part.txt')

trainx = selector.training_set()
validx = selector.validation_set()
testtx = selector.test_set()

print(len(trainx), len(validx), len(testtx))


def logscale(x):
    return -1 if x < 1 else np.log10(x)


def bounds(array):
    return math.floor(array.min()) - 0.5, math.ceil(array.max()) + 0.5


def bins(array):
    return math.ceil(array.max()) - math.floor(array.min()) + 1


if opt.visual != 0:
    for i in range(datax.shape[1]):
        x = trainx[:, i]
        d1, e1 = np.histogram(x, range=bounds(x), bins=bins(x))
        x = validx[:, i]
        d2, e2 = np.histogram(x, range=bounds(x), bins=bins(x))
        x = testtx[:, i]
        d3, e3 = np.histogram(x, range=bounds(x), bins=bins(x))

        plt.figure()
        plt.title(str(i))
        plt.plot((e1[:-1] + e1[1:]) / 2, [logscale(x) for x in d1], 'o--', lw=1.5, label='training')
        plt.plot((e2[:-1] + e2[1:]) / 2, [logscale(x) for x in d2], '*--', lw=1.5, label='validation')
        plt.plot((e3[:-1] + e3[1:]) / 2, [logscale(x) for x in d3], 'x--', lw=1.5, label='test')

        plt.legend(loc='upper right')
        plt.ylim(0, 6)

        plt.savefig(opt.output + '/part-%03i.png' % i)
