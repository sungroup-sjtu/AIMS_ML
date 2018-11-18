#!/usr/bin/env python3

from MDLearn import preprocessing, example

import numpy as np 
import matplotlib.pyplot as plt


datax, _, name = example.load_altp(target='Density', load_names=True)
molecule_names = [
    n.split('\t', 1)[0] for n in name
]

selector0 = preprocessing.Selector(np.array(list(set(molecule_names))))
selector0.partition(0.9)
mol_tv = selector0.training_set()

# Add some randomness
# TODO This function should be removed

selector = preprocessing.Selector(datax)
tv_index = np.array([m in mol_tv for m in molecule_names], dtype=bool)
selector.partition(7/9)
selector.training_index = np.logical_and(selector.training_index, tv_index)
selector.validation_index = np.logical_and(selector.test_index, tv_index)
selector.test_index = np.logical_not(tv_index)

selector.save('tests/partition.txt')

trainx = selector.training_set()
validx = selector.validation_set()
testx = selector.test_set()

print(len(trainx), len(validx), len(testx))

for i in range(datax.shape[1]):

    d1, e1 = np.histogram(trainx[:, i], normed=True)
    d2, e2 = np.histogram(validx[:, i], normed=True)
    d3, e3 = np.histogram(testx[:, i], normed=True)
    
    plt.figure()
    plt.title(str(i))
    plt.plot((e1[:-1] + e1[1:])/2, d1, 'x-', lw=0.5, label='training')
    plt.plot((e2[:-1] + e2[1:])/2, d2, 'x-', lw=0.5, label='validation')
    plt.plot((e3[:-1] + e3[1:])/2, d3, 'x-', lw=0.5, label='test')

    plt.legend(loc='upper right')

    plt.show()

