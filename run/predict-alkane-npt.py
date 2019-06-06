#!/usr/bin/env python3

import sys
import argparse
import numpy as np

sys.path.append('..')
from mdlearn import fitting, preprocessing, encoding

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='out', help='Model directory')
parser.add_argument('-i', '--input', default='CC,207,100', help='Molecule,T,P')
parser.add_argument('-f', '--fp', help='Using GPU')
parser.add_argument('--batch', default='', help='Batch input file')
parser.add_argument('--gpu', default=0, type=int, help='Using GPU')

opt = parser.parse_args()

model = fitting.TorchMLPRegressor(None, None, [])
model.is_gpu = opt.gpu == 1
model.load(opt.dir + '/model.pt')

scaler = preprocessing.Scaler()
scaler.load(opt.dir + '/scale.txt')

encoders = opt.fp.split(',')
encoder = encoding.FPEncoder(encoders, fp_name=opt.dir + '/fp_predict')

smiles = []
t = []
p = []

if not opt.batch:
    smiles_, t_, p_ = opt.input.split(',')
    smiles.append(smiles_)
    t.append(float(t_))
    p.append(float(p_))

else:
    with open(opt.batch, 'r') as f:
        for line in f.readlines():
            s_, t_, p_ = line.split()
            smiles.append(s_)
            t.append(float(t_))
            p.append(float(p_))

encoder.load_data(np.array(smiles), np.array(t), np.array(p))
datax = encoder.encode()
datax = scaler.transform(datax)
datay = model.predict_batch(datax)

print('SMILES\tT\tP\tResult')
for s_, t_, p_, y_ in zip(smiles, t, p, datay):
    print('%s\t%g\t%g\t%.3g' % (s_, t_, p_, y_))
