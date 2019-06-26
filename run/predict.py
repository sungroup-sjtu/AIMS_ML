#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np

sys.path.append('..')
from mdlearn import fitting, preprocessing, encoding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='out', help='Model directory')
    parser.add_argument('-i', '--input', default='CC,207,100', help='Molecule,T,P')
    parser.add_argument('-f', '--fp', help='Fingerprints')
    parser.add_argument('--batch', default='', help='Batch input file')
    parser.add_argument('--gpu', default=0, type=int, help='Using GPU')

    opt = parser.parse_args()

    model = fitting.TorchMLPRegressor(None, None, [])
    model.is_gpu = opt.gpu == 1
    model.load(opt.dir + '/model.pt')

    scaler = preprocessing.Scaler()
    scaler.load(opt.dir + '/scale.txt')

    encoders = opt.fp.split(',')
    encoder = encoding.FPEncoder(encoders, fp_name=opt.dir + '/fp')

    smiles_list = []
    t_list = []
    p_list = []

    if not opt.batch:
        words = opt.input.split(',')
        smiles_list.append(words[0])
        if len(words) >= 2:
            t_list.append(float(words[1]))
        if len(words) >= 3:
            p_list.append(float(words[2]))

    else:
        with open(opt.batch, 'r') as f:
            for line in f.readlines():
                words = line.split(',')
                smiles_list.append(words[0])
                if len(words) >= 2:
                    t_list.append(float(words[1]))
                if len(words) >= 3:
                    p_list.append(float(words[2]))

    args = [np.array(smiles_list)]
    if t_list != []:
        args.append(np.array(t_list))
    if p_list != []:
        args.append(np.array(p_list))

    encoder.load_data(*args)
    datax = encoder.encode(save_fp=False)
    datax = scaler.transform(datax)
    datay = model.predict_batch(datax)

    print('SMILES\tT\tP\tResult')
    for s_, t_, p_, y_ in zip(smiles_list, t_list or [0] * len(smiles_list), p_list or [0] * len(smiles_list), datay):
        print('%s\t%g\t%g\t%.3g' % (s_, t_, p_, y_))


if __name__ == '__main__':
    main()
