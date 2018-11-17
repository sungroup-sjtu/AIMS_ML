#!/usr/bin/env python3

import sys 
import argparse
import numpy as np 

from MDLearn import example, fitting, preprocessing, encoding, fp 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tests/', help='Model directory')
    parser.add_argument('--input', default='CC,207,100', help='Molecule,T,P')
    parser.add_argument('--batch', default='', help='Batch input file')
    parser.add_argument('--gpu', default=1, type=int, help='Using GPU')

    opt = parser.parse_args()

    fitting.backend = 'tch'

    nn = fitting.PerceptronFitter(None, None, [])
    nn.regressor.is_gpu = opt.gpu == 1
    nn.regressor.load(opt.model + 'model.pt')

    scaler = preprocessing.Scaler()
    scaler.load(opt.model + 'scale.txt')

    encoder = encoding.FPEncoder(fp.AlkaneSKIndexer, cache_filename=None)


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
    # sel = np.ones(datax.shape[1], dtype=bool)
    # sel[[14,15,17,18,19,20,21,22]] = False
    # datax = datax[:, sel]
    datax = scaler.transform(datax)
    datay = nn.predict_batch(datax)

    print('SMILES\tT\tP\tResult')
    for s_, t_, p_, y_ in zip(smiles, t, p, datay):
        print('%s\t%g\t%g\t%.3g' % (s_, t_, p_, y_))

if __name__ == '__main__':
    main()