import pickle
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

import os
import sys
import argparse
import logging
from sklearn.decomposition import PCA

logging.captureWarnings(True)

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append('..')
from mdlearn import fitting, visualize, metrics, preprocessing, validation, dataloader

def sobol_analyze(model, X, logger, N=50000, **kwargs ):
    ''' INPUT:
            ::model::  the NN that takes X as input,
            ::X:: a 2D vector of input [batch_size, feature_length] 
            ::N:: the analyze point size, see more detail in SALib
            ::**kwargs:: specify optional kwargs for 'sobel.analyze(**kwargs)'
        RETURN:  
            a dict of sobol importance analysis
        '''
    upper = X.max(axis=0)
    lower = X.min(axis=0)
    bounds = [[i, j] for i,j in zip(lower, upper)] 
    all_feature_problem = {
        'num_vars': X.shape[1],
        'names': ['x'+str(i) for i in range(X.shape[1])],
        'bounds': bounds
    }
    logger.info("Generate feature samples")
    params_value = saltelli.sample(all_feature_problem, N)
    y_est = model.predict_batch(params_value).flatten()
    logger.info("Start analyze")
    result = sobol.analyze( all_feature_problem, y_est, **kwargs )

    return result

def load_data(opt, logger):
    if opt.layer != "":
        layers = list(map(int, opt.layer.split(',')))
    else:
        layers = []

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    if opt.featrm == 'auto':
        featrm = [14, 15, 17, 18, 19, 20, 21, 22]
    elif opt.featrm == '':
        featrm = []
    else:
        featrm = list(map(int, opt.featrm.split(',')))

    logger.info('loading data...')
    datax, datay, data_names = dataloader.load(filename=opt.input, target=opt.target, fps=opt.fp.split(','), featrm=featrm)

    selector = preprocessing.Selector(datax, datay, data_names)
    if opt.part:
        selector.load(opt.part)
    else:
        selector.partition(0.8, 0.1)
        selector.save(opt.output + '/part.txt')
        
    trainx, trainy, trainname = selector.training_set()
    validx, validy, validname = selector.validation_set()
    
    logger.info('loading model...')
    scaler = preprocessing.Scaler()
    scaler.load(opt.output + '/scale.txt')
    normed_trainx = scaler.transform(trainx)
    normed_validx = scaler.transform(validx)
    model = fitting.TorchMLPRegressor(None, None, [],
                                      is_gpu= False,
                                      )
    model.load(opt.output + '/model.pt')
    #if  opt.pca != -1:
    #    normed_trainx, normed_validx, _ = pca_nd(normed_trainx, normed_validx, len(normed_trainx[0]) - opt.pca)
    return normed_validx, validy, model

def main():
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-f', '--fp', type=str, help='Fingerprints')
    parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')
    parser.add_argument('-t', '--target', default='raw_density', type=str, help='Fitting target')
    parser.add_argument('-p', '--part', default='', type=str, help='Partition cache file')
    parser.add_argument('-l', '--layer', default='16,16', type=str, help='Size of hidden layers')
    parser.add_argument('--visual', default=1, type=int, help='Visualzation data')
    parser.add_argument('--gpu', default=1, type=int, help='Using gpu')
    parser.add_argument('--epoch', default="200", type=str, help='Number of epochs')
    parser.add_argument('--step', default=500, type=int, help='Number of steps trained for each batch')
    parser.add_argument('--batch', default=int(1e9), type=int, help='Batch size')
    parser.add_argument('--lr', default="0.005", type=str, help='Initial learning rate')
    parser.add_argument('--l2', default=0.000, type=float, help='L2 Penalty')
    parser.add_argument('--check', default=10, type=int,
                        help='Number of epoch that do convergence check. Set 0 to disable.')
    parser.add_argument('--minstop', default=0.2, type=float, help='Minimum fraction of step to stop')
    parser.add_argument('--maxconv', default=2, type=int, help='Times of true convergence that makes a stop')
    parser.add_argument('--featrm', default='', type=str, help='Remove features')
    parser.add_argument('--optim', default='rms', type=str, help='optimizer')
    parser.add_argument('--continuation', default=False, type=bool, help='continue training')
    parser.add_argument('--pca', default=0, type=int, help='dimension to discard')
    parser.add_argument('--sobol', default=-1, type=int, help='dimensions to reduce according to sensitivity analysis')

    opt = parser.parse_args()

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    flog = logging.FileHandler(opt.output + '/log.txt', mode='w')
    flog.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
    flog.setFormatter(formatter)
    clog = logging.StreamHandler()
    clog.setFormatter(formatter)
    logger.addHandler(flog)
    logger.addHandler(clog)

    validx, validy, model = load_data(opt, logger)
    logger.info('performing sobel sensitivity analysis...')
    result = sobol_analyze(model, validx, logger, 1000)
    sobel_idx = np.argsort(result['S1'][:-2])
    logger.info('saving model')
    with open(opt.output + '/sobol_S1.pkl', 'wb') as file:
        pickle.dump(result['S1'], file)
    with open(opt.output + '/sobol_idx.pkl', 'wb') as file:
        pickle.dump(sobel_idx, file)
        logger.info('save success')
    logger.info('analysis success')


if __name__=="__main__":
    main()