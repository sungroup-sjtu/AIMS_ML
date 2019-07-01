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

def sobol_analyze(model, X, logger, N=5000, **kwargs, ):
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
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-f', '--fp', type=str, help='Fingerprints')
    parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')
    parser.add_argument('-t', '--target', default='raw_density', type=str, help='Fitting target')
    parser.add_argument('-p', '--part', default='', type=str, help='Partition cache file')
    parser.add_argument('--featrm', default='', type=str, help='Remove features')
    parser.add_argument('--pca', default=0, type=int, help='dimension to discard')
    parser.add_argument('-l', '--layer', default='16,16', type=str, help='Size of hidden layers')

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
    with open(opt.output + '/pickle_example.pickle', 'wb') as file:
        pickle.dump(sobel_idx, file)
        logger.info('save success')
    logger.info('analysis success')


if __name__=="__main__":
    main()