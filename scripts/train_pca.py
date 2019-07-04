#!/usr/bin/env python3

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


def main():
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

    if opt.layer != "":
        layers = list(map(int, opt.layer.split(',')))
    else:
        layers = []

    opt_lr = list(map(float, opt.lr.split(',')))
    opt_epochs = list(map(int, opt.epoch.split(',')))

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

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

    if opt.featrm == 'auto':
        logger.info('Automatically remove features')
        featrm = [14, 15, 17, 18, 19, 20, 21, 22]
    elif opt.featrm == '':
        featrm = []
    else:
        featrm = list(map(int, opt.featrm.split(',')))
    logger.info('Remove Feature: %s' % featrm)

    logger.info('Reading data...')
    datax, datay, data_names = dataloader.load(filename=opt.input, target=opt.target, fps=opt.fp.split(','), featrm=featrm)

    logger.info('Selecting data...')
    selector = preprocessing.Selector(datax, datay, data_names)
    if opt.part:
        logger.info('Loading partition file %s' % opt.part)
        selector.load(opt.part)
    else:
        logger.warning("Partition file not found. Using auto-partition instead.")
        selector.partition(0.8, 0.1)
        selector.save(opt.output + '/part.txt')
    trainx, trainy, trainname = selector.training_set()
    validx, validy, validname = selector.validation_set()

    logger.info('Training size = %d, Validation size = %d' % (len(trainx), len(validx)))
    logger.info('X input example: (size=%d) %s' % (len(datax[0]), ','.join(map(str, datax[0]))))
    logger.info('Y input example: (size=%d) %s' % (len(datay[0]), ','.join(map(str, datay[0]))))
    logger.info('Normalizing...')
    scaler = preprocessing.Scaler()
    scaler.fit(trainx)
    scaler.save(opt.output + '/scale.txt')
    normed_trainx = scaler.transform(trainx)
    normed_validx = scaler.transform(validx)

    logger.info('Building network...')
    logger.info('Hidden layers = %r' % layers)
    logger.info('optimizer = %s' % (opt.optim) )
    logger.info('Initial learning rate = %f' % opt_lr[0])
    logger.info('L2 penalty = %f' % opt.l2)
    logger.info('Total %d epochs' % sum(opt_epochs))
    logger.info('Batch = (%d values x %d steps)' % (opt.batch, opt.step))

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD
    elif opt.optim == 'adam':
        optimizer = torch.optim.Adam
    elif opt.optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif opt.optim == 'ada':
        optimizer = torch.optim.Adagrad

    result = []

    for i in range(20, len(trainx[0]), 5):
        logger.info( 'Start PCA trainning of dimension '+str(i) )
        pca_i_result = pca_train(i, normed_trainx, trainy,  normed_validx, validy, opt, logger, layers, opt_lr, opt_epochs, optimizer)
        logger.info('PCA reduced result of dimension %d :' % (i))
        logger.info( '%.3f variance_explained,\t acc2: %.3f,\t MSE %.3f ' % (pca_i_result) )
        result.append(pca_i_result)

    logger.info(result)


def pca_nd(X, X_valid, n, logger=None):
    X_mol = X#[:, :-2]
    X_unique = np.unique(X_mol, axis=0)
    pca = PCA(n_components=n)
    pca.fit(X_unique)
    X_transform = pca.transform(X_mol)
    # X_transform = np.c_[X_transform, X[:, -2:]]
    X_valid_transform  =  X_valid#[:, :-2]
    X_valid_transform  =  pca.transform(X_valid_transform)
    # X_valid_transform = np.c_[X_valid_transform, X_valid_transform[:, -2:]]
    if logger!=None:
        logger.info("total variance explained:%.3f" % (pca.explained_variance_ratio_.sum()) )
    else:
        print( "total variance explained:%.3f" % (pca.explained_variance_ratio_.sum()) )
    return X_transform, X_valid_transform, pca.explained_variance_ratio_.sum()


def pca_train(n, normed_trainx, trainy,  normed_validx, validy,  opt, logger, layers, opt_lr, opt_epochs, optimizer):
    trainx, validx, var_ex = pca_nd(normed_trainx, normed_validx, n, logger)
    validy_ = validy.copy() # for further convenience
    trainy_ = trainy.copy()
    if opt.gpu: # store everything to GPU all at once
        logger.info('Using GPU acceleration')
        device = torch.device( "cuda:0")
        trainx = torch.Tensor(trainx).to(device)
        trainy = torch.Tensor(trainy).to(device)
        validx = torch.Tensor(validx).to(device)
        validy = torch.Tensor(validy).to(device)
    model = fitting.TorchMLPRegressor(len(trainx[0]), len(trainy[0]), layers, batch_size=opt.batch, 
                                      is_gpu=opt.gpu != 0,
                                      args_opt={'optimizer': torch.optim.Adam,
                                                'lr': opt.lr,
                                                'weight_decay': opt.l2
                                                }
                                      )
    model.init_session()
    print(model.regressor)
    model.load_data(trainx, trainy)

    header = 'Epoch Loss MeaSquE MeaSigE MeaUnsE MaxRelE Acc1% Acc2% Acc5% Acc10%'.split()
    logger.info('%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % tuple(header))
    total_epoch = 0

    for k, each_epoch in enumerate(opt_epochs):
        # implement seperated learning rate
        model.reset_optimizer({'optimizer':optimizer, 'lr':opt_lr[k], 'weight_decay':opt.l2 } )
        for i_epoch in range(each_epoch):
            total_epoch += 1
            loss = model.fit_epoch(trainx, trainy)
            if (i_epoch + 1) % 20 == 0 or i_epoch + 1 == each_epoch:
                predy = model.predict_batch(validx)
                err_line = '%d/%d %8.3e %8.3e %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f' % (
                    total_epoch,
                    sum(opt_epochs),
                    loss,
                    metrics.mean_squared_error(validy_, predy),
                    metrics.mean_signed_error(validy_, predy) * 100,
                    metrics.mean_unsigned_error(validy_, predy) * 100,
                    metrics.max_relative_error(validy_, predy) * 100,
                    metrics.accuracy(validy_, predy, 0.01) * 100,
                    metrics.accuracy(validy_, predy, 0.02) * 100,
                    metrics.accuracy(validy_, predy, 0.05) * 100,
                    metrics.accuracy(validy_, predy, 0.10) * 100)
                logger.info(err_line)

    return  var_ex, metrics.accuracy(validy_, predy, 0.02) * 100, metrics.mean_squared_error(validy, predy)

if __name__ == '__main__':
    main()
