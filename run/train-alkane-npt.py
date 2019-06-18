#!/usr/bin/env python3

import os
import sys
import argparse
import logging

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

    validy_ = validy.copy() # for further convenience
    trainy_ = trainy.copy()
    if opt.gpu: # store everything to GPU all at once
        logger.info('Using GPU acceleration')
        device = torch.device( "cuda:0")
        normed_trainx = torch.Tensor(normed_trainx).to(device)
        trainy = torch.Tensor(trainy).to(device)
        normed_validx = torch.Tensor(normed_validx).to(device)
        validy = torch.Tensor(validy).to(device)

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD
    elif opt.optim == 'adam':
        optimizer = torch.optim.Adam
    elif opt.optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif opt.optim == 'ada':
        optimizer = torch.optim.Adagrad



    model = fitting.TorchMLPRegressor(len(trainx[0]), len(trainy[0]), layers, batch_size=opt.batch, batch_step=opt.step,
                                      is_gpu=opt.gpu != 0,
                                      args_opt={'optimizer'   : optimizer,
                                                'lr'          : opt.lr,
                                                'weight_decay': opt.l2
                                                }
                                      )

    model.init_session()
    model.load_data(normed_trainx, trainy)
    if opt.continuatiuon:
        file_dir = opt.output + '/model.pt'
        logger.info('Continue training from checkpoint %s' % (file_dir) )
        model.load( file_dir )

    logger.info('Optimizer = %s' % (optimizer))

    header = 'Step Loss MeaSquE MeaSigE MeaUnsE MaxRelE Acc1% Acc2% Acc5% Acc10%'.split()
    logger.info('%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s' % tuple(header))

    mse_history = []
    converge_times = 0
    best_last_score = None
    model_saved = False
    all_epoch = sum(opt_epochs)
    total_epoch = 0

    for k, each_epoch in enumerate(opt_epochs):
        # implement seperated learning rate
        model.reset_optimizer({'optimizer':optimizer, 'lr':opt_lr[k], 'weight_decay':opt.l2 } )
        for i_epoch in range(each_epoch):
            total_epoch += 1
            step, loss = model.fit_epoch()
            if (i_epoch + 1) % 1 == 0 or i_epoch + 1 == each_epoch:
                predy = model.predict_batch(normed_validx)
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

            if opt.check != 0:
                if i_epoch % opt.check == 0:  # check convergence
                    predy = model.predict_batch(normed_validx)
                    mse_history.append(metrics.mean_squared_error(validy, predy))

                    if i_epoch > opt.epoch * opt.minstop:
                        conv, cur_conv = validation.is_converge(np.array(mse_history), nskip=25)
                        if conv:
                            logger.info('Model converge detected at epoch %d' % i_epoch)
                            converge_times += 1

                        if converge_times >= opt.maxconv and cur_conv:
                            logger.info('Model converged at epoch: %d' % i_epoch)
                            break

                elif i_epoch > opt.epoch - opt.check:  # save best one in last frames
                    predy = model.predict_batch(normed_validx)
                    score = metrics.mean_squared_error(validy, predy)
                    if best_last_score and best_last_score < score:
                        logger.info('Saving best model in advance: Epoch %d' % i_epoch)
                        model.save(opt.output + '/model.pt')
                        model_saved = True

                    best_last_score = min(best_last_score, score) if best_last_score else score

    if mse_history:
        conv, cur_conv = validation.is_converge(np.array(mse_history))
        if not conv:
            logger.warning('Model not converged.')
        elif not cur_conv:
            logger.warning('Model converged, but the model saved may not be the best.')

    if not model_saved:
        model.save(opt.output + '/model.pt')

    predy = model.predict_batch(normed_validx)
    visualizer = visualize.LinearVisualizer(trainy_.reshape(-1), model.predict_batch(normed_trainx).reshape(-1), trainname, 'train')
    visualizer.append(validy_.reshape(-1), predy.reshape(-1), validname, 'test')
    visualizer.dump(opt.output + '/fit.txt')
    logger.info('Fitting result saved')

    if opt.visual:
        visualizer.scatter_yy(annotate_threshold=0.1, marker='x', lw=0.2, s=5)
        visualizer.scatter_error(annotate_threshold=0.1, marker='x', lw=0.2, s=5)
        visualizer.hist_error(label='test', histtype='step', bins=50)

        plt.show()


if __name__ == '__main__':
    main()
