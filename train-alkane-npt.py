#!/usr/bin/env python3

import sys
import argparse
import logging

logging.captureWarnings(True)

import matplotlib.pyplot as plt
import numpy as np
import torch

from MDLearn import example, fitting, visualize, metrics, preprocessing, validation


def main():
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser.add_argument('--input', default='_example_', type=str, help='Fitting object')
    parser.add_argument('--output', default='tests/', help='Output directory')
    parser.add_argument('--encoder', default='alkane', type=str, help='Fitting object')
    parser.add_argument('--visual', default=1, type=int, help='Visualzation data')
    parser.add_argument('--gpu', default=1, type=int, help='Using gpu')
    parser.add_argument('--obj', default='Density', type=str, help='Fitting object')
    parser.add_argument('--epoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--step', default=500, type=int, help='Number of steps trained for each batch')
    parser.add_argument('--batch', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.005, type=float, help='Initial learning rate')
    parser.add_argument('--layer', default='16,8', type=str, help='Size of hidden layers')
    parser.add_argument('--partition', default='', type=str, help='Partition cache file')
    parser.add_argument('--l2', default=0, type=float, help='L2 Penalty')
    parser.add_argument('--check', default=10, type=int,
                        help='Number of epoch that do convergence check. Set 0 to disable.')
    parser.add_argument('--minstop', default=0.2, type=float, help='Minimum fraction of step to stop')
    parser.add_argument('--maxconv', default=2, type=int, help='Times of true convergence that makes a stop')
    parser.add_argument('--featrm', default='auto', type=str, help='Remove features')

    opt = parser.parse_args()

    if opt.layer != "":
        layers = list(map(int, opt.layer.split(',')))
    else:
        layers = []

    logger = logging.getLogger('altp')
    logger.setLevel(logging.INFO)
    flog = logging.FileHandler(opt.output + 'log.txt', mode='w')
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
    datax, datay, data_names = example.load_altp(filename=opt.input, target=opt.obj, load_names=True,
                                                 featrm=featrm, encoder=opt.encoder)

    logger.info('Selecting data...')
    selector = preprocessing.Selector(datax, datay, data_names)
    if opt.partition:
        logger.info('Loading partition file %s' % opt.partition)
        selector.load(opt.partition)
    else:
        logger.warning("Partition file not found. Using auto-parition instead.")
        selector.partition(0.7, 0.2)
        selector.save(opt.output + 'part.txt')
    trainx, trainy, trainname = selector.training_set()
    testx, testy, testname = selector.validation_set()

    logger.info('Training size =%d, Testing size=%d' % (len(trainx), len(testx)))
    logger.info('X input example: (size=%d) %s' % (len(trainx[0]), ','.join(map(str, trainx[0]))))
    logger.info('Y input example: (size=%d) %s' % (len(trainy[0]), ','.join(map(str, trainy[0]))))
    logger.info('Normalizing...')
    scaler = preprocessing.Scaler()
    scaler.fit(trainx)
    scaler.save(opt.output + 'scale.txt')
    normed_trainx = scaler.transform(trainx)
    normed_testx = scaler.transform(testx)

    logger.info('Building network...')
    fitting.backend = 'tch'

    logger.info('Hidden layers = %r' % layers)
    logger.info('Initial learning rate = %f' % opt.lr)
    logger.info('L2 penalty = %f' % opt.l2)
    logger.info('Total %d epochs' % opt.epoch)
    logger.info('Batch = (%d values x %d step)' % (opt.batch, opt.step))
    if opt.gpu:
        logger.info('Using GPU acceleration')

    nn = fitting.PerceptronFitter(len(trainx[0]), len(trainy[0]), layers, batch_size=opt.batch, batch_step=opt.step,
                                  args_opt={'optimizer'   : torch.optim.Adam,
                                            'lr'          : opt.lr,
                                            'weight_decay': opt.l2
                                            }
                                  )

    nn.regressor.is_gpu = opt.gpu != 0
    nn.regressor.init_session()
    nn.regressor.load_data(normed_trainx, trainy)

    logger.info('Optimizer = %s' % type(nn.regressor.optimizer))

    errlogger = open(opt.output + 'err.log', 'w')

    err_head = 'step\tLoss\tMARE\tMSE\tACC1%\tACC2%\tACC10%'
    logger.info(err_head)
    errlogger.write(err_head + '\n')

    mse_history = []
    converge_times = 0
    best_last_score = None
    model_saved = False

    for i in range(opt.epoch):

        step, loss = nn.regressor.fit_epoch(0)

        if (i + 1) % 10 == 0:
            predy = nn.predict_batch(normed_testx)
            err_line = '%d\t%.3e\t%.3f\t%.3e\t%.2f\t%.2f\t%.2f' % (
                step * (i + 1),
                loss,
                metrics.mean_relative_error(testy, predy) * 100,
                metrics.mean_squared_error(testy, predy),
                metrics.accuracy(testy, predy, 0.01) * 100,
                metrics.accuracy(testy, predy, 0.02) * 100,
                metrics.accuracy(testy, predy, 0.10) * 100)

            logger.info(err_line)
            errlogger.write(err_line + '\n')
            errlogger.flush()

        if opt.check != 0:

            if (i + 1) % opt.check == 0:  # check convergence
                predy = nn.predict_batch(normed_testx)
                mse_history.append(metrics.mean_squared_error(testy, predy))

                if i > opt.epoch * opt.minstop:
                    # check stop
                    conv, cur_conv = validation.is_converge(np.array(mse_history), nskip=25)

                    if conv:
                        logger.info('Model converge detected at epoch %d' % i)
                        converge_times += 1

                    if converge_times >= opt.maxconv and cur_conv:
                        logger.info('Model converged at epoch: %d' % i)
                        break

            elif (i + 1) > opt.epoch - opt.check:  # save best one in last frames
                predy = nn.predict_batch(normed_testx)
                score = metrics.mean_squared_error(testy, predy)
                if best_last_score and best_last_score < score:
                    logger.info('Saving best model in advance: Epoch %d' % i)
                    nn.regressor.save(opt.output + 'model.pt')
                    model_saved = True

                best_last_score = min(best_last_score, score) if best_last_score else score

    errlogger.close()

    if mse_history:
        conv, cur_conv = validation.is_converge(np.array(mse_history))
        if not conv:
            logger.warning('Model not converged.')
        elif not cur_conv:
            logger.warning('Model converged, but the model saved may not be the best.')

    logger.info('MARE=%f%%\nMax ARE=%f%%\nAccuracy(1%%)=%f%%\nAccuracy(2%%)=%f%%\nAccuracy(10%%)=%f%%' % (
        metrics.mean_relative_error(testy, predy) * 100,
        metrics.max_relative_error(testy, predy) * 100,
        metrics.accuracy(testy, predy, 0.01) * 100,
        metrics.accuracy(testy, predy, 0.02) * 100,
        metrics.accuracy(testy, predy, 0.1) * 100))

    if not model_saved:
        nn.regressor.save(opt.output + 'model.pt')

    testy = testy[:, 0]
    predy = nn.predict_batch(normed_testx)[:, 0]
    visualizer = visualize.LinearVisualizer(trainy[:, 0], nn.predict_batch(normed_trainx)[:, 0], trainname, 'train')
    visualizer.append(testy, predy, testname, 'test')
    visualizer.dump(opt.output + 'fit.txt')

    if opt.visual:
        visualizer.scatter_yy(annotate_threshold=0.1, marker='x', lw=0.2, s=5)
        visualizer.scatter_error(annotate_threshold=0.1, marker='x', lw=0.2, s=5)
        visualizer.hist_error(label='test', histtype='step', bins=50)

        plt.show()


if __name__ == '__main__':
    main()
