#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from MDLearn import example, preprocessing, metrics

from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, normalize

parser = argparse.ArgumentParser(description='RFE selector')
parser.add_argument('--output', default='rfe.txt', help='Output file')
parser.add_argument('--obj', default='Density', help='Fitting object')
parser.add_argument('--end', type=int, default=5, help='Minimum feature to stop')
parser.add_argument('--partition', type=str, help='Partition cache file')

opt = parser.parse_args()

datax, datay = example.load_altp(opt.obj)
selector = preprocessing.Selector(datax, datay)
selector.load(opt.partition)
trainx, trainy = selector.training_set()
ymax = np.max(np.abs(trainy))
trainy = trainy.flatten() / ymax

scaler = StandardScaler()
scaler.fit(trainx)
trainx = scaler.transform(trainx)

n_feature = len(trainx[0])

svr = SVR(kernel='linear')
svr.fit(trainx, trainy)

mse = mean_squared_error(trainy, svr.predict(trainx))
print(mse)

pool = []

output = open(opt.output, 'w')
output.write('STEP\tRMFEA\tMSE\tFEATURES\n')
output.write('0\tfull\t%.4e\t[FULL]\n' % mse)

step = 1
while n_feature > opt.end:

    svr = SVR(kernel='linear')
    rfe = RFE(svr, n_feature - 1, step=1)

    rfe.fit(trainx, trainy)
    print(rfe.ranking_)

    rmidx = np.where(rfe.ranking_ == 2)[0][0]
    iflag = False

    for i in range(len(pool)):
        if pool[i] <= rmidx:
            rmidx += 1
        else:
            iflag = True
            pool.insert(i, rmidx)
            break

    if not iflag:
        pool.append(rmidx)

    print('Removed feature:', rmidx)
    mse = mean_squared_error(trainy, rfe.predict(trainx))
    print(mse)
    output.write('%d\t%d\t%.4e\t%s\n' % (step, rmidx, mse, ','.join(map(str, pool))))

    trainx = rfe.transform(trainx)

    n_feature -= 1
    step += 1

output.close()
