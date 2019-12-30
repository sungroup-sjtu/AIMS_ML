#!/usr/bin/env python3

import os
import numpy as np


def get_result_list(fn):
    last_info = None
    for line in open(fn, 'r').readlines()[-10:]:
        info = line.split()
        if len(info) == 12:
            last_info = info
    if last_info is None:
        return None
    else:
        info = last_info[5:6] + last_info[7:10]
        return np.array(list(map(float, info)))


for maxPath in [3, 4, 5, 6, 7]:
    result = np.zeros(4)
    for i in range(5):
        Dir = 't0-%i_repeat%i' % (maxPath, i)
        result += get_result_list(os.path.join(Dir, 'log.txt'))
    result /= 5
    print('topological-0-%i' % maxPath, result)

for mcut in [0, 50, 100, 150, 200]:
    result = np.zeros(4)
    for i in range(5):
        Dir = 'm%i_repeat%i' % (mcut, i)
        result += get_result_list(os.path.join(Dir, 'log.txt'))
    result /= 5
    print('morgan1-%i' % mcut, result)

for mcut in [0, 50, 100, 150, 200]:
    result = np.zeros(4)
    for i in range(5):
        Dir = 'm%i-s_repeat%i' % (mcut, i)
        result += get_result_list(os.path.join(Dir, 'log.txt'))
    result /= 5
    print('morgan1-%i-simple' % mcut, result)
