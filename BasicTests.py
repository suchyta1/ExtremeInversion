#!/usr/bin/env python

import sys
import os
import numpy as np
import numpy.lib.recfunctions as rec

import matplotlib.pyplot as plt
import suchyta_utils.plot


def AnalyticallyTransformed1D(data, f=None, params=None):
    if f is None:
        raise Exception('Must give an f')
    if params is None:
        raise Exception('Must give params')

    if f.lower()=='cauchy':
        '''
        params[0, :] = x0
        params[1, :] = gamma
        '''
        return data + np.random.standard_cauchy(size=len(data))*params[1,:] + params[0,:]



if __name__ == "__main__":
    suchyta_utils.plot.Setup()

    start = 15
    range = 10
    tmag = np.random.power(2, size=1000000)*range + start
    bias = 0.15 * (tmag-start)/range
    gamma = 0.3 * (tmag-start)/range
    mmag = AnalyticallyTransformed1D(tmag, f='cauchy', params=np.array([bias,gamma]))

    bins = np.arange(15, 24, 0.05)
    cent = (bins[1:]+bins[:-1])/2.0
    t, b = np.histogram(tmag, bins=bins, density=False)
    m, b = np.histogram(mmag, bins=bins, density=False)
    fig, ax = plt.subplots(1,1, tight_layout=True)
    ax.plot(cent, t, color='black')
    ax.plot(cent, m, color='red')
    plt.show()
