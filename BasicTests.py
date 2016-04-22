#!/usr/bin/env python

import sys
import os

import numpy as np
import numpy.lib.recfunctions as rec

import sklearn.mixture as mixture
import sklearn.neighbors as neighbors

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

def GMMFit(data):
    #gmm = mixture.VBGMM(n_components=10).fit(data)
    gmm = mixture.DPGMM(n_components=50, covariance_type='full').fit(data)
    #gmm = mixture.GMM(n_components=50, covariance_type='full').fit(data)
    return gmm

def KDEFit(data, bandwidth='thumb'):
    if bandwidth=='thumb':
        #bandwidth = np.power( (len(data)*(data.shape[-1]+2.0)/4.0), -1.0/(data.shape[-1]+4.0) )
        bandwidth = np.power( (len(data)*(data.shape[-1]+2.0)/4.0), -1.0/(data.shape[-1]+4.0) ) * 0.5
    kde = neighbors.KernelDensity(bandwidth=bandwidth).fit(data)
    return kde


if __name__ == "__main__":
    suchyta_utils.plot.Setup()

    npoints = 100000
    nsample = 1000000

    start = 15
    range = 12
    tmag = np.random.power(2, size=npoints)*range + start
    bias = 0.15 * (tmag-start)/range
    gamma = 0.45 * (tmag-start)/range
    #window = -2.0/np.pi * np.arctan( (tmag-30) )
    window = 1 - np.exp(tmag-25)
    window[window < 0] = 0
    mmag = AnalyticallyTransformed1D(tmag, f='cauchy', params=np.array([bias,gamma]))
    tm = np.dstack( (tmag,mmag) )[0]
    det = (window > 0)

    bins = np.arange(15, 27.01, 0.05)
    cent = (bins[1:]+bins[:-1])/2.0
    t, b = np.histogram(tmag, bins=bins, density=False)
    m1, b = np.histogram(mmag, bins=bins, density=False)
    m2, b = np.histogram(mmag, bins=bins, density=False, weights=window)

    tfit = GMMFit( np.reshape(tmag,(len(tmag),1)) )
    #tfit = KDEFit( np.reshape(tmag,(len(tmag),1)) )
    tsample = tfit.sample(n_samples=nsample)
    tf, b = np.histogram(tsample.reshape((len(tsample),)), bins=bins, density=False)

    tmfit = GMMFit(tm[det])
    #tmfit = KDEFit(tm[det])
    #tmsample = tmfit.sample(n_samples=np.sum(det))
    tmsample = tmfit.sample(n_samples=nsample)
    tmf, bx, by = np.histogram2d(tmsample[:,0], tmsample[:,1], bins=[bins,bins])

    fig, ax = plt.subplots(1,1, tight_layout=True)
    ax.axvline(x=25, color='black', ls='dashed')
    ax.plot(cent, t/float(len(tmag)), color='black')
    #ax.plot(cent, m1, color='red')
    #ax.plot(cent, m2, color='blue')
    ax.plot(cent, tf/float(len(tsample)), color='green')

    fig, ax = plt.subplots(1,1, tight_layout=True)
    h, bx, by = np.histogram2d(tmag[det], mmag[det], bins=[bins,bins])
    vmax = np.amax(h.flatten()/len(tmag))
    im = ax.imshow(h.T/len(tmag), interpolation='nearest', origin='lower', extent=[bins[0],bins[-1],bins[0],bins[-1]], vmin=0, vmax=vmax)
    suchyta_utils.plot.AddColorbar(ax, im)

    fig, ax = plt.subplots(1,1, tight_layout=True)
    im = ax.imshow(tmf.T/len(tmsample), interpolation='nearest', origin='lower', extent=[bins[0],bins[-1],bins[0],bins[-1]], vmin=0, vmax=vmax)
    suchyta_utils.plot.AddColorbar(ax, im)

    plt.show()
