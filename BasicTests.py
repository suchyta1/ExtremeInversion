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
    gmm = mixture.GMM(n_components=10).fit(data)
    return gmm

def KDEFit(data, bandwidth='thumb'):
    if bandwidth=='thumb':
        bandwidth = np.power( (len(data)*(data.shape[-1]+2.0)/4.0), -1.0/(data.shape[-1]+4.0) ) 
    kde = neighbors.KernelDensity(bandwidth=bandwidth).fit(data)
    return kde


if __name__ == "__main__":
    suchyta_utils.plot.Setup()

    start = 15
    range = 12
    tmag = np.random.power(2, size=1000000)*range + start
    bias = 0.15 * (tmag-start)/range
    gamma = 0.45 * (tmag-start)/range
    #window = -2.0/np.pi * np.arctan( (tmag-30) )
    window = 1 - np.exp(tmag-25)
    window[window < 0] = 0
    mmag = AnalyticallyTransformed1D(tmag, f='cauchy', params=np.array([bias,gamma]))
    tm = np.dstack( (tmag,mmag) )[0]

    bins = np.arange(15, 27.01, 0.05)
    cent = (bins[1:]+bins[:-1])/2.0
    t, b = np.histogram(tmag, bins=bins, density=False)
    m1, b = np.histogram(mmag, bins=bins, density=False)
    m2, b = np.histogram(mmag, bins=bins, density=False, weights=window)

    fig, ax = plt.subplots(1,1, tight_layout=True)
    ax.axvline(x=25, color='black', ls='dashed')
    ax.plot(cent, t, color='black')
    #ax.plot(cent, m1, color='red')
    #ax.plot(cent, m2, color='blue')

    #tgmm = GMMFit( np.reshape(tmag,(len(tmag),1)) )
    tgmm = KDEFit( np.reshape(tmag,(len(tmag),1)) )
    tfit = tgmm.sample(n_samples=1000000)
    tf, b = np.histogram(tfit, bins=bins, density=False)
    ax.plot(cent, tf, color='green')

    fig, ax = plt.subplots(1,1, tight_layout=True)
    h, bx, by = np.histogram2d(tmag, mmag, bins=[bins,bins])
    im = ax.imshow(h.T, interpolation='nearest', origin='lower', extent=[bins[0],bins[-1],bins[0],bins[-1]])
    suchyta_utils.plot.AddColorbar(ax, im)

    '''
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    '''

    #ax4.set_title('Title of ax4')
    #im4 = ax4.imshow(tot2, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
    #divider4 = make_axes_locatable(ax4)
    #cax4 = divider4.append_axes("right", size="20%", pad=0.05)
    #cbar4 = plt.colorbar(im4, cax=cax4)

    plt.show()
