#!/usr/bin/env python

import sys
import os
import time
import datetime

import numpy as np
import numpy.lib.recfunctions as rec
import sklearn.mixture as mixture
import scipy.linalg

import matplotlib.pyplot as plt
import suchyta_utils.plot as esplt


class GMMException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class MinimalGMM(object):
    def __init__(self):
        self.n_components = -1

def TestGMM(mean=0, cov=1.0):
    #gmm = MinimalGMM()
    gmm = mixture.GMM(n_components=1, covariance_type='full')
    gmm.means_ = np.ones( (1,1) )*mean
    gmm.covars_ = np.array( [np.identity(1)*cov] )
    gmm.icovars_ = 1.0/gmm.covars_
    gmm.weights_ = np.ones( (1,) )
    return gmm


def NoneArrays(gmm):
    gmm.weights_ = None
    gmm.means_ = None
    gmm.covars_ = None
    gmm.icovars_ = None


class GMM(object):

    def _log(self):
        rootlog = logging.getLogger()
        rootlog.setLevel(logging.NOTSET)
        t = int(time.time()) + datetime.datetime.now().microsecond
        self.log = logging.getLogger('id-%i'%(t))
        self.log.setLevel(logging.DEBUG)

    def _CheckArray(self, w, dim, s):
        msg = '%s must be a %i-dimensional array' %(s,dim)
        if type(w)!=np.ndarray:
            try: w = np.array(w)
            except: raise GMMException(msg)
        if w.ndim!=dim:
            raise GMMException(msg)
        return w

    def SetWeights(self, w):
        w = _CheckArray(w, 1, 'weights')
        self.gmm.weights_ = w

    def SetMeans(self, m):
        m = _CheckArray(w, 2, 'means')
        self.gmm.means_ = m

    def _Inv(self, k1, inv):
        self.gmm.__dict__[k1] = np.zeros(inv.shape)
        for i in range(len(self.gmm.__dict__[k1])):
            self.gmm.__dict__[k1][i] = np.linalg.inv(inv[i])

    def SetCovars(self, c, doinv=True):
        c = _CheckArray(c, 3, 'covars')
        self.gmm.covars_ = c
        if doinv:
            self.Inv('icovars_', self.gmm.covars_)

    def SetICovars(self, i, docov=True):
        i = _CheckArray(i, 3, 'icovars')
        self.gmm.icovars_ = i
        if docov:
            self.Inv('covars_', self.gmm.icovars_)


    def __init__(self, weights=None, means=None, covars=None, icovars=None, sklearnfit={}):
        self._log()

        self.gmm = MinimalGMM()
        if sklearnfit != {}:
            self.gmm = mixture.GMM(**sklearnfit)
        self.gmm.NoneArrays()
        
        if (covars is not None) and (icovars is not None):
            raise GMMException('only allowed to set one of covars or icovars')
        if weights is not None:
            self.SetWeights(w)
        if means is not None:
            self.SetMeans(means)
        if covars is not None:
            self.SetCovars(covars, doinv=True)
        if icovars is not None:
            self.SetICovars(icovars, docov=True)

    def __mul__(self, other):
        pass

def _CheckD(gmm1, gmm2):
    if (gmm1.means_.shape[-1]!=gmm2.means_.shape[-1]):
        raise GMMException('given GMMs are not the same dimensionality')


def _BlockIndexes(d12, d3):
    bsize = d3*d3
    dstart = np.arange(d12)
    start = np.repeat( dstart*d3,  bsize)
    start = start.reshape( (start.shape[0]/bsize, d3, d3) )

    rowoffset = np.repeat(np.arange(d3), d3).reshape(1,d3,d3)
    row = start + rowoffset
    coloffset = np.tile(np.arange(d3), d3).reshape(1,d3,d3)
    col = start + coloffset
    return [row, col]


def SumOfTwoSquaredForms(gmm1, gmm2):
    _CheckD(gmm1, gmm2)
    ncomps1 = gmm1.means_.shape[0]
    ncomps2 = gmm2.means_.shape[0]
    ncomps12 = ncomps1 * ncomps2
    dim = gmm1.means_.shape[-1]

    icovars = gmm1.icovars_.reshape((ncomps1,1,dim,dim)) + gmm2.icovars_.reshape((1,ncomps2,dim,dim)) 
    icov = scipy.linalg.block_diag( *icovars.reshape((ncomps12,dim,dim)) )
    cov = np.linalg.inv(icov)
    row, col = _BlockIndexes(ncomps12,dim)
    covars = cov[ row, col ]
    covars = covars.reshape( (ncomps1,ncomps2,dim,dim) )
    
    m1 = np.dot(scipy.linalg.block_diag(*gmm1.icovars_), gmm1.means_.flatten()).reshape((ncomps1,1,dim))
    m2 = np.dot(scipy.linalg.block_diag(*gmm2.icovars_), gmm2.means_.flatten()).reshape((1,ncomps2,dim))
    m = m1 + m2
    mean = np.dot(cov, m.reshape(ncomps12,dim))
    means = mean.reshape((ncomps1,ncomps2,dim))

    mm = mean * np.dot(cov,mean)
    mm = mm.reshape(ncomps1, ncomps2, dim)
    mm = np.sum(mm, axis=-1)
    c1 = np.dot(scipy.linalg.block_diag(*gmm1.icovars_),gmm1.means_.flatten()).reshape(ncomps1,dim)
    c1 = np.sum(gmm1.means_*c1, axis=-1)
    c1 = c1.reshape(c1.shape[0],1)
    c2 = np.dot(scipy.linalg.block_diag(*gmm2.icovars_),gmm2.means_.flatten()).reshape(ncomps2,dim)
    n2 = np.sum(gmm2.means_*c2, axis=-1)
    c2 = c2.reshape(1,c1.shape[0])
    cc = c1 + c2
    consts = 0.5 * (mm - cc)

    return [consts, means, covars, icovars]


if __name__=='__main__':
    gmm1 = TestGMM(cov=1.0) 
    gmm2 = TestGMM(mean=2, cov=2.0)
    sosf = SumOfTwoSquaredForms(gmm1, gmm2)
    print sosf

    gmm4 = mixture.GMM(n_components=1, covariance_type='full')
    gmm4.means_ = sosf[1][0]
    gmm4.covars_ = sosf[2][0]
    #gmm4.icovars_ = sosf[3]
    gmm4.weights_ = np.array([1])

    s1 = gmm1.sample(n_samples=int(1e6))
    s2 = gmm2.sample(n_samples=int(1e6))
    s3 = s1*s2
    s4 = gmm4.sample(n_samples=int(1e6)) 
    
    bins = np.arange(-4, 6.01, 0.1)
    cent = (bins[1:]+bins[:-1])/2.0
    h1, b = np.histogram(s1, bins, density=True)
    h2, b = np.histogram(s2, bins, density=True)
    h3, b = np.histogram(s3, bins, density=True)
    h4, b = np.histogram(s4, bins, density=True)

    esplt.Setup()
    plt.plot(cent, h1, color='blue')
    plt.plot(cent, h2, color='red')
    plt.plot(cent, h3, color='green')
    plt.plot(cent, h4, color='black')
    plt.show()
