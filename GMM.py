#!/usr/bin/env python

import sys
import os
import time
import datetime

import numpy as np
import numpy.lib.recfunctions as rec
import sklearn.mixture as mixture
import scipy.linalg


class GMMException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class MinimalGMM(Object):
    def __init__(self):
        self.n_components = -1


def NoneArrays(gmm):
    gmm.weights_ = None
    gmm.means_ = None
    gmm.covars_ = None
    gmm.icovars_ = None


def class GMM(object):

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
            raise Exception('only allowed to set one of covars or icovars')
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


def SumOfTwoSquaredFormes(gmm1, gmm2):
    icovars = gmm1._icovars.reshape((gmm1.icovars_.shape[0],1,gmm1.icovars_.shape[1],gmm1.icovars_.shape[2])) + gmm2._icovars.reshape((1,gmm2.icovars_.shape[0],gmm2.icovars_.shape[1],gmm2.icovars_.shape[2])) 

    icov = scipy.linalg.block_diag( *np.reshape(icovars,(icovars.shape[0]*icovars.shape[1],icovars.shape[2],icovars.shape[3])) )
    cov = np.linalg.inv(icov)
    elems = np.arange(icovars.shape[0]*icovars.shape[1])
    covs = np.zeros( (icovars.shape[0],icovars.shape[1],icovars.shape[2],icovars.shape[3]) )
    
    ix = np.ix_()
    covs = 
