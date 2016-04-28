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
import GMM


class XXD(object):
    
    def __init__(self, tgmm, lgmm):
        self.truth = tgmm
        self.likelihood = lgmm
        self._CheckD()

    def _CheckD(self):
        if (self.truth.means_.shape[-1]!=self.likelihood.means_.shape[-1]):
            raise GMM.GMMException('given truth and likelihood GMMs are not the same dimensionality')

    def __mul__(self, other):
        pass

    def SumOfTwoSquaredForms(self):
        ncomps1 = self.truth.means_.shape[0]
        ncomps2 = self.likelihood.means_.shape[0]
        ncomps12 = ncomps1 * ncomps2
        dim = self.truth.means_.shape[-1]

        icovars = self.truth.icovars_.reshape((ncomps1,1,dim,dim)) + self.likelihood.icovars_.reshape((1,ncomps2,dim,dim)) 
        icov = scipy.linalg.block_diag( *icovars.reshape((ncomps12,dim,dim)) )
        cov = np.linalg.inv(icov)
        row, col = _BlockIndexes(ncomps12,dim)
        covars = cov[ row, col ]
        covars = covars.reshape( (ncomps1,ncomps2,dim,dim) )
        
        m1 = np.dot(scipy.linalg.block_diag(*self.truth.icovars_), self.truth.means_.flatten()).reshape((ncomps1,1,dim))
        m2 = np.dot(scipy.linalg.block_diag(*self.likelihood.icovars_), self.likelihood.means_.flatten()).reshape((1,ncomps2,dim))
        m = m1 + m2
        mean = np.dot(cov, m.reshape(ncomps12,dim))
        means = mean.reshape((ncomps1,ncomps2,dim))

        mm = mean * np.dot(cov,mean)
        mm = mm.reshape(ncomps1, ncomps2, dim)
        mm = np.sum(mm, axis=-1)
        c1 = np.dot(scipy.linalg.block_diag(*self.truth.icovars_),self.truth.means_.flatten()).reshape(ncomps1,dim)
        c1 = np.sum(self.truth.means_*c1, axis=-1)
        c1 = c1.reshape(c1.shape[0],1)
        c2 = np.dot(scipy.linalg.block_diag(*self.likelihood.icovars_),self.likelihood.means_.flatten()).reshape(ncomps2,dim)
        n2 = np.sum(self.likelihood.means_*c2, axis=-1)
        c2 = c2.reshape(1,c2.shape[0])
        cc = c1 + c2
        consts = 0.5 * (mm - cc)

        return [consts, means, covars, icovars]


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




if __name__=='__main__':
    print 'test'
