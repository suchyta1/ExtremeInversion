#!/usr/bin/env python

import sys
import os
import numpy as np
import numpy.lib.recfunctions as rec



class EEDC(object):
    '''
    GMMs are shape (Ncomps, 3), where the 3 is for [amp, mean, cov]
    '''
    
    def __init__(self, data, likelihood=None, guesses=None):
        self.data = data
        self.likelihood = None
        self.guesses = None

        if likelihood is not None:
            self.SetLikelihood(likelihood)
        if guesses is not None:
            self.SetGuesses(guesses)


    def _ArrCheck(self, thing, s, ss):
        try:
            len(thing)
        except:
            raise Exception('%s part of %s must be an array'%(ss, s))

    def _CheckShape(self, thing, s):
        try:
            thing = np.array(thing)
        except:
            raise Exception("%s must be a (3,k,-) array"%(s))

        if thing.shape[0] != 3:
            raise Exception("First axis of %s must be lenth 3, (amps, means, covs)"%(s))
        ss = ['amps','means','covs']
        for i in range(len(thing)):
            self._ArrCheck(thing[i],s,ss[i])
        if (len(thing[0])!=len(thing[1])):
            raise Exception("len(%s) component of %s != len(%s) component of %s"%(ss[0],s,ss[1],s))
        if (len(thing[1])!=len(thing[2])):
            raise Exception("len(%s) component of %s != len(%s) component of %s"%(ss[1],s,ss[2],s))

        return thing

    def _CheckNorm(self, thing, s):
        if np.abs(thing-1.0) > 0.001:
            raise Exception("%s amplitudes must sum to 1."%(s))

    def _CheckGuesses(self, guesses):
        guesses = self._CheckShape(guesses, 'guesses')
        s = np.sum(guesses[:,0])
        self._CheckNorm(s, 'guesses')
        return guesses
    
    def _CheckLikelihood(self, like):
        like = self._CheckShape(like)


    def SetLikelihood(self, like):
        self.likelihood = self._CheckLikelihood(like)

    def SetGuesses(self, guess):
        self.guesses = self._CheckGuesses(guess)

    
    def _Nij(self):
        diff = np.reshape(data,(len(data),1,data.shape[-1])) - np.reshape(self.guesses[1,:],(1,len(self.guesses[1]),self.guesses[1].shape[-1]))
        cov_diff = np.einsum('...jk,...k',self.guess[2],diff)
        dcd = np.einsum('...k,...k',diff,cov_diff)


    def RunEM(self, guesses=None, likelihood=None):
        if guesses is not None:
            self.SetGuesses(guesses)
        if likelihood is not None:
            self.Likelihood = self.SetLikelihood(likelihood)


if __name__ == "__main__":

    d = np.reshape( np.arange(100), (100,1) )
    eedc = EEDC(d)
