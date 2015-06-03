#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the ODE's setup mehthod (collocation struct builder)

import casadi as ca
import pylab as pl
import pecas

import unittest

class BSPESetupTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(pesetup = self.bssetup, yN = self.yN, wv = self.wv)
        pecas.LSq(pesetup = self.bssetup, yN = self.yN.T, wv = self.wv)
        pecas.LSq(pesetup = self.bssetup, yN = self.yN, wv = self.wv.T)

    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, pesetup = self.bssetup, \
            yN = pl.atleast_2d(self.yN)[:, :-1], wv = self.wv)
        self.assertRaises(ValueError, pecas.LSq, pesetup = self.bssetup, \
            yN = self.yN, wv = pl.atleast_2d(self.wv)[:-1])


class ODEPESetupTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = self.wwe)
        pecas.LSq(pesetup = self.odesetup, yN = self.yN.T, wv = self.wv, \
            wwe = self.wwe)
        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv.T, \
            wwe = self.wwe)
        
        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = self.wwe)
        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = [self.wwe])
        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = pl.atleast_1d(self.wwe))

        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = self.wwe, wwu = self.wwu)
        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = self.wwe, wwu = [self.wwu])
        pecas.LSq(pesetup = self.odesetup, yN = self.yN, wv = self.wv, \
            wwe = self.wwe, wwu = pl.atleast_1d(self.wwu))

    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, pesetup = self.odesetup, \
            yN = pl.atleast_2d(self.yN)[:, :-1], wv = self.wv)
        self.assertRaises(ValueError, pecas.LSq, pesetup = self.odesetup, \
            yN = self.yN, wv = pl.atleast_2d(self.wv)[:-1])
        # self.assertRaises(ValueError, pecas.LSq, pesetup = self.odesetup, \
        #     yN = self.yN, wv = self.wv, wwe = pl.asarray([1, 2, 3]))
