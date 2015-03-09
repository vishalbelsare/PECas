#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the ODE's setup mehthod (collocation struct builder)

import casadi as ca
import pylab as pl
import pecas

import unittest

class PESetupTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(pesetup = self.odesol, yN = self.yN, stdyN = self.stdyN)
        pecas.LSq(pesetup = self.odesol, yN = self.yN.T, stdyN = self.stdyN)
        pecas.LSq(pesetup = self.odesol, yN = self.yN, stdyN = self.stdyN.T)
        
        pecas.LSq(pesetup = self.odesol, yN = self.yN, stdyN = self.stdyN, \
            stds = self.stds)
        pecas.LSq(pesetup = self.odesol, yN = self.yN, stdyN = self.stdyN, \
            stds = [self.stds])
        pecas.LSq(pesetup = self.odesol, yN = self.yN, stdyN = self.stdyN, \
            stds = pl.atleast_1d(self.stds))

    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, pesetup = self.odesol, \
            yN = pl.atleast_2d(self.yN)[:, :-1], stdyN = self.stdyN)
        self.assertRaises(ValueError, pecas.LSq, pesetup = self.odesol, \
            yN = self.yN, stdyN = pl.atleast_2d(self.stdyN)[:-1])
        self.assertRaises(ValueError, pecas.LSq, pesetup = self.odesol, \
            yN = self.yN, stdyN = self.stdyN, stds = [1, 2])
