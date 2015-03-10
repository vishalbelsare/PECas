#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the ODE's setup mehthod (collocation struct builder)

import casadi as ca
import pylab as pl
import pecas

import unittest

class BSPERunTest(object):

    def test_lsq_run(self):

        # Run parameter estimation and assure that the results is correct

        lsqpe = pecas.LSq(pesetup = self.bssetup, yN = self.yN, \
            stdyN = self.stdyN)

        lsqpe.run_parameter_estimation()
        phat = self.bssetup.V()(lsqpe.Vhat)["P"]
        print phat

        for k, pk in enumerate(phat):
            self.assertAlmostEqual(pk, self.phat[k], places = 5)

class ODEPERunTest(object):

    def test_lsq_run(self):

        # Run parameter estimation and assure that the results is correct

        lsqpe = pecas.LSq(pesetup = self.odesetup, yN = self.yN, \
            stdyN = self.stdyN, stds = self.stds)

        lsqpe.run_parameter_estimation()
        phat = self.odesetup.V()(lsqpe.Vhat)["P"]
        print phat

        for k, pk in enumerate(phat):
            self.assertAlmostEqual(pk, self.phat[k], places = 5)
