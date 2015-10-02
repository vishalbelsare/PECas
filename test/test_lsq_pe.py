#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import testing

import casadi as ca
import pecas

import unittest

class BSLsqPETest(object):

    def lsq_run(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.lsqpe.run_parameter_estimation()

        phat = self.lsqpe.phat

        testing.assert_almost_equal(phat, self.phat, decimal = 5)

        self.lsqpe.show_results()


    def comp_covmat(self):

        # Run computation of the covariance matrix for the estimated parameters

        self.assertRaises(AttributeError, getattr, self.lsqpe, "Cvox")

        self.lsqpe.compute_covariance_matrix()

        self.lsqpe.show_results()


    def test_pe(self):

        self.lsq_run()

        # self.comp_covmat()


class ODELsqPETest(object):

    def lsq_run(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = self.wwe, wwu = self.wwu)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.lsqpe.run_parameter_estimation()

        phat = self.lsqpe.phat
        Xhat = self.lsqpe.Xhat

        testing.assert_almost_equal(phat, self.phat, decimal = 5)

        self.lsqpe.show_results()


    def comp_covmat(self):

        # Run computation of the covariance matrix for the estimated parameters

        self.assertRaises(AttributeError, getattr, self.lsqpe, "Cvox")

        self.lsqpe.compute_covariance_matrix()

        self.lsqpe.show_results()


    def test_pe(self):

        self.lsq_run()

        self.comp_covmat()
