#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the ODE's setup mehthod (collocation struct builder)

import casadi as ca
import pecas

import unittest

class BSPERunTest(object):

    def lsq_run(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, u = self.uN, \
            pmin = self.pmin, pmax = self.pmax, pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.lsqpe.run_parameter_estimation()

        phat = self.lsqpe.phat
        print(phat)

        for k, pk in enumerate(phat):
            self.assertAlmostEqual(pk, self.phat[k], places = 5)

        self.lsqpe.show_system_information(showEquations = True)
        self.lsqpe.show_results()


    def comp_covmat(self):

        # Run computation of the covariance matrix for the estimated parameters

        self.assertRaises(AttributeError, getattr, self.lsqpe, "Cvox")

        self.lsqpe.compute_covariance_matrix()

        self.lsqpe.show_results()


    def test_pe(self):

        self.lsq_run()

        self.comp_covmat()


class ODEPERunTest(object):

    def lsq_run(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, u = self.uN, \
            pmin = self.pmin, pmax = self.pmax, pinit = self.pinit, \
            xmin = self.xmin, xmax = self.xmax, xinit = self.xinit, \
            x0min = self.x0max, x0max = self.x0max, \
            xNmin = self.xNmin, xNmax = self.xNmax, \
            yN = self.yN, \
            wv = self.wv, wwe = self.wwe, wwu = self.wwu)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.lsqpe.run_parameter_estimation()

        phat = self.lsqpe.phat
        print(phat)
        
        Xhat = self.lsqpe.Xhat
        print(Xhat)

        for k, pk in enumerate(phat):
            self.assertAlmostEqual(pk, self.phat[k], places = 5)

        self.lsqpe.show_system_information(showEquations = True)
        self.lsqpe.show_results()


    def comp_covmat(self):

        # Run computation of the covariance matrix for the estimated parameters

        self.assertRaises(AttributeError, getattr, self.lsqpe, "Cvox")

        self.lsqpe.compute_covariance_matrix()

        self.lsqpe.show_results()


    def test_pe(self):

        self.lsq_run()

        self.comp_covmat()
