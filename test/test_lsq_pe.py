#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import testing

import casadi as ca
import pecas

import unittest

class BSLsqPETest(object):

    def lsq_run_exact_hessian(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.assertRaises(AttributeError, self.lsqpe.show_results)

        self.lsqpe.run_parameter_estimation(hessian = "exact-hessian")

        phat = self.lsqpe.phat

        testing.assert_almost_equal(phat, self.phat, decimal = 5)

        self.lsqpe.show_results()


    def lsq_run_gauss_newton(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.assertRaises(AttributeError, self.lsqpe.show_results)

        self.lsqpe.run_parameter_estimation(hessian = "gauss-newton")

        phat = self.lsqpe.phat

        testing.assert_almost_equal(phat, self.phat, decimal = 5)

        self.lsqpe.show_results()


    def comp_covmat_valid(self):

        # Run computation of the covariance matrix for the estimated parameters

        self.assertRaises(AttributeError, getattr, self.lsqpe, "Cvox")

        self.lsqpe.compute_covariance_matrix()

        self.lsqpe.show_results()


    def test_pe_exact_hessian(self):

        self.lsq_run_exact_hessian()

        # self.comp_covmat_valid()


    def test_covmat_invalid(self):

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(AttributeError, self.lsqpe.compute_covariance_matrix)


    def test_pe_gauss_newton(self):

        self.lsq_run_gauss_newton()

        # self.comp_covmat_valid()


    def test_pe_invalid_method(self):

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(NotImplementedError, \
            self.lsqpe.run_parameter_estimation, hessian = "dummy")


class ODELsqPETest(object):

    def lsq_run_exact_hessian(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.assertRaises(AttributeError, self.lsqpe.show_results)

        self.lsqpe.run_parameter_estimation(hessian = "exact-hessian")

        phat = self.lsqpe.phat
        Xhat = self.lsqpe.Xhat

        testing.assert_almost_equal(phat, self.phat, decimal = 5)

        self.lsqpe.show_results()


    def lsq_run_gauss_newton(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(AttributeError, getattr, self.lsqpe, "phat")
        self.assertRaises(AttributeError, getattr, self.lsqpe, "Xhat")

        self.assertRaises(AttributeError, self.lsqpe.show_results)

        self.lsqpe.run_parameter_estimation(hessian = "gauss-newton")

        phat = self.lsqpe.phat
        Xhat = self.lsqpe.Xhat

        testing.assert_almost_equal(phat, self.phat, decimal = 5)

        self.lsqpe.show_results()


    def comp_covmat_valid(self):

        # Run computation of the covariance matrix for the estimated parameters

        self.assertRaises(AttributeError, getattr, self.lsqpe, "Cvox")

        self.lsqpe.compute_covariance_matrix()

        self.lsqpe.show_results()


    def test_covmat_invalid(self):

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(AttributeError, self.lsqpe.compute_covariance_matrix)


    def test_pe_exact_hessian(self):

        self.lsq_run_exact_hessian()

        self.comp_covmat_valid()

    
    def test_pe_gauss_newton(self):

        self.lsq_run_gauss_newton()

        self.comp_covmat_valid()


    def test_pe_invalid_method(self):

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(NotImplementedError, \
            self.lsqpe.run_parameter_estimation, hessian = "dummy")
