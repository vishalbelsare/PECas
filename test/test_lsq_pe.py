#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

from numpy import testing

import casadi as ca
import pecas

import unittest

from mock import patch

class NDLsqPETest(object):

    def lsq_run_exact_hessian(self):

        # Run parameter estimation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.ndsys, \
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

        self.lsqpe = pecas.LSq(system = self.ndsys, \
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

        # There is not yet a covariance computation for BasicSystem


    def test_covmat_invalid(self):

        self.lsqpe = pecas.LSq(system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(AttributeError, self.lsqpe.compute_covariance_matrix)


    @patch("matplotlib.pyplot.show")
    def plot_ellipsoid(self, mock_show):

        # mock_show.return_value = None

        # There is not yet a covariance computation for BasicSystem, 
        # so no ellipsoids can be drawn as well

        pass

    def test_pe_exact_hessian(self):

        self.lsq_run_exact_hessian()

        self.comp_covmat_valid()

        self.plot_ellipsoid()


    def test_pe_gauss_newton(self):

        self.lsq_run_gauss_newton()

        self.comp_covmat_valid()

        self.plot_ellipsoid()


    def test_pe_invalid_method(self):

        self.lsqpe = pecas.LSq(system = self.ndsys, \
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


    @patch("matplotlib.pyplot.show")
    def plot_ellipsoid(self, mock_show):

        mock_show.return_value = None

        if self.phat.size == 1:

            self.assertRaises(ValueError, \
                self.lsqpe.plot_confidence_ellipsoids)

        else:

            self.lsqpe.plot_confidence_ellipsoids()

            self.assertRaises(TypeError, \
                self.lsqpe.plot_confidence_ellipsoids, \
                indices = "dummy")

            self.assertRaises(TypeError, \
                self.lsqpe.plot_confidence_ellipsoids, \
                indices = ["1", "2"])


    def test_pe_exact_hessian(self):

        self.lsq_run_exact_hessian()

        self.comp_covmat_valid()

        self.plot_ellipsoid()

    
    def test_pe_gauss_newton(self):

        self.lsq_run_gauss_newton()

        self.comp_covmat_valid()

        self.plot_ellipsoid()


    def test_pe_invalid_method(self):

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(NotImplementedError, \
            self.lsqpe.run_parameter_estimation, hessian = "dummy")


    def test_plot_ellipsoid_invalid_call(self):

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(AttributeError, \
            self.lsqpe.plot_confidence_ellipsoids)
        