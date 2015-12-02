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
# Test the function for setting bounds and initials

import numpy as np
from numpy.testing import assert_array_equal

import casadi as ca
from pecas.setups.setupsbaseclass import SetupsBaseClass

import unittest
import mock

class FakeSetupsBaseClass(SetupsBaseClass):

    # Cf.: http://stackoverflow.com/questions/27105491/
    # how-can-i-unit-test-a-method-without-instantiating-the-class 

    def __init__(self, *args, **kwargs):

        pass


class CheckAndSetTimepoints(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()
        self.tp_ref = np.linspace(0, 49, 50)


    def test_input_list(self):

        tp = [k for k in range(50)]

        tp = self.fakesbc.check_and_set_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_onedim_time_vector(self):

        tp = np.linspace(0, 49, 50)

        tp = self.fakesbc.check_and_set_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_row_time_vector(self):

        tp = np.atleast_2d(np.linspace(0, 49, 50))

        tp = self.fakesbc.check_and_set_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_column_time_vector(self):

        tp = np.atleast_2d(np.linspace(0, 49, 50)).T

        tp = self.fakesbc.check_and_set_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_invalid_time_vector(self):

        tp = np.random.randn(2,2)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_time_points_input, tp)


class CheckAndSetControlTimepoints(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()
        # Cf. https://docs.python.org/3/library/unittest.mock.html#quick-guide        
        self.fakesbc.check_and_set_time_points_input = mock.MagicMock()

        self.tu_ref = np.linspace(0, 49, 50)

    def test_input_valid(self):

        # check_and_set_control_time_points_input passes it's input values to
        # check_and_set_time_points_input for checking, whose return value
        # is mocked here, which makes the input values for
        # check_and_set_control_time_points_input irrelevant for testing

        self.fakesbc.check_and_set_time_points_input.return_value = self.tu_ref
        self.fakesbc.check_and_set_control_time_points_input(None)
        assert_array_equal(self.fakesbc.tu, self.tu_ref)

    def test_input_invalid(self):

        self.fakesbc.check_and_set_time_points_input.side_effect = \
            ValueError
        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_control_time_points_input, None)


class CheckAndSetMeasurementsTimepoints(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()
        self.fakesbc.check_and_set_time_points_input = mock.MagicMock()

        self.tu_ref = np.linspace(0, 49, 50)
        self.ty_ref = np.linspace(50, 99, 50)


    def test_input_none(self):

        self.fakesbc.tu = self.tu_ref
        self.fakesbc.check_and_set_measurement_time_points_input(None)
        assert_array_equal(self.fakesbc.ty, self.tu_ref)


    def test_input_valid(self):

        self.fakesbc.check_and_set_time_points_input.return_value = self.ty_ref
        self.fakesbc.check_and_set_measurement_time_points_input(self.ty_ref)
        assert_array_equal(self.fakesbc.ty, self.ty_ref)


    def test_input_invalid(self):

        self.fakesbc.check_and_set_time_points_input.side_effect = \
            ValueError
        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_measurement_time_points_input, \
            self.ty_ref)


class CheckAndSetControlsData(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.nsteps = 20
        # self.fakesbc.ncontrols = self.fakesbc.nsteps + 1 for NonDyn,
        # but makes no difference for testing the function itself
        self.fakesbc.ncontrols = self.fakesbc.nsteps

    def test_input_rows(self):

        self.fakesbc.nu = 3
        udata_ref = np.random.rand(self.fakesbc.nu, self.fakesbc.nsteps)

        self.fakesbc.check_and_set_controls_data(udata_ref)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_columns(self):

        self.fakesbc.nu = 3
        udata_ref = np.random.rand(self.fakesbc.nu, self.fakesbc.nsteps)

        self.fakesbc.check_and_set_controls_data(udata_ref.T)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_none(self):

        self.fakesbc.nu = 3

        udata_ref = np.zeros((self.fakesbc.nu, self.fakesbc.nsteps))

        self.fakesbc.check_and_set_controls_data(None)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_zero_controls(self):

        self.fakesbc.nu = 0

        udata_ref = ca.DMatrix(0, self.fakesbc.nsteps)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        self.fakesbc.check_and_set_controls_data(None)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_invalid(self):

        self.fakesbc.nu = 3
        udata_ref = np.random.rand(self.fakesbc.nu + 1, self.fakesbc.nsteps)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_controls_data, udata_ref)


class CheckAndSetParameterData(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()
        self.fakesbc.np = 5


    def test_input_rows(self):

        pdata_ref = np.linspace(0, self.fakesbc.np - 1, self.fakesbc.np)

        self.fakesbc.check_and_set_parameter_data(pdata_ref)
        assert_array_equal(self.fakesbc.pdata, pdata_ref)


    def test_input_columns(self):

        pdata_ref = np.linspace(0, self.fakesbc.np - 1, self.fakesbc.np)

        self.fakesbc.check_and_set_parameter_data(pdata_ref)
        assert_array_equal(self.fakesbc.pdata, pdata_ref)


    def test_input_none(self):

        pdata_ref = np.zeros(self.fakesbc.np)

        self.fakesbc.check_and_set_parameter_data(None)
        assert_array_equal(self.fakesbc.pdata, pdata_ref)


    def test_input_invalid_onedim(self):

        pdata_ref = np.linspace(0, self.fakesbc.np - 2, self.fakesbc.np - 1)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_parameter_data, pdata_ref)


    def test_input_invalid_twodim(self):

        pdata_ref = np.random.rand(self.fakesbc.np, 2)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_parameter_data, pdata_ref)   


class CheckAndSetStatesData(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.nsteps = 20


    def test_input_rows(self):

        self.fakesbc.nx = 4
        xdata_ref = np.random.rand(self.fakesbc.nx, self.fakesbc.nsteps + 1)

        self.fakesbc.check_and_set_states_data(xdata_ref)
        assert_array_equal(self.fakesbc.xdata, xdata_ref)


    def test_input_columns(self):

        self.fakesbc.nx = 4
        xdata_ref = np.random.rand(self.fakesbc.nx, self.fakesbc.nsteps + 1)

        self.fakesbc.check_and_set_states_data(xdata_ref.T)
        assert_array_equal(self.fakesbc.xdata, xdata_ref)


    def test_input_none(self):

        self.fakesbc.nx = 4

        xdata_ref = np.zeros((self.fakesbc.nx, self.fakesbc.nsteps + 1))

        self.fakesbc.check_and_set_states_data(None)
        assert_array_equal(self.fakesbc.xdata, xdata_ref)


    def test_zero_states(self):

        self.fakesbc.nx = 0

        xdata_ref = ca.DMatrix(0, 0)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        self.fakesbc.check_and_set_states_data(None)
        assert_array_equal(self.fakesbc.xdata, xdata_ref)


    def test_input_invalid(self):

        self.fakesbc.nx = 4
        xdata_ref = np.random.rand(self.fakesbc.nx, self.fakesbc.nsteps)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_states_data, xdata_ref)


class CheckAndSetMeasurementData(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.ty = np.linspace(0, 19, 20)
        self.fakesbc.nphi = 2


    def test_input_rows(self):

        ydata_ref = np.random.rand(self.fakesbc.nphi, self.fakesbc.ty.size)

        self.fakesbc.check_and_set_measurement_data(ydata_ref)
        assert_array_equal(self.fakesbc.ydata, ydata_ref)


    def test_input_columns(self):

        ydata_ref = np.random.rand(self.fakesbc.nphi, self.fakesbc.ty.size)

        self.fakesbc.check_and_set_measurement_data(ydata_ref.T)
        assert_array_equal(self.fakesbc.ydata, ydata_ref)


    def test_input_none(self):

        ydata_ref = np.zeros((self.fakesbc.nphi, self.fakesbc.ty.size))

        self.fakesbc.check_and_set_measurement_data(None)
        assert_array_equal(self.fakesbc.ydata, ydata_ref)


    def test_input_invalid(self):

        ydata_ref = np.random.rand(self.fakesbc.nphi + 1, self.fakesbc.ty.size)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_measurement_data, ydata_ref)


class CheckAndSetMeasurementWeightings(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.ydata = np.random.rand(4, 3)


    def test_input_rows(self):

        wv_ref = np.random.rand(*self.fakesbc.ydata.shape)

        self.fakesbc.check_and_set_measurement_weightings(wv_ref)
        assert_array_equal(self.fakesbc.wv, wv_ref)


    def test_input_columns(self):

        wv_ref = np.random.rand(*self.fakesbc.ydata.shape)

        self.fakesbc.check_and_set_measurement_weightings(wv_ref.T)
        assert_array_equal(self.fakesbc.wv, wv_ref)


    def test_input_none(self):

        wv_ref = np.ones((self.fakesbc.ydata.shape))

        self.fakesbc.check_and_set_measurement_weightings(None)
        assert_array_equal(self.fakesbc.wv, wv_ref)


    def test_input_invalid(self):

        wv_ref = np.random.rand(self.fakesbc.ydata.shape[0] + 1, \
            self.fakesbc.ydata.shape[1])

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_measurement_weightings, wv_ref)


class CheckAndSetEquationErrorWeightings(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.neps_e = 3


    def test_input_rows(self):

        weps_e_ref = \
            np.linspace(0, self.fakesbc.neps_e - 1, self.fakesbc.neps_e)

        self.fakesbc.check_and_set_equation_error_weightings(weps_e_ref)
        assert_array_equal(self.fakesbc.weps_e, weps_e_ref)


    def test_input_columns(self):

        weps_e_ref = \
            np.linspace(0, self.fakesbc.neps_e - 1, self.fakesbc.neps_e)

        self.fakesbc.check_and_set_equation_error_weightings(weps_e_ref.T)
        assert_array_equal(self.fakesbc.weps_e, weps_e_ref)

    def test_input_none(self):

        weps_e_ref = np.ones(self.fakesbc.neps_e)

        self.fakesbc.check_and_set_equation_error_weightings(None)
        assert_array_equal(self.fakesbc.weps_e, weps_e_ref)


    def test_zero_equation_errors(self):

        self.fakesbc.neps_e = 0

        weps_e_ref = ca.DMatrix(0, 0)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        self.fakesbc.check_and_set_equation_error_weightings(None)
        assert_array_equal(self.fakesbc.weps_e, weps_e_ref)


    def test_input_invalid_onedim(self):

        weps_e_ref = \
            np.linspace(0, self.fakesbc.neps_e, self.fakesbc.neps_e + 1)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_equation_error_weightings, weps_e_ref)


    def test_input_invalid_twodim(self):

        weps_e_ref = np.random.rand(self.fakesbc.neps_e, 2)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_equation_error_weightings, weps_e_ref)


class CheckAndSetInputErrorWeightings(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.neps_u = 3


    def test_input_rows(self):

        weps_u_ref = \
            np.linspace(0, self.fakesbc.neps_u - 1, self.fakesbc.neps_u)

        self.fakesbc.check_and_set_input_error_weightings(weps_u_ref)
        assert_array_equal(self.fakesbc.weps_u, weps_u_ref)


    def test_input_columns(self):

        weps_u_ref = \
            np.linspace(0, self.fakesbc.neps_u - 1, self.fakesbc.neps_u)

        self.fakesbc.check_and_set_input_error_weightings(weps_u_ref)
        assert_array_equal(self.fakesbc.weps_u, weps_u_ref)

    def test_input_none(self):

        weps_u_ref = np.ones(self.fakesbc.neps_u)

        self.fakesbc.check_and_set_input_error_weightings(None)
        assert_array_equal(self.fakesbc.weps_u, weps_u_ref)


    def test_zero_input_errors(self):

        self.fakesbc.neps_u = 0

        weps_u_ref = ca.DMatrix(0, 0)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        self.fakesbc.check_and_set_input_error_weightings(None)
        assert_array_equal(self.fakesbc.weps_u, weps_u_ref)


    def test_input_invalid_onedim(self):

        weps_u_ref = \
            np.linspace(0, self.fakesbc.neps_u, self.fakesbc.neps_u + 1)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_input_error_weightings, weps_u_ref)


    def test_input_invalid_twodim(self):

        weps_u_ref = np.random.rand(self.fakesbc.neps_u, 2)

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_input_error_weightings, weps_u_ref)


class CheckSetProblemDimensionsFromSystemInformation(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()

        self.fakesbc.system = mock.MagicMock()

        self.fakesbc.system.u.shape = (2, 1)
        self.fakesbc.system.p.shape = (3, 1)
        self.fakesbc.system.phi.shape = (4, 1)

        self.fakesbc.system.x.shape = (5, 1)
        self.fakesbc.system.eps_e.shape = (6, 1)
        self.fakesbc.system.eps_u.shape = (7, 1)


    def test_set_control_dimension(self):

        self.fakesbc.set_problem_dimensions_from_system_information()
        self.assertEqual(self.fakesbc.nu, self.fakesbc.system.u.shape[0])


    def test_set_parameter_dimension(self):

        self.fakesbc.set_problem_dimensions_from_system_information()
        self.assertEqual(self.fakesbc.np, self.fakesbc.system.p.shape[0])


    def test_set_measurement_function_dimension(self):

        self.fakesbc.set_problem_dimensions_from_system_information()
        self.assertEqual(self.fakesbc.nphi, self.fakesbc.system.phi.shape[0])


    def test_set_states_dimension(self):

        self.fakesbc.set_problem_dimensions_from_system_information()
        self.assertEqual(self.fakesbc.nx, self.fakesbc.system.x.shape[0])


    def test_set_equation_error_dimension(self):

        self.fakesbc.set_problem_dimensions_from_system_information()
        self.assertEqual(self.fakesbc.neps_e, \
            self.fakesbc.system.eps_e.shape[0])


    def test_set_input_error_dimension(self):

        self.fakesbc.set_problem_dimensions_from_system_information()
        self.assertEqual(self.fakesbc.neps_u, \
            self.fakesbc.system.eps_u.shape[0])


    def test_no_states(self):

        self.fakesbc.system.x = None
        self.fakesbc.set_problem_dimensions_from_system_information()

        self.assertEqual(self.fakesbc.nx, 0)
        

    def test_no_equaiton_errors(self):

        self.fakesbc.system.eps_e = None
        self.fakesbc.set_problem_dimensions_from_system_information()

        self.assertEqual(self.fakesbc.neps_e, 0)
        

    def test_no_input_errors(self):

        self.fakesbc.system.eps_u = None
        self.fakesbc.set_problem_dimensions_from_system_information()

        self.assertEqual(self.fakesbc.neps_u, 0)
        