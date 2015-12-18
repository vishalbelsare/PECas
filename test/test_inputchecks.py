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

from pecas.interfaces import casadi_interface as ci
from pecas import inputchecks

import unittest
import mock

# class FakeSetupsBaseClass(SetupsBaseClass):

#     # Cf.: http://stackoverflow.com/questions/27105491/
#     # how-can-i-unit-test-a-method-without-instantiating-the-class 

#     def __init__(self, *args, **kwargs):

#         pass


#     def define_number_of_control_intervals(self):

#         pass


#     def discretize_problem(self):

#         pass


#     def set_initials(self):

#         pass


# class Discretization(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.discretization_method = "collocation"
#         self.number_of_collocation_points = 4
#         self.collocation_scheme = "legendre"

#     def test_set_default_method(self):

#         self.ds = self.fakesbc.Discretization()
#         self.assertEqual(self.ds.discretization_method, None)

    
#     def test_set_default_number_of_collocation_points(self):

#         self.ds = self.fakesbc.Discretization()
#         self.assertEqual(self.ds.number_of_collocation_points, 3)


#     def test_set_default_collocation_scheme(self):

#         self.ds = self.fakesbc.Discretization()
#         self.assertEqual(self.ds.collocation_scheme, "radau")


#     def test_set_custom_method(self):

#         self.ds = self.fakesbc.Discretization( \
#             discretization_method = self.discretization_method)
#         self.assertEqual(self.ds.discretization_method, \
#             self.discretization_method)

    
#     def test_set_custom_number_of_collocation_points(self):

#         self.ds = self.fakesbc.Discretization( \
#             number_of_collocation_points= self.number_of_collocation_points)
#         self.assertEqual(self.ds.number_of_collocation_points, \
#             self.number_of_collocation_points)


#     def test_set_custom_collocation_scheme(self):

#         self.ds = self.fakesbc.Discretization( \
#             collocation_scheme= self.collocation_scheme)
#         self.assertEqual(self.ds.collocation_scheme, \
#             self.collocation_scheme)

#     def test_return_collocation_points_for_collocation(self):

#         # collocation_points = casadi.collocationPoints(3, "radau")
#         collocation_points = \
#             [0.0, 0.15505102572168222, 0.6449489742783179, 1.0]

#         self.ds = self.fakesbc.Discretization( \
#             discretization_method = self.discretization_method)
#         self.assertEqual(self.ds.get_collocation_points(), collocation_points)


#     def test_return_collocation_polynomial_degree_for_collocation(self):

#         # collocation_points = casadi.collocationPoints(3, "radau")
#         collocation_polynomial_degree = 3

#         self.ds = self.fakesbc.Discretization( \
#             discretization_method = self.discretization_method)
#         self.assertEqual(self.ds.get_collocation_polynomial_degree(), \
#             collocation_polynomial_degree)


#     def test_return_no_collocation_points_for_other_methods(self):

#         collocation_points = []

#         self.ds = self.fakesbc.Discretization( \
#             discretization_method = None)
#         self.assertEqual(self.ds.get_collocation_points(), collocation_points)


#     def test_return_collocation_polynomial_degree_for_other_methods(self):

#         collocation_polynomial_degree = 0

#         self.ds = self.fakesbc.Discretization( \
#             discretization_method = None)
#         self.assertEqual(self.ds.get_collocation_polynomial_degree(), \
#             collocation_polynomial_degree)


class SetSystem(unittest.TestCase):

    def setUp(self):

        self.system = "system"


    def test_set_system(self):

        system = "system"

        # A better test would be needed here also to make sure that system is
        # subclass of systems.SystemBaseClass, but this is yet problematic

        system = inputchecks.set_system(system)
        self.assertEqual(self.system, system)


class CheckTimepointsInput(unittest.TestCase):

    def setUp(self):

        self.tp_ref = np.linspace(0, 49, 50)


    def test_input_list(self):

        tp = [k for k in range(50)]

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_onedim_time_vector(self):

        tp = np.linspace(0, 49, 50)

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_row_time_vector(self):

        tp = np.atleast_2d(np.linspace(0, 49, 50))

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_column_time_vector(self):

        tp = np.atleast_2d(np.linspace(0, 49, 50)).T

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_invalid_time_vector(self):

        tp = np.random.randn(2,2)

        self.assertRaises(ValueError, \
            inputchecks.check_time_points_input, tp)


class CheckControlsData(unittest.TestCase):

    def setUp(self):

        self.number_of_controls = 20

    def test_input_rows(self):

        nu = 3
        udata_ref = np.random.rand(nu, self.number_of_controls)

        udata = inputchecks.check_controls_data(udata_ref, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)


    def test_input_columns(self):

        nu = 3
        udata_ref = np.random.rand(nu, self.number_of_controls)

        udata = inputchecks.check_controls_data(udata_ref.T, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)


    def test_input_none(self):

        nu = 3
        udata_ref = np.zeros((nu, self.number_of_controls))

        udata = inputchecks.check_controls_data(None, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)

    def test_zero_controls(self):

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        nu = 0
        udata_ref = ci.dmatrix(0, self.number_of_controls)

        udata = inputchecks.check_controls_data(None, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)

    def test_input_invalid(self):

        nu = 3
        udata_ref = np.zeros((nu + 1, self.number_of_controls))

        self.assertRaises(ValueError, \
            inputchecks.check_controls_data, udata_ref, \
            nu, self.number_of_controls)


class CheckAndSetStatesData(unittest.TestCase):

    def setUp(self):

        self.number_of_intervals = 20


    def test_input_rows(self):

        nx = 4
        xdata_ref = np.random.rand(nx, self.number_of_intervals + 1)

        xdata = inputchecks.check_states_data(xdata_ref, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_input_columns(self):

        nx = 4
        xdata_ref = np.random.rand(nx, self.number_of_intervals + 1)

        xdata = inputchecks.check_states_data(xdata_ref.T, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_input_none(self):

        nx = 4
        xdata_ref = np.zeros((nx, self.number_of_intervals + 1))

        xdata = inputchecks.check_states_data(None, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_zero_states(self):

        nx = 0
        xdata_ref = ci.dmatrix(0, 0)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        xdata = inputchecks.check_states_data(None, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_input_invalid(self):

        nx = 4
        xdata_ref = np.random.rand(nx, self.number_of_intervals)

        self.assertRaises(ValueError, \
            inputchecks.check_states_data, xdata_ref, nx, \
            self.number_of_intervals + 1)


# class CheckAndSetParameterData(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()
#         self.fakesbc.np = 5


#     def test_input_rows(self):

#         pdata_ref = np.linspace(0, self.fakesbc.np - 1, self.fakesbc.np)

#         self.fakesbc.check_and_set_parameter_data(pdata_ref)
#         assert_array_equal(self.fakesbc.pdata, pdata_ref)


#     def test_input_columns(self):

#         pdata_ref = np.linspace(0, self.fakesbc.np - 1, self.fakesbc.np)

#         self.fakesbc.check_and_set_parameter_data(pdata_ref)
#         assert_array_equal(self.fakesbc.pdata, pdata_ref)


#     def test_input_none(self):

#         pdata_ref = np.zeros(self.fakesbc.np)

#         self.fakesbc.check_and_set_parameter_data(None)
#         assert_array_equal(self.fakesbc.pdata, pdata_ref)


#     def test_input_invalid_onedim(self):

#         pdata_ref = np.linspace(0, self.fakesbc.np - 2, self.fakesbc.np - 1)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_parameter_data, pdata_ref)


#     def test_input_invalid_twodim(self):

#         pdata_ref = np.random.rand(self.fakesbc.np, 2)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_parameter_data, pdata_ref)   




# class CheckAndSetMeasurementData(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.fakesbc.ty = np.linspace(0, 19, 20)
#         self.fakesbc.nphi = 2


#     def test_input_rows(self):

#         ydata_ref = np.random.rand(self.fakesbc.nphi, self.fakesbc.ty.size)

#         self.fakesbc.check_and_set_measurement_data(ydata_ref)
#         assert_array_equal(self.fakesbc.ydata, ydata_ref)


#     def test_input_columns(self):

#         ydata_ref = np.random.rand(self.fakesbc.nphi, self.fakesbc.ty.size)

#         self.fakesbc.check_and_set_measurement_data(ydata_ref.T)
#         assert_array_equal(self.fakesbc.ydata, ydata_ref)


#     def test_input_none(self):

#         ydata_ref = np.zeros((self.fakesbc.nphi, self.fakesbc.ty.size))

#         self.fakesbc.check_and_set_measurement_data(None)
#         assert_array_equal(self.fakesbc.ydata, ydata_ref)


#     def test_input_invalid(self):

#         ydata_ref = np.random.rand(self.fakesbc.nphi + 1, self.fakesbc.ty.size)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_measurement_data, ydata_ref)


# class CheckAndSetMeasurementWeightings(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.fakesbc.ydata = np.random.rand(4, 3)


#     def test_input_rows(self):

#         wv_ref = np.random.rand(*self.fakesbc.ydata.shape)

#         self.fakesbc.check_and_set_measurement_weightings(wv_ref)
#         assert_array_equal(self.fakesbc.wv, wv_ref)


#     def test_input_columns(self):

#         wv_ref = np.random.rand(*self.fakesbc.ydata.shape)

#         self.fakesbc.check_and_set_measurement_weightings(wv_ref.T)
#         assert_array_equal(self.fakesbc.wv, wv_ref)


#     def test_input_none(self):

#         wv_ref = np.ones((self.fakesbc.ydata.shape))

#         self.fakesbc.check_and_set_measurement_weightings(None)
#         assert_array_equal(self.fakesbc.wv, wv_ref)


#     def test_input_invalid(self):

#         wv_ref = np.random.rand(self.fakesbc.ydata.shape[0] + 1, \
#             self.fakesbc.ydata.shape[1])

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_measurement_weightings, wv_ref)


# class CheckAndSetEquationErrorWeightings(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.fakesbc.neps_e = 3


#     def test_input_rows(self):

#         weps_e_ref = \
#             np.linspace(0, self.fakesbc.neps_e - 1, self.fakesbc.neps_e)

#         self.fakesbc.check_and_set_equation_error_weightings(weps_e_ref)
#         assert_array_equal(self.fakesbc.weps_e, weps_e_ref)


#     def test_input_columns(self):

#         weps_e_ref = \
#             np.linspace(0, self.fakesbc.neps_e - 1, self.fakesbc.neps_e)

#         self.fakesbc.check_and_set_equation_error_weightings(weps_e_ref.T)
#         assert_array_equal(self.fakesbc.weps_e, weps_e_ref)

#     def test_input_none(self):

#         weps_e_ref = np.ones(self.fakesbc.neps_e)

#         self.fakesbc.check_and_set_equation_error_weightings(None)
#         assert_array_equal(self.fakesbc.weps_e, weps_e_ref)


#     def test_zero_equation_errors(self):

#         self.fakesbc.neps_e = 0

#         weps_e_ref = ci.dmatrix(0, 0)

#         # In this case, the input value is not used by the function, and
#         # therefor irrelevant at this point

#         self.fakesbc.check_and_set_equation_error_weightings(None)
#         assert_array_equal(self.fakesbc.weps_e, weps_e_ref)


#     def test_input_invalid_onedim(self):

#         weps_e_ref = \
#             np.linspace(0, self.fakesbc.neps_e, self.fakesbc.neps_e + 1)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_equation_error_weightings, weps_e_ref)


#     def test_input_invalid_twodim(self):

#         weps_e_ref = np.random.rand(self.fakesbc.neps_e, 2)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_equation_error_weightings, weps_e_ref)


# class CheckAndSetInputErrorWeightings(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.fakesbc.neps_u = 3


#     def test_input_rows(self):

#         weps_u_ref = \
#             np.linspace(0, self.fakesbc.neps_u - 1, self.fakesbc.neps_u)

#         self.fakesbc.check_and_set_input_error_weightings(weps_u_ref)
#         assert_array_equal(self.fakesbc.weps_u, weps_u_ref)


#     def test_input_columns(self):

#         weps_u_ref = \
#             np.linspace(0, self.fakesbc.neps_u - 1, self.fakesbc.neps_u)

#         self.fakesbc.check_and_set_input_error_weightings(weps_u_ref)
#         assert_array_equal(self.fakesbc.weps_u, weps_u_ref)

#     def test_input_none(self):

#         weps_u_ref = np.ones(self.fakesbc.neps_u)

#         self.fakesbc.check_and_set_input_error_weightings(None)
#         assert_array_equal(self.fakesbc.weps_u, weps_u_ref)


#     def test_zero_input_errors(self):

#         self.fakesbc.neps_u = 0

#         weps_u_ref = ci.dmatrix(0, 0)

#         # In this case, the input value is not used by the function, and
#         # therefor irrelevant at this point

#         self.fakesbc.check_and_set_input_error_weightings(None)
#         assert_array_equal(self.fakesbc.weps_u, weps_u_ref)


#     def test_input_invalid_onedim(self):

#         weps_u_ref = \
#             np.linspace(0, self.fakesbc.neps_u, self.fakesbc.neps_u + 1)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_input_error_weightings, weps_u_ref)


#     def test_input_invalid_twodim(self):

#         weps_u_ref = np.random.rand(self.fakesbc.neps_u, 2)

#         self.assertRaises(ValueError, \
#             self.fakesbc.check_and_set_input_error_weightings, weps_u_ref)


# class CheckSetProblemDimensionsFromSystemInformation(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.fakesbc.system = mock.MagicMock()

#         self.fakesbc.system.u.shape = (2, 1)
#         self.fakesbc.system.p.shape = (3, 1)
#         self.fakesbc.system.phi.shape = (4, 1)

#         self.fakesbc.system.x.shape = (5, 1)
#         self.fakesbc.system.eps_e.shape = (6, 1)
#         self.fakesbc.system.eps_u.shape = (7, 1)


#     def test_set_control_dimension(self):

#         self.fakesbc.set_problem_dimensions_from_system_information()
#         self.assertEqual(self.fakesbc.nu, self.fakesbc.system.u.shape[0])


#     def test_set_parameter_dimension(self):

#         self.fakesbc.set_problem_dimensions_from_system_information()
#         self.assertEqual(self.fakesbc.np, self.fakesbc.system.p.shape[0])


#     def test_set_measurement_function_dimension(self):

#         self.fakesbc.set_problem_dimensions_from_system_information()
#         self.assertEqual(self.fakesbc.nphi, self.fakesbc.system.phi.shape[0])


#     def test_set_states_dimension(self):

#         self.fakesbc.set_problem_dimensions_from_system_information()
#         self.assertEqual(self.fakesbc.nx, self.fakesbc.system.x.shape[0])


#     def test_set_equation_error_dimension(self):

#         self.fakesbc.set_problem_dimensions_from_system_information()
#         self.assertEqual(self.fakesbc.neps_e, \
#             self.fakesbc.system.eps_e.shape[0])


#     def test_set_input_error_dimension(self):

#         self.fakesbc.set_problem_dimensions_from_system_information()
#         self.assertEqual(self.fakesbc.neps_u, \
#             self.fakesbc.system.eps_u.shape[0])


#     def test_no_states(self):

#         self.fakesbc.system.x = None
#         self.fakesbc.set_problem_dimensions_from_system_information()

#         self.assertEqual(self.fakesbc.nx, 0)
        

#     def test_no_equaiton_errors(self):

#         self.fakesbc.system.eps_e = None
#         self.fakesbc.set_problem_dimensions_from_system_information()

#         self.assertEqual(self.fakesbc.neps_e, 0)
        

#     def test_no_input_errors(self):

#         self.fakesbc.system.eps_u = None
#         self.fakesbc.set_problem_dimensions_from_system_information()

#         self.assertEqual(self.fakesbc.neps_u, 0)


# class CheckAndSetTimepoints(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()
#         self.fakesbc.check_and_set_control_time_points_input = \
#             mock.MagicMock()      
#         self.fakesbc.check_and_set_measurement_time_points_input = \
#             mock.MagicMock()
#         self.fakesbc.set_number_of_control_intervals = \
#             mock.MagicMock()

#     def test_functions_are_called_with_corresponding_arguments(self):

#         controls = {"tu": "controls"}
#         measurements = {"ty": "measurements"}

#         self.fakesbc.check_and_set_time_points(controls = controls, \
#             measurements = measurements)

#         self.fakesbc.check_and_set_control_time_points_input.\
#             assert_called_with("controls")
#         self.fakesbc.check_and_set_measurement_time_points_input.\
#             assert_called_with("measurements")
#         self.assertTrue(self.fakesbc.set_number_of_control_intervals.called)


# class SetOptimizationVariables(unittest.TestCase):

#     def setUp(self):

#         self.fakesbc = FakeSetupsBaseClass()

#         self.fakesbc.np = 1
#         self.fakesbc.nphi = 2
#         self.fakesbc.nx = 3

#         self.fakesbc.neps_e = 4
#         self.fakesbc.neps_u = 5
#         self.fakesbc.nu = 6

#         self.fakesbc.nintervals = 12
#         self.fakesbc.discretization = mock.MagicMock()
#         self.fakesbc.discretization.get_collocation_polynomial_degree = \
#             mock.MagicMock()
#         self.fakesbc.discretization.get_collocation_polynomial_degree.return_value = 3

#     def test_dimension_optimvar_p(self):

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["P"].shape, \
#             (self.fakesbc.np, 1))


#     def test_dimension_optimvar_v(self):

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["V"].shape, \
#             (self.fakesbc.nphi, self.fakesbc.nintervals + 1))


#     def test_dimension_optimvar_x_not_zero(self):

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["X"].shape, \
#             (self.fakesbc.nx, \
#             (self.fakesbc.discretization.get_collocation_polynomial_degree() + 1) * \
#                 self.fakesbc.nintervals))


#     def test_dimension_optimvar_x_zero(self):

#         self.fakesbc.nx = 0

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["X"].shape, \
#             (0, self.fakesbc.nintervals))


#     def test_dimension_optimvar_eps_e_not_zero(self):

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["EPS_E"].shape, \
#             (self.fakesbc.neps_e, \
#             self.fakesbc.discretization.get_collocation_polynomial_degree() * \
#                 self.fakesbc.nintervals))


#     def test_dimension_optimvar_eps_e_zero(self):

#         self.fakesbc.neps_e = 0

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["EPS_E"].shape, \
#             (0, self.fakesbc.nintervals))


#     def test_dimension_optimvar_eps_e_ntauroot_zero(self):

#         self.fakesbc.discretization.get_collocation_polynomial_degree.return_value = 0

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["EPS_E"].shape, \
#             (self.fakesbc.neps_e, self.fakesbc.nintervals))


#     def test_dimension_optimvar_eps_u_not_zero(self):

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["EPS_U"].shape, \
#             (self.fakesbc.neps_u, \
#             self.fakesbc.discretization.get_collocation_polynomial_degree() * \
#                 self.fakesbc.nintervals))


#     def test_dimension_optimvar_eps_u_zero(self):

#         self.fakesbc.neps_u = 0

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["EPS_U"].shape, \
#             (0, self.fakesbc.nintervals))


#     def test_dimension_optimvar_eps_u_ntauroot_zero(self):

#         self.fakesbc.discretization.get_collocation_polynomial_degree.return_value = 0

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["EPS_U"].shape, \
#             (self.fakesbc.neps_u, self.fakesbc.nintervals))


#     def test_dimension_optimvar_u_not_zero(self):

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["U"].shape, \
#             (self.fakesbc.nu, self.fakesbc.nintervals))


#     def test_dimension_optimvar_u_zero(self):

#         self.fakesbc.nu = 0

#         self.fakesbc.set_optimization_variables()
#         self.assertEqual(self.fakesbc.optimvars["U"].shape, \
#             (0, self.fakesbc.nintervals))