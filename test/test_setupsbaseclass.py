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
import pecas

import unittest
import mock

class FakeSetupsBaseClass(pecas.setups.SetupsBaseClass):

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


class CheckAndSetControls(unittest.TestCase):

    def setUp(self):

        self.fakesbc = FakeSetupsBaseClass()
        self.fakesbc.system = mock.MagicMock(spec = pecas.systems.ExplODE)

        self.fakesbc.nsteps = 20
        # self.fakesbc.ncontrols = self.fakesbc.nsteps + 1 for NonDyn,
        # but makes no difference for testing the function itself
        self.fakesbc.ncontrols = self.fakesbc.nsteps

    def test_input_rows(self):

        self.fakesbc.nu = 3

        udata_ref = \
            np.reshape(np.linspace(0, \
                (self.fakesbc.nu * self.fakesbc.nsteps) - 1, \
                (self.fakesbc.nu * self.fakesbc.nsteps)), \
            (self.fakesbc.nu, -1))

        self.fakesbc.check_and_set_controls_data(udata_ref)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_columns(self):

        self.fakesbc.nu = 3

        udata_ref = \
            np.reshape(np.linspace(0, \
                (self.fakesbc.nu * self.fakesbc.nsteps) - 1, \
                (self.fakesbc.nu * self.fakesbc.nsteps)), \
            (self.fakesbc.nu, -1))

        self.fakesbc.check_and_set_controls_data(udata_ref.T)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_none(self):

        self.fakesbc.nu = 3

        udata_ref = np.zeros((self.fakesbc.nu, self.fakesbc.nsteps))

        self.fakesbc.check_and_set_controls_data(None)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_zero(self):

        self.fakesbc.nu = 0

        udata_ref = ca.DMatrix(0, self.fakesbc.nsteps)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        self.fakesbc.check_and_set_controls_data(None)
        assert_array_equal(self.fakesbc.udata, udata_ref)


    def test_input_invalid(self):

        self.fakesbc.nu = 3

        udata_ref = \
            np.reshape(np.linspace(0, \
                (self.fakesbc.nu * self.fakesbc.nsteps) - 1, \
                (self.fakesbc.nu * self.fakesbc.nsteps)), \
            (self.fakesbc.nu + 1, -1))

        self.assertRaises(ValueError, \
            self.fakesbc.check_and_set_controls_data, udata_ref)

