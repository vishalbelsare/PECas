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

import casadi as ca
import pecas

import unittest

class BSSetInitialsTest(object):

    def test_valid_timegrid_inputs(self):

        # Test valid input dimensions for tu

        pecas.setups.BSsetup(system = self.bsys, \
            tu = self.tu)

        pecas.setups.BSsetup(system = self.bsys, tu = \
            self.tu.T)


    def test_invalid_systems_input(self):

        # Support an invalid systems-type

        odesys = pecas.systems.ExplODE(p = self.p, phi = self.p, x = self.p, \
            eps_e = self.p, eps_u = self.p, f = self.p)

        self.assertRaises(TypeError, pecas.setups.BSsetup, \
            system = odesys, tu = self.tu)


    def test_invalid_parameter_initials(self):

        # Test some invalid values for p-arguments

        for parg in self.invalidpargs:

            self.assertRaises(ValueError, pecas.setups.BSsetup, \
                system = self.bsys, tu = self.tu, pinit = parg)


    def test_valid_parameter_initials(self):

        # Test some valid values for p-arguments

        for parg in self.validpargs:

            pecas.setups.BSsetup( \
                system = self.bsys, tu = self.tu, pinit = parg)


    def test_invalid_control_inputs(self):

        # Test some invalid values for uN-arguments       

        for uarg in self.invaliduargs:

            self.assertRaises(ValueError, pecas.setups.BSsetup, \
                system = self.bsys, tu = self.tu, uN = uarg)
    

    def test_valid_control_inputs(self):

        # Test some valid values for uN-arguments

        for uarg in self.validuargs:

            pecas.setups.BSsetup( \
                system = self.bsys, tu = self.tu, uN = uarg)


class ODESetInitialsTest(object):

    def test_valid_timegrid_inputs(self):

        # Test valid input dimensions for tu

        pecas.setups.ODEsetup(system = self.odesys, \
            tu = self.tu)

        pecas.setups.ODEsetup(system = self.odesys, tu = \
            self.tu.T)


    def test_invalid_systems_input(self):

        # Support an invalid systems-type

        bssys = pecas.systems.BasicSystem(p = self.p, phi = self.p)
        
        self.assertRaises(TypeError, pecas.setups.ODEsetup, \
            system = bssys, tu = self.tu)


    def test_invalid_parameter_initials(self):

        # Test some invalid values for p-arguments

        for parg in self.invalidpargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, tu = self.tu, pinit = parg)


    def test_valid_parameter_initials(self):

        # Test some valid values for p-arguments

        for parg in self.validpargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, tu = self.tu, pinit = parg)


    def test_invalid_state_initials(self):

        # Test some invalid values for x-arguments

        for xarg in self.invalidxargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, tu = self.tu, xinit = xarg)


    def test_valid_state_initials(self):

        # Test some valid values for x-arguments

        for xarg in self.validxargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, tu = self.tu, xinit = xarg)


    def test_invalid_control_inputs(self):

        # Test some invalid values for uN-arguments       

        for uarg in self.invaliduargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, tu = self.tu, uN = uarg)
    

    def test_valid_control_inputs(self):

        # Test some valid values for uN-arguments

        for uarg in self.validuargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, tu = self.tu, uN = uarg)
