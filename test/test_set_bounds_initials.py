#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the function for setting bounds and initials

import casadi as ca
import pylab as pl
import pecas

import unittest

class BSSetBoundsInitialsTest(object):

    def test_valid_timegrid_inputs(self):

        # Test valid input dimensions for timegrid

        pecas.setups.BSsetup(system = self.bsys, \
            timegrid = self.timegrid)
        pecas.setups.BSsetup(system = self.bsys, timegrid = \
            self.timegrid.T)


    def test_invalid_systems_input(self):

        # Support an invalid systems-type

        odesys = pecas.systems.ExplODE(p = self.p, y = self.p, x = self.p, \
            w = self.p, f = self.p)
        self.assertRaises(TypeError, pecas.setups.BSsetup, \
            system = odesys, timegrid = self.timegrid)


    def test_invalid_parameter_bounds_and_initials(self):

        # Test some invalid values for p-arguments

        for parg in self.invalidpargs:

            self.assertRaises(ValueError, pecas.setups.BSsetup, \
                system = self.bsys, timegrid = self.timegrid, pinit = parg)
            self.assertRaises(ValueError, pecas.setups.BSsetup, \
                system = self.bsys, timegrid = self.timegrid, pmin = parg)
            self.assertRaises(ValueError, pecas.setups.BSsetup, \
                system = self.bsys, timegrid = self.timegrid, pmax = parg)


    def test_valid_parameter_bounds_and_initials(self):

        # Test some valid values for p-arguments

        for parg in self.validpargs:

            pecas.setups.BSsetup( \
                system = self.bsys, timegrid = self.timegrid, pinit = parg)
            pecas.setups.BSsetup( \
                system = self.bsys, timegrid = self.timegrid, pmin = parg)
            pecas.setups.BSsetup( \
                system = self.bsys, timegrid = self.timegrid, pmax = parg)


    def test_invalid_control_bounds_and_initials_inputs(self):

        # Test some invalid values for u-arguments       

        for uarg in self.invaliduargs:

            self.assertRaises(ValueError, pecas.setups.BSsetup, \
                system = self.bsys, timegrid = self.timegrid, u = uarg)
    

    def test_valid_control_bounds_and_initials_inputs(self):

        # Test some valid values for u-arguments

        for uarg in self.validuargs:

            pecas.setups.BSsetup( \
                system = self.bsys, timegrid = self.timegrid, u = uarg)


class ODESetBoundsInitialsTest(object):

    def test_valid_timegrid_inputs(self):

        # Test valid input dimensions for timegrid

        pecas.setups.ODEsetup(system = self.odesys, \
            timegrid = self.timegrid)
        pecas.setups.ODEsetup(system = self.odesys, timegrid = \
            self.timegrid.T)


    def test_invalid_systems_input(self):

        # Support an invalid systems-type

        bssys = pecas.systems.BasicSystem(p = self.p, y = self.p)
        self.assertRaises(TypeError, pecas.setups.ODEsetup, \
            system = bssys, timegrid = self.timegrid)


    def test_invalid_parameter_bounds_and_initials(self):

        # Test some invalid values for p-arguments

        for parg in self.invalidpargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, pinit = parg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, pmin = parg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, pmax = parg)


    def test_valid_parameter_bounds_and_initials(self):

        # Test some valid values for p-arguments

        for parg in self.validpargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, pinit = parg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, pmin = parg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, pmax = parg)


    def test_invalid_state_bounds_and_initials(self):

        # Test some invalid values for x-arguments

        for xarg in self.invalidxargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xinit = xarg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xmin = xarg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xmax = xarg)


    def test_valid_state_bounds_and_initials(self):

        # Test some valid values for x-arguments

        for xarg in self.validxargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xinit = xarg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xmin = xarg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xmax = xarg)


    def test_invalid_state_bvp_inputs(self):

        # Test some invalid values for xbvp-arguments

        for xbvparg in self.invalidxbvpargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, x0min = xbvparg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, x0max = xbvparg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xNmin = xbvparg)
            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xNmax = xbvparg)
    

    def test_valid_state_bvp_inputs(self):

        # Test some valid values for xbvp-arguments

        for xbvparg in self.validxbvpargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, x0min = xbvparg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, x0max = xbvparg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xNmin = xbvparg)
            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xNmax = xbvparg)


    def test_invalid_control_bounds_and_initials_inputs(self):

        # Test some invalid values for u-arguments       

        for uarg in self.invaliduargs:

            self.assertRaises(ValueError, pecas.setups.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, u = uarg)
    

    def test_valid_control_bounds_and_initials_inputs(self):

        # Test some valid values for u-arguments

        for uarg in self.validuargs:

            pecas.setups.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, u = uarg)
