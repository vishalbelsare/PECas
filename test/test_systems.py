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

# Test the classes for system definitions

import casadi as ca
import pecas

import unittest

class TestNonDyn(unittest.TestCase):

    def setUp(self):

        self.t = ca.MX.sym("t", 1)
        self.u = ca.MX.sym("u", 1)
        self.p = ca.MX.sym("p", 1)
        self.phi = ca.MX.sym("phi", 1)
        self.g = ca.MX.sym("g", 1)


    def test_nondyn_init_p_phi(self):

        sys = pecas.systems.NonDyn(p = self.p, phi = self.phi)
        sys.show_system_information(showEquations = True)


    def test_nondyn_init_t_p_phi(self):
        
        sys = pecas.systems.NonDyn(t = self.t, p = self.p, phi = self.phi)
        sys.show_system_information(showEquations = True)
        

    def test_nondyn_init_t_u_p_phi(self):

        sys = pecas.systems.NonDyn(t = self.t, u = self.u, p = self.p, \
            phi = self.phi)
        sys.show_system_information(showEquations = True)

    def test_nondyn_init_t_u_p_phi_g(self):

        sys = pecas.systems.NonDyn(t = self.t, u = self.u, p = self.p, \
            phi = self.phi, g = self.g)
        sys.show_system_information(showEquations = True)


    def test_nondyn_init_no_args(self):

        self.assertRaises(TypeError, pecas.systems.NonDyn)


    def test_nondyn_init_no_phi(self):

        self.assertRaises(TypeError, pecas.systems.NonDyn, p = None, \
            phi = self.phi)


    def test_nondyn_init_no_p(self):

        self.assertRaises(TypeError, pecas.systems.NonDyn, p = self.p, \
            phi = None)


class TestExplODE(unittest.TestCase):

    def setUp(self):

        self.t = ca.MX.sym("t", 1)
        self.u = ca.MX.sym("u", 1)
        self.x = ca.MX.sym("x", 1)
        self.p = ca.MX.sym("p", 1)
        self.eps_e = ca.MX.sym("eps_e", 1)
        self.eps_u = ca.MX.sym("eps_u", 1)
        self.phi = ca.MX.sym("phi", 1)
        self.f = ca.MX.sym("f", 1)


    def test_explode_init_x_p_epse_phi_f(self):

        sys = pecas.systems.ExplODE(x = self.x, p = self.p, \
            eps_e = self.eps_e, phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)


    def test_explode_init_t_x_p_epse_phi_f(self):

        sys = pecas.systems.ExplODE(t = self.t, x = self.x, p = self.p, \
            eps_e = self.eps_e, phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)


    def test_explode_init_t_u_x_p_epse_phi_f(self):

        sys = pecas.systems.ExplODE(t = self.t, u = self.u, x = self.x, \
            p = self.p, eps_e = self.eps_e, phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)


    def test_explode_init_t_u_x_p_epse_epsu_phi_f(self):

        sys = pecas.systems.ExplODE(t = self.t, u = self.u, x = self.x,\
            p = self.p, eps_e = self.eps_e, eps_u = self.eps_u, \
            phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)


    def test_explode_init_no_args(self):

        self.assertRaises(TypeError, pecas.systems.ExplODE)


    def test_explode_init_no_x(self):

        self.assertRaises(TypeError, pecas.systems.ExplODE, x = None, \
            p = self.p, phi = self.phi, f = self.f)


    def test_explode_init_no_p(self):

        self.assertRaises(TypeError, pecas.systems.ExplODE, x = self.x, \
            p = None, phi = self.phi, f = self.f)


    def test_explode_init_no_phi(self):

        self.assertRaises(TypeError, pecas.systems.ExplODE, x = self.x, \
            p = self.p, phi = None, f = self.f)


    def test_explode_init_no_f(self):

        self.assertRaises(TypeError, pecas.systems.ExplODE, x = self.x, \
            p = self.p, phi = self.phi, f = None)


    def test_assure_no_explicit_time_dependecy(self):

        # Assure as long as explicit time dependecy is not allowed

        self.assertRaises(NotImplementedError, pecas.systems.ExplODE, \
            t = self.t, u = self.u, x = self.x, \
            p = self.p, eps_e = self.eps_e, phi = self.phi, f = self.t)


class TestImplDAE(unittest.TestCase):


    def test_impldae_system_init(self):

        # Assure as long as not implemented

        self.assertRaises(NotImplementedError, pecas.systems.ImplDAE)
    