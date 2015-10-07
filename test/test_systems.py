#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the classes fo system definitions

import pecas
import casadi as ca

import unittest

class TestSystemsInit(unittest.TestCase):

    def test_basic_system_init(self):

        self.t = ca.MX.sym("t", 1)
        self.u = ca.MX.sym("u", 1)
        self.p = ca.MX.sym("p", 1)
        self.phi = ca.MX.sym("phi", 1)
        self.g = ca.MX.sym("g", 1)

        sys = pecas.systems.BasicSystem(p = self.p, phi = self.phi)
        sys.show_system_information(showEquations = True)
        
        sys = pecas.systems.BasicSystem(t = self.t, p = self.p, phi = self.phi)
        sys.show_system_information(showEquations = True)
        
        sys = pecas.systems.BasicSystem(t = self.t, u = self.u, p = self.p, \
            phi = self.phi)
        sys.show_system_information(showEquations = True)
        
        sys = pecas.systems.BasicSystem(t = self.t, u = self.u, p = self.p, \
            phi = self.phi, g = self.g)
        sys.show_system_information(showEquations = True)

        self.assertRaises(TypeError, pecas.systems.BasicSystem)
        self.assertRaises(TypeError, pecas.systems.BasicSystem, p = None)
        self.assertRaises(TypeError, pecas.systems.BasicSystem, phi = None)


    def test_explode_system_init(self):

        self.t = ca.MX.sym("t", 1)
        self.u = ca.MX.sym("u", 1)
        self.x = ca.MX.sym("x", 1)
        self.p = ca.MX.sym("p", 1)
        self.eps_e = ca.MX.sym("eps_e", 1)
        self.eps_u = ca.MX.sym("eps_u", 1)
        self.phi = ca.MX.sym("phi", 1)
        self.f = ca.MX.sym("f", 1)

        sys = pecas.systems.ExplODE(x = self.x, p = self.p, \
            eps_e = self.eps_e, phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)

        sys = pecas.systems.ExplODE(t = self.t, x = self.x, p = self.p, \
            eps_e = self.eps_e, phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)

        sys = pecas.systems.ExplODE(t = self.t, u = self.u, x = self.x, \
            p = self.p, eps_e = self.eps_e, phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)

        sys = pecas.systems.ExplODE(t = self.t, u = self.u, x = self.x,\
            p = self.p, eps_e = self.eps_e, eps_u = self.eps_u, \
            phi = self.phi, f = self.f)
        sys.show_system_information(showEquations = True)

        self.assertRaises(TypeError, pecas.systems.ExplODE)
        self.assertRaises(TypeError, pecas.systems.ExplODE, x = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, p = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, w = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, phi = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, f = None)

        # while explicit time dependecy is not allowed:

        self.assertRaises(NotImplementedError, pecas.systems.ExplODE, \
            t = self.t, u = self.u, x = self.x, \
            p = self.p, eps_e = self.eps_e, phi = self.phi, f = self.t)


    def test_implade_system_init(self):

        self.assertRaises(NotImplementedError, pecas.systems.ImplDAE)
    