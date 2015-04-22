#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the classes fo system definitions

import pecas
import casadi as ca

import unittest

class TestSystemsInit(unittest.TestCase):

    def test_basic_system_init(self):

        self.t = ca.SX.sym("t", 1)
        self.u = ca.SX.sym("u", 1)
        self.p = ca.SX.sym("p", 1)
        self.y = ca.SX.sym("y", 1)
        self.g = ca.SX.sym("g", 1)

        pecas.systems.BasicSystem(p = self.p, y = self.y)
        pecas.systems.BasicSystem(t = self.t, p = self.p, y = self.y)
        pecas.systems.BasicSystem(t = self.t, u = self.u, p = self.p, \
            y = self.y)
        pecas.systems.BasicSystem(t = self.t, u = self.u, p = self.p, \
            y = self.y, g = self.g)

        self.assertRaises(TypeError, pecas.systems.BasicSystem)
        self.assertRaises(TypeError, pecas.systems.BasicSystem, p = None)
        self.assertRaises(TypeError, pecas.systems.BasicSystem, y = None)


    def test_explode_system_init(self):

        self.t = ca.SX.sym("t", 1)
        self.u = ca.SX.sym("u", 1)
        self.x = ca.SX.sym("x", 1)
        self.p = ca.SX.sym("p", 1)
        self.w = ca.SX.sym("w", 1)
        self.y = ca.SX.sym("y", 1)
        self.f = ca.SX.sym("f", 1)

        pecas.systems.ExplODE(x = self.x, p = self.p, w = self.w, \
            y = self.y, f = self.f)
        pecas.systems.ExplODE(t = self.t, x = self.x, p = self.p, \
            w = self.w, y = self.y, f = self.f)
        pecas.systems.ExplODE(t = self.t, u = self.u, x = self.x, p = self.p, \
            w = self.w, y = self.y, f = self.f)

        self.assertRaises(TypeError, pecas.systems.ExplODE)
        self.assertRaises(TypeError, pecas.systems.ExplODE, x = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, p = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, w = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, y = None)
        self.assertRaises(TypeError, pecas.systems.ExplODE, f = None)


    def test_implade_system_init(self):

        self.assertRaises(NotImplementedError, pecas.systems.ImplDAE)
    