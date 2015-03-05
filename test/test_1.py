#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dummy_test.py: just something to get the testing working

import pecas
import casadi as ca
from nose.tools import assert_raises

def test_define_basic_systems():

    t = ca.SX.sym("t", 1)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 1)
    y = ca.SX.sym("y", 1)
    g = ca.SX.sym("g", 1)

    pecas.systems.BasicSystem(p = p, y = y)
    pecas.systems.BasicSystem(t = t, p = p, y = y)
    pecas.systems.BasicSystem(t = t, u = u, p = p, y = y)
    pecas.systems.BasicSystem(t = t, u = u, p = p, y = y, g =g)

    assert_raises(TypeError, pecas.systems.BasicSystem)
    assert_raises(TypeError, pecas.systems.BasicSystem, p = None)
    assert_raises(TypeError, pecas.systems.BasicSystem, y = None)

def test_define_explode_systems():

    t = ca.SX.sym("t", 1)
    u = ca.SX.sym("u", 1)
    x = ca.SX.sym("x", 1)
    p = ca.SX.sym("p", 1)
    y = ca.SX.sym("y", 1)
    f = ca.SX.sym("f", 1)
    g = ca.SX.sym("g", 1)

    pecas.systems.ExplODE(x = x, p = p, y = y, f = f)
    pecas.systems.ExplODE(t = t, x = x, p = p, y = y, f = f)
    pecas.systems.ExplODE(t = t, u = u, x = x, p = p, y = y, f = f)
    pecas.systems.ExplODE(t = t, u = u, x = x, p = p, y = y, f = f, g = g)

    assert_raises(TypeError, pecas.systems.ExplODE)
    assert_raises(TypeError, pecas.systems.ExplODE, x = None)
    assert_raises(TypeError, pecas.systems.ExplODE, p = None)
    assert_raises(TypeError, pecas.systems.ExplODE, y = None)
    assert_raises(TypeError, pecas.systems.ExplODE, f = None)
