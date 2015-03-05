#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat

class BasicSystem(object):

    def __init__(self, \
                 t = ca.SX.sym("t", 1), \
                 u = ca.SX.sym("u", 0), \
                 p = None, \
                 y = None, \
                 g = ca.SX.sym("g", 0)):

        if not all(isinstance(arg, (ca.casadi.SX, ca.casadi.MX)) for \
            arg in [t, u, p, y, g]):

            raise TypeError("Input arguments must be CasADi symbolic types.")

        self.v = cat.struct_MX([
                (
                    cat.entry("t", expr = t),
                    cat.entry("u", expr = u),
                    cat.entry("p", expr = p)
                )
            ])

        self.fcn = cat.struct_MX([
                (
                    cat.entry("y", expr = y),
                    cat.entry("g", expr = g)
                )
            ])


class ExplODE(BasicSystem):

    def __init__(self, \
                 t = ca.SX.sym("t", 1),
                 u = ca.SX.sym("u", 0), \
                 x = None, \
                 p = None, \
                 y = None, \
                 f = None, \
                 g = ca.SX.sym("g", 0)):

        if not all(isinstance(arg, (ca.casadi.SX, ca.casadi.MX)) for \
            arg in [t, u, x, p, y, f, g]):

            raise TypeError("Input arguments must be CasADi symbolic types.")

        self.v = cat.struct_MX([
                (
                    cat.entry("t", expr = t),
                    cat.entry("u", expr = u),
                    cat.entry("x", expr = x),
                    cat.entry("p", expr = p)
                )
            ])

        self.fcn = cat.struct_MX([
                (
                    cat.entry("y", expr = y),
                    cat.entry("f", expr = f),
                    cat.entry("g", expr = g)
                )
            ])
