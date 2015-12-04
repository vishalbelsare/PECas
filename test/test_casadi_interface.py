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

import casadi as ca

import unittest

import pecas.interfaces.casadi_interface as ci

class MXSymbolic(unittest.TestCase):

    def setUp(self):

        self.name = "varname"
        self.dim1 = 3
        self.dim2 = 2


    def test_mx_sym_is_mx_sym_instance(self):

        mx_sym = ci.mx_sym(self.name, self.dim1)
        
        self.assertTrue(isinstance(mx_sym, ca.casadi.MX))


    def test_mx_sym_nodim(self):

        mx_sym = ci.mx_sym(self.name)
        
        self.assertEqual(mx_sym.shape, (1, 1))


    def test_mx_sym_1dim(self):

        mx_sym = ci.mx_sym(self.name, self.dim1)
        
        self.assertEqual(mx_sym.shape, (self.dim1, 1))


    def test_mx_sym_2dim(self):

        mx_sym = ci.mx_sym(self.name, self.dim1, self.dim2)
        
        self.assertEqual(mx_sym.shape, (self.dim1, self.dim2))


class MXFunction(unittest.TestCase):

    def setUp(self):

        self.name = "funcname"
        a = ca.MX.sym("a", 1)
        self.input = [a]
        self.output = [a**2]


    def test_mx_function_is_mx_function_instance(self):

        mx_function = ci.mx_function(self.name, self.input, self.output)
        
        self.assertTrue(isinstance(mx_function, ca.casadi.MXFunction))


    def test_mx_function_call(self):

        mx_function = ci.mx_function(self.name, self.input, self.output)
        b = 2
        c = 4

        self.assertTrue(mx_function([b]), c)


class DMatrix(unittest.TestCase):

    def setUp(self):

        self.dim1 = 3
        self.dim2 = 2


    def test_dmatrix_1dim(self):

        dmatrix = ci.dmatrix(self.dim1)
        
        self.assertEqual(dmatrix.shape, (self.dim1, 1))


    def test_dmatrix_2dim(self):

        dmatrix = ci.dmatrix(self.dim1, self.dim2)
        
        self.assertEqual(dmatrix.shape, (self.dim1, self.dim2))


class DependsOn(unittest.TestCase):

    def setUp(self):

        self.a = ca.MX.sym("a")


    def test_expression_does_depend(self):

        b = 2 * self.a
        self.assertTrue(ci.depends_on(b, self.a))


    def test_expression_does_not_depend(self):

        b = ca.MX.sym("b")
        self.assertFalse(ci.depends_on(b, self.a))


class CollocationPoints(unittest.TestCase):

    def setUp(self):

        self.order = 3
        self.scheme = "radau"
        # self.collocation_points = casadi.collocationPoints(3, "radau")
        self.collocation_points = \
            [0.0, 0.15505102572168222, 0.6449489742783179, 1.0]

    def test_collocation_points(self):

        self.assertEqual(ci.collocation_points(self.order, self.scheme), \
            self.collocation_points)
