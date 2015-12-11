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

def sx_sym(name, dim1 = 1, dim2 = 1):

    return ca.SX.sym(name, dim1, dim2)


def sx_function(name, inputs, outputs):

    return ca.SXFunction(name, inputs, outputs)


def mx_sym(name, dim1 = 1, dim2 = 1):

    return ca.MX.sym(name, dim1, dim2)


def mx_function(name, inputs, outputs):

    return ca.MXFunction(name, inputs, outputs)


def dmatrix(dim1, dim2 = 1):

    return ca.DMatrix(dim1, dim2)


def depends_on(b, a):

    return ca.dependsOn(b, a)


def collocation_points(order, scheme):

    return ca.collocationPoints(order, scheme)


def vertcat(inputlist):

    return ca.vertcat(inputlist)


def veccat(inputlist):

    return ca.veccat(inputlist)


def horzcat(inputlist):

    return ca.horzcat(inputlist)
    