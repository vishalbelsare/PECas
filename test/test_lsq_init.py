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
import numpy as np
import pecas

import unittest

class NDLsqInitTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        pecas.LSq(system = self.ndsys, \
            tu = np.atleast_2d(self.tu).T, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        pecas.LSq(system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN.T, wv = self.wv)

        pecas.LSq(system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv.T)


    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, system = self.ndsys, \
            tu = np.zeros((self.tu.shape[0] - 1, self.tu.shape[0] - 1)) , \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(ValueError, pecas.LSq, system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = np.atleast_2d(self.yN)[:, :-1], wv = self.wv)

        self.assertRaises(ValueError, pecas.LSq, system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = np.atleast_2d(self.wv)[:-1])

        self.assertRaises(ValueError, pecas.LSq, system = self.ndsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = np.atleast_2d(self.wv)[:-1])

    def test_invalid_system_input(self):

        self.assertRaises(NotImplementedError, pecas.LSq, system = "dummy", \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)


class ODELsqInitTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = np.atleast_2d(self.tu).T, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            ty = self.tu, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            ty = np.atleast_2d(self.tu).T, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN.T, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv.T, weps_e = self.weps_e, weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = [self.weps_e], weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = np.atleast_1d(self.weps_e), \
            weps_u = self.weps_u)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = [self.weps_u])

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, \
            weps_u = np.atleast_1d(self.weps_u))


    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = np.atleast_2d(self.yN)[:, :-1], \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            ty = np.zeros((self.tu.shape[0] - 1, self.tu.shape[0] - 1)) , \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = np.zeros((self.tu.shape[0] - 1, self.tu.shape[0] - 1)), \
            uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = np.atleast_2d(self.wv)[:-1], weps_e = self.weps_e, \
            weps_u = self.weps_u)

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = np.atleast_2d(self.wv)[:-1], weps_e = self.weps_e[:-1], \
            weps_u = self.weps_u)


    def test_invalid_system_input(self):

        self.assertRaises(NotImplementedError, pecas.LSq, system = "dummy", \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, \
            weps_u = np.atleast_1d(self.weps_u))


        # self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
        #     tu = self.tu, uN = self.uN, \
        #     pmin = self.pmin, pmax = self.pmax, pinit = self.pinit, \
        #     xmin = self.xmin, xmax = self.xmax, xinit = self.xinit, \
        #     x0min = self.x0max, x0max = self.x0max, \
        #     xNmin = self.xNmin, xNmax = self.xNmax, \
        #     yN = self.yN, \
        #     wv = self.wv, weps_e = np.atleast_2d(self.weps_e)[:-1], \
        #     weps_u = self.weps_u)

        # self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
        #     tu = self.tu, uN = self.uN, \
        #     pmin = self.pmin, pmax = self.pmax, pinit = self.pinit, \
        #     xmin = self.xmin, xmax = self.xmax, xinit = self.xinit, \
        #     x0min = self.x0max, x0max = self.x0max, \
        #     xNmin = self.xNmin, xNmax = self.xNmax, \
        #     yN = self.yN, \
        #     wv = self.wv, weps_e = self.weps_e, \
        #     weps_u = np.atleast_2d(self.weps_u)[:-1])

