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

from numpy import testing

import casadi as ca
import pecas

import unittest

class BSLsqSimTest(object):

    def test_sim(self):

        # There is no simulation for BasicSystem

        self.lsqpe = pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        self.assertRaises(NotImplementedError, \
            self.lsqpe.run_simulation, x0 = None)


class ODELsqSimTest(object):

    def test_valid_sim(self):

        # Run simulation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        self.lsqpe.run_simulation(x0 = self.yN[0, :], psim = self.phat)


    def test_invalid_sim(self):

        # Run simulation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, weps_e = self.weps_e, weps_u = self.weps_u)

        # No x0

        self.assertRaises(ValueError, self.lsqpe.run_simulation, \
            x0 = None, psim = self.phat)

        # Wrong dimension for x0

        self.assertRaises(ValueError, self.lsqpe.run_simulation, \
            x0 = self.yN[0, :-1], psim = self.phat)

        # No psim

        self.assertRaises(AttributeError, self.lsqpe.run_simulation, \
            x0 = self.yN[0, :])

        # Wrong dimension for psim

        self.assertRaises(ValueError, self.lsqpe.run_simulation, \
            x0 = self.yN[0, :], psim = self.phat[:-1])

        # Unsupported integrator

        self.assertRaises(RuntimeError, self.lsqpe.run_simulation, \
            x0 = self.yN[0, :], psim = self.phat, method = "dummy")
