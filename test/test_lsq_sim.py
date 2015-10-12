#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
