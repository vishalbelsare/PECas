#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import testing

import casadi as ca
import pecas

import unittest

class BSLsqSimTest(object):

    def test_sim(self):

        # There is no simulation for BasicSystem

        pass


class ODELsqSimTest(object):

    def test_sim(self):

        # Run simulation and assure that the results is correct

        self.lsqpe = pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = self.wwe, wwu = self.wwu)

        self.lsqpe.run_simulation(x0 = self.yN[0, :], psim = self.phat)
