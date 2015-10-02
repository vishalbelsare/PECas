#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import numpy as np
import pecas

import unittest

class BSLsqInitTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv)

        pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN.T, wv = self.wv)

        pecas.LSq(system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = self.wv.T)


    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = np.atleast_2d(self.yN)[:, :-1], wv = self.wv)

        self.assertRaises(ValueError, pecas.LSq, system = self.bsys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            yN = self.yN, wv = np.atleast_2d(self.wv)[:-1])


class ODELsqInitTest(object):

    def test_valid_lsq_init(self):

        # Test valid setup cases

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = self.wwe, wwu = self.wwu)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN.T, \
            wv = self.wv, wwe = self.wwe, wwu = self.wwu)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv.T, wwe = self.wwe, wwu = self.wwu)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = [self.wwe], wwu = self.wwu)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = np.atleast_1d(self.wwe), wwu = self.wwu)

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = self.wwe, wwu = [self.wwu])

        pecas.LSq(system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = self.wv, wwe = self.wwe, wwu = np.atleast_1d(self.wwu))


    def test_invalid_lsq_init(self):

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = np.atleast_2d(self.yN)[:, :-1], \
            wv = self.wv, wwe = self.wwe, wwu = self.wwu)

        self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
            tu = self.tu, uN = self.uN, \
            pinit = self.pinit, \
            xinit = self.xinit, \
            yN = self.yN, \
            wv = np.atleast_2d(self.wv)[:-1], wwe = self.wwe, wwu = self.wwu)

        # self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
        #     tu = self.tu, uN = self.uN, \
        #     pmin = self.pmin, pmax = self.pmax, pinit = self.pinit, \
        #     xmin = self.xmin, xmax = self.xmax, xinit = self.xinit, \
        #     x0min = self.x0max, x0max = self.x0max, \
        #     xNmin = self.xNmin, xNmax = self.xNmax, \
        #     yN = self.yN, \
        #     wv = self.wv, wwe = np.atleast_2d(self.wwe)[:-1], wwu = self.wwu)

        # self.assertRaises(ValueError, pecas.LSq, system = self.odesys, \
        #     tu = self.tu, uN = self.uN, \
        #     pmin = self.pmin, pmax = self.pmax, pinit = self.pinit, \
        #     xmin = self.xmin, xmax = self.xmax, xinit = self.xinit, \
        #     x0min = self.x0max, x0max = self.x0max, \
        #     xNmin = self.xNmin, xNmax = self.xNmax, \
        #     yN = self.yN, \
        #     wv = self.wv, wwe = self.wwe, wwu = np.atleast_2d(self.wwu)[:-1])

