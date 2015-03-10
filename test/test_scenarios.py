import casadi as ca
import pylab as pl
import pecas

import unittest
import test_ode_setup
import test_lsq_init
import test_lsq_run

class TestLotkaVolterra(unittest.TestCase, \
    test_ode_setup.ODESetupTest, \
    test_lsq_init.PESetupTest, \
    test_lsq_run.PERunTest):

    def setUp(self):

        # System

        self.x = ca.SX.sym("x", 2)
        self.p = ca.SX.sym("p", 4)
        self.u = ca.SX.sym("u", 0)

        self.f = ca.vertcat( \
            [-self.p[0] * self.x[0] + self.p[1] * self.x[0] * self.x[1], 
            self.p[2] * self.x[1] - self.p[3] * self.x[0] * self.x[1]])

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, f = self.f, y = self.y)

        # Inputs

        data = pl.array(pl.loadtxt("test/data_lotka_volterra.txt"))

        self.timegrid = data[:, 0]

        self.invalidpargs = [[0, 1, 2], [[2, 3], [2, 3]], \
            pl.asarray([1, 2, 3]), pl.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, [0, 1, 2, 3], pl.asarray([1, 2, 3, 4]), \
            pl.asarray([1, 2, 3, 4]).T, pl.asarray([[2], [3], [2], [3]])]

        self.invalidxargs = [pl.ones((self.x.size() - 1, self.timegrid.size)), \
            pl.ones((self.timegrid.size - 1, self.x.size()))]
        self.validxargs = [None, pl.ones((self.x.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2, 1], pl.ones(3), \
            pl.ones((1, 3)), pl.ones((3, 1))]
        self.validxbvpargs = [None, [1, 1], [[1], [1]], pl.ones((2,1)), \
            pl.ones((1, 2)), pl.ones(2)]

        # Since the supported values are never used, there is no case of
        # invalid u-arguments in this testcase

        self.invaliduargs = []
        self.validuargs = [None, [], [1, 2, 3]]

        # -- TODO! --
        # None of the checks will detect an invalidxpvbarg of
        # [[2, 1], [3]], since shape and size both fit.
        # --> How to check for this?

        self.yN = data[:, 1::2]
        self.stdyN = data[:, 2::2]
        self.stds = 1e-2

        self.phat = [1, 0.703278, 1, 0.342208]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, timegrid = self.timegrid, \
            x0min = [self.yN[0,0], self.yN[0,1]], \
            x0max = [self.yN[0,0], self.yN[0,1]], \
            pmin = [1.0, -pl.inf, 1.0, -pl.inf], \
            pmax = [1.0, pl.inf, 1.0, pl.inf], \
            pinit = [1.0, 0.5, 1.0, 1.0])


class Test1DVehicle(unittest.TestCase, \
    test_ode_setup.ODESetupTest, \
    test_lsq_init.PESetupTest, \
    test_lsq_run.PERunTest):

    def setUp(self):

        # System

        self.x = ca.SX.sym("x", 1)
        self.p = ca.SX.sym("p", 3)
        self.u = ca.SX.sym("u", 1)

        self.f = self.p[0] * self.u - self.p[1] - self.p[2] * self.x

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, f = self.f, y = self.y)

        # Inputs

        data = pl.array(pl.loadtxt("test/data_1d_vehicle.txt"))

        self.timegrid = data[:, 0]

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            pl.asarray([1, 2, 3, 4]), pl.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, [0, 1, 2], pl.asarray([3, 4, 5]), \
            pl.asarray([1, 2, 1]).T, pl.asarray([[2], [3], [4]])]

        self.invalidxargs = [pl.ones((self.x.size() - 1, self.timegrid.size)), \
            pl.ones((self.timegrid.size - 1, self.x.size()))]
        self.validxargs = [None, pl.ones((self.x.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2], pl.ones(3), \
            pl.ones((1, 3)), pl.ones((3, 1))]
        self.validxbvpargs = [None, [1], 1, pl.ones(1), pl.ones((1, 1))]

        self.invaliduargs = [pl.ones((self.u.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.u.size()))]
        self.validuargs = [None, pl.ones((self.u.size(), \
            self.timegrid.size - 1)), \
            pl.ones((self.timegrid.size - 1, self.u.size()))]

        self.yN = data[:, 1]
        self.stdyN = 0.01 * pl.ones(self.yN.shape)
        self.uN = data[:-1, 2]
        self.stds = 1e-3

        self.phat = [10.0, 0.000236, 0.614818]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, timegrid = self.timegrid,
            umin = self.uN, umax = self.uN, uinit = self.uN, \
            x0min = self.yN[0], x0max = self.yN[0], \
            xNmin = self.yN[-1:], xNmax = self.yN[-1:], \
            pmin = [10.0, 0.0, 0.4], pmax = [10.0, 2, 0.7], \
            pinit = [10.0, 0.08, 0.5])


class Test2DVehicle(unittest.TestCase, \
    # test_ode_setup.ODESetupTest, \
    # test_lsq_init.PESetupTest, \
    test_lsq_run.PERunTest):

    def setUp(self):

        # System

        self.x = ca.SX.sym("x", 6)
        self.p = ca.SX.sym("p", 6)
        self.u = ca.SX.sym("u", 2)

        self.f = ca.vertcat( \

            [self.x[3] * pl.cos(self.x[2] + 0.6 * self.p[0] * self.x[4]),

            self.x[3] * pl.sin(self.x[2] + 0.6 * self.p[0] * self.x[4]),

            self.x[3] * self.x[4] * 16.5 * self.p[1],

            11.5 * self.p[2] * self.x[5] \
                - 1.5 * self.p[3] * self.x[5] * self.x[3] \
                - 0.15 * self.p[4] * self.x[3]**2 - 0.5 * self.p[5] \
                - (self.x[3]  * self.x[5])**2 \
                * 16.5 * self.p[1] * 0.6 * self.p[0],

            self.u[0],

            self.u[1]])

        self.y = self.x[:4]

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, f = self.f, y = self.y)

        # Inputs

        data = pl.array(pl.loadtxt("test/data_2d_vehicle.txt"))

        self.timegrid = data[::100, 0]

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            pl.asarray([1, 2, 3, 4, 5]), pl.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, pl.asarray([3, 4, 5, 5, 6, 7]), \
            pl.asarray([1, 2, 1, 5, 6, 7]).T, \
            pl.asarray([[2], [3], [4], [5], [6], [7]]), \
            [1, 2, 3, 4, 5, 6]]

        self.invalidxargs = [pl.ones((self.x.size() - 1, self.timegrid.size)), \
            pl.ones((self.timegrid.size - 1, self.x.size()))]
        self.validxargs = [None, pl.ones((self.x.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2, 5], pl.ones(5), \
            pl.ones((1, 5)), pl.ones((4, 1))]
        self.validxbvpargs = [None, [1] * 6, pl.ones(6), pl.ones((1, 6)), \
            pl.ones((6, 1))]

        self.invaliduargs = [pl.ones((self.u.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.u.size()))]
        self.validuargs = [None, pl.ones((self.u.size(), \
            self.timegrid.size - 1)), \
            pl.ones((self.timegrid.size - 1, self.u.size()))]

        self.yN = data[::100, 1:5]
        self.stdyN = 0.01 * pl.ones(self.yN.shape)
        self.uN = data[:-1:100, 5:]
        print self.uN.shape
        self.stds = 1e-3

        self.phat = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, timegrid = self.timegrid,
            umin = self.uN, umax = self.uN, uinit = self.uN, \
            # pmin = [0.4, 16.0, 11.0, 1.0, 0.05, 0.4], \
            # pmax = [0.7, 18.0, 13.2, 3, 0.2, 0.75], \
            # pinit = [0.6, 16.5, 11.5, 2.7, 0.07, 0.7])
            pmin = [0.1] * 6, \
            pmax = [2] * 6, \
            pinit = [1] * 6)
