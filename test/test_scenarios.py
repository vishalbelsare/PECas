import casadi as ca
import pylab as pl
import pecas

import unittest
import test_set_bounds_initials
import test_lsq_init
import test_lsq_run

class TestBasicSystemNoConstraints(unittest.TestCase, \
    test_set_bounds_initials.BSSetBoundsInitialsTest, \
    test_lsq_init.BSPESetupTest, \
    test_lsq_run.BSPERunTest):

    _multiprocess_can_split_ = True

    def setUp(self):

        # System

        self.u = ca.SX.sym("u", 1)
        self.p = ca.SX.sym("p", 1)

        self.y = self.u * self.p

        self.bsys = pecas.systems.BasicSystem(u = self.u, p = self.p, \
            y = self.y)

        # Inputs

        self.timegrid = pl.linspace(0, 3, 4)

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            pl.asarray([1, 2]), pl.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, 1, [0], pl.asarray([1]), \
            pl.asarray([1]).T, pl.asarray([[2]])]

        self.invaliduargs = [pl.ones((self.u.size(), \
            self.timegrid.size - 1)), \
            pl.ones((self.timegrid.size - 1, self.u.size()))]
        self.validuargs = [None, pl.ones((self.u.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.u.size()))]

        self.yN = pl.asarray([2.5, 4.1, 6.3, 8.2])
        self.wv = pl.asarray([1.0 / 0.01] * 4)

        self.uN = (1. / 3.) * pl.linspace(1, 4, 4)

        self.phat = [6.24]

        self.bssetup = pecas.setups.BSsetup( \
            system = self.bsys, timegrid = self.timegrid, \
            umin = self.uN, umax = self.uN, uinit = self.uN)


class TestBasicSystemConstraints(unittest.TestCase, \
    test_set_bounds_initials.BSSetBoundsInitialsTest, \
    test_lsq_init.BSPESetupTest, \
    test_lsq_run.BSPERunTest):

    def setUp(self):

        # System

        self.u = ca.SX.sym("u", 2)
        self.p = ca.SX.sym("p", 2)

        self.y = self.u[0] * self.p[0] + self.u[1] * self.p[1]**2
        self.g = (2 - ca.mul(self.p.T, self.p))
        self.pinit = [1, 1]

        self.bsys = pecas.systems.BasicSystem(u = self.u, p = self.p, \
            y = self.y, g = self.g)

        # Inputs

        self.timegrid = pl.linspace(0, 3, 4)

        self.invalidpargs = [[0, 1, 2], [[2, 2, 3], [2, 2, 3]], \
            pl.asarray([1, 2, 2]), pl.asarray([[2, 3, 3], [2, 3, 3]])]
        self.validpargs = [None, [0, 1], pl.asarray([1, 1]), \
            pl.asarray([1, 2]).T, pl.asarray([[2], [2]])]

        self.invaliduargs = [pl.ones((self.u.size(), \
            self.timegrid.size - 1)), \
            pl.ones((self.timegrid.size - 1, self.u.size()))]
        self.validuargs = [None, pl.ones((self.u.size(), self.timegrid.size)), \
            pl.ones((self.timegrid.size, self.u.size()))]

        self.yN = pl.asarray([2.23947, 2.84568, 4.55041, 5.08583])
        self.wv = pl.asarray([1.0 / (0.5**2)] * 4)

        self.uN = pl.vstack([pl.ones(4), pl.linspace(1, 4, 4)])

        self.phat = [0.961943, 1.03666]

        self.bssetup = pecas.setups.BSsetup( \
            system = self.bsys, timegrid = self.timegrid, \
            umin = self.uN, umax = self.uN, uinit = self.uN, \
            pinit = self.pinit)


class TestLotkaVolterra(unittest.TestCase, \
    test_set_bounds_initials.ODESetBoundsInitialsTest, \
    test_lsq_init.ODEPESetupTest, \
    test_lsq_run.ODEPERunTest):

    # (model and data taken from Bock, Sager et al.: Uebungen Numerische
    # Mathematik II, Blatt 9, IWR, Universitaet Heidelberg, 2006)

    def setUp(self):

        # System

        self.x = ca.SX.sym("x", 2)
        self.p = ca.SX.sym("p", 4)
        self.u = ca.SX.sym("u", 0)

        # self.v = ca.SX.sym("v", 2)
        self.w = ca.SX.sym("w", 2)

        self.f = ca.vertcat( \
            [-self.p[0] * self.x[0] + self.p[1] * self.x[0] * self.x[1], 
            self.p[2] * self.x[1] - self.p[3] * self.x[0] * self.x[1]]) + self.w

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, w = self.w, f = self.f, y = self.y)

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
        self.wv = 1.0 / data[:, 2::2]**2
        self.ww = [1.0 / 1e-4, 1.0 / 1e-4]

        # self.phat = [1, 0.703278, 1, 0.342208]
        self.phat = [1, 0.703902, 1, 0.342233]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, timegrid = self.timegrid, \
            x0min = [self.yN[0,0], self.yN[0,1]], \
            x0max = [self.yN[0,0], self.yN[0,1]], \
            pmin = [1.0, -pl.inf, 1.0, -pl.inf], \
            pmax = [1.0, pl.inf, 1.0, pl.inf], \
            pinit = [1.0, 0.5, 1.0, 1.0])


# class Test1DVehicle(unittest.TestCase, \
#     test_set_bounds_initials.ODESetBoundsInitialsTest, \
#     test_lsq_init.ODEPESetupTest, \
#     test_lsq_run.ODEPERunTest):

#     # (model and data taken from Diehl, Moritz: Course on System Identification,
#     # Exercises 5 and 6, SYSCOP, IMTEK, University of Freiburg, 2014/2015)

#     def setUp(self):

#         # System

#         self.x = ca.SX.sym("x", 1)
#         self.p = ca.SX.sym("p", 3)
#         self.u = ca.SX.sym("u", 1)

#         self.f = self.p[0] * self.u - self.p[1] - self.p[2] * self.x

#         self.y = self.x

#         self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
#             p = self.p, f = self.f, y = self.y)

#         # Inputs

#         data = pl.array(pl.loadtxt("test/data_1d_vehicle.txt"))

#         self.timegrid = data[:, 0]

#         self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
#             pl.asarray([1, 2, 3, 4]), pl.asarray([[2, 3], [2, 3]])]
#         self.validpargs = [None, [0, 1, 2], pl.asarray([3, 4, 5]), \
#             pl.asarray([1, 2, 1]).T, pl.asarray([[2], [3], [4]])]

#         self.invalidxargs = [pl.ones((self.x.size() - 1, self.timegrid.size)), \
#             pl.ones((self.timegrid.size - 1, self.x.size()))]
#         self.validxargs = [None, pl.ones((self.x.size(), self.timegrid.size)), \
#             pl.ones((self.timegrid.size, self.x.size()))]

#         self.invalidxbvpargs = [[3, 2], pl.ones(3), \
#             pl.ones((1, 3)), pl.ones((3, 1))]
#         self.validxbvpargs = [None, [1], 1, pl.ones(1), pl.ones((1, 1))]

#         self.invaliduargs = [pl.ones((self.u.size(), self.timegrid.size)), \
#             pl.ones((self.timegrid.size, self.u.size()))]
#         self.validuargs = [None, pl.ones((self.u.size(), \
#             self.timegrid.size - 1)), \
#             pl.ones((self.timegrid.size - 1, self.u.size()))]

#         self.yN = data[:, 1]
#         self.wy = 1/(0.01**2) * pl.ones(self.yN.shape)
#         self.uN = data[:-1, 2]
#         self.ws = 1 / 1e-6

#         self.phat = [10.0, 0.000236, 0.614818]

#         self.odesetup = pecas.setups.ODEsetup( \
#             system = self.odesys, timegrid = self.timegrid,
#             umin = self.uN, umax = self.uN, uinit = self.uN, \
#             x0min = self.yN[0], x0max = self.yN[0], \
#             xNmin = self.yN[-1:], xNmax = self.yN[-1:], \
#             pmin = [10.0, 0.0, 0.4], pmax = [10.0, 2, 0.7], \
#             pinit = [10.0, 0.08, 0.5])


# class Test2DVehicle(unittest.TestCase, \
#     test_set_bounds_initials.ODESetBoundsInitialsTest, \
#     test_lsq_init.ODEPESetupTest, \
#     # test_lsq_run.ODEPERunTest, \
#     ):

#     # (model and data taken from Verschueren, Robin: Design and implementation 
#     # of a time-optimal controller for model race cars, KU Leuven, 2014)

#     def setUp(self):

#         # System

#         self.x = ca.SX.sym("x", 4)
#         self.p = ca.SX.sym("p", 3)
#         self.u = ca.SX.sym("u", 2)

#         self.f = ca.vertcat( \

#             # [self.x[3] * pl.cos(self.x[2] + self.p[0] * self.u[0]),
#             [self.x[3] * pl.cos(self.x[2] + 0.5 * self.u[0]),

#             # self.x[3] * pl.sin(self.x[2] + self.p[0] * self.u[0]),
#             self.x[3] * pl.sin(self.x[2] + 0.5 * self.u[0]),

#             # self.x[3] * self.u[0] * self.p[1],
#             self.x[3] * self.u[0] * 17.06,

#             # self.p[2] * self.u[1] \
#             #     - self.p[3] * self.u[1] * self.x[3] \
#             #     - self.p[4] * self.x[3]**2 - self.p[5] \
#             #     - (self.x[3]  * self.u[0])**2 \
#             #     * self.p[1] * self.p[0]])

#             self.p[0] * self.u[1] \
#                 - 2.17 * self.u[1] * self.x[3] \
#                 - self.p[1] * self.x[3]**2 - self.p[2] \
#                 - (self.x[3]  * self.u[0])**2 \
#                 * 17.06 * 0.5])

#         self.y = self.x

#         self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
#             p = self.p, f = self.f, y = self.y)

#         # Inputs

#         data = pl.array(pl.loadtxt( \
#             "test/controlReadings_ACADO_MPC_rates_Betterweights.dat", \
#             delimiter = ", ", skiprows = 1))

#         self.timegrid = data[200:250, 0]

#         self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
#             pl.asarray([1, 2, 3, 4, 5]), pl.asarray([[2, 3], [2, 3]])]
#         self.validpargs = [None, pl.asarray([3, 4, 5, 5, 6, 7]), \
#             pl.asarray([1, 2, 1, 5, 6, 7]).T, \
#             pl.asarray([[2], [3], [4], [5], [6], [7]]), \
#             [1, 2, 3, 4, 5, 6]]

#         self.invalidxargs = [pl.ones((self.x.size() - 1, self.timegrid.size)), \
#             pl.ones((self.timegrid.size - 1, self.x.size()))]
#         self.validxargs = [None, pl.ones((self.x.size(), self.timegrid.size)), \
#             pl.ones((self.timegrid.size, self.x.size()))]

#         self.invalidxbvpargs = [[3, 2, 5], pl.ones(5), \
#             pl.ones((1, 5)), pl.ones((3, 1))]
#         self.validxbvpargs = [None, [1] * 4, pl.ones(4), pl.ones((1, 4)), \
#             pl.ones((4, 1))]

#         self.invaliduargs = [pl.ones((self.u.size(), self.timegrid.size)), \
#             pl.ones((self.timegrid.size, self.u.size()))]
#         self.validuargs = [None, pl.ones((self.u.size(), \
#             self.timegrid.size - 1)), \
#             pl.ones((self.timegrid.size - 1, self.u.size()))]

#         self.yN = data[200:250, [2, 4, 6, 8]]
#         self.wy = 0.01 * pl.ones(self.yN.shape)
#         self.wy[:, 3] = 1e-1
#         self.uN = data[200:249, [9, 10]]
#         self.ws = 1e-2

#         self.phat = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

#         self.odesetup = pecas.setups.ODEsetup( \
#             system = self.odesys, timegrid = self.timegrid,
#             umin = self.uN, umax = self.uN, uinit = self.uN, \
#             pmin = [0.5, 17.06, 0.0, 0, 0.0, 0.0], \
#             pmax = [0.5, 17.06, 13.2, 10, 10, 3], \
#             pinit = [0.5, 17.06, 11.5, 5, 0.07, 0.7])
