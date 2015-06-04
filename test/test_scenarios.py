import casadi as ca
import numpy as np
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

        self.u = ca.MX.sym("u", 1)
        self.p = ca.MX.sym("p", 1)

        self.y = self.u * self.p

        self.bsys = pecas.systems.BasicSystem(u = self.u, p = self.p, \
            y = self.y)

        # Inputs

        self.tu = np.linspace(0, 3, 4)

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            np.asarray([1, 2]), np.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, 1, [0], np.asarray([1]), \
            np.asarray([1]).T, np.asarray([[2]])]

        self.invaliduargs = [np.ones((self.u.size(), \
            self.tu.size - 1)), \
            np.ones((self.tu.size - 1, self.u.size()))]
        self.validuargs = [None, np.ones((self.u.size(), self.tu.size)), \
            np.ones((self.tu.size, self.u.size()))]

        self.yN = np.asarray([2.5, 4.1, 6.3, 8.2])
        self.wv = np.asarray([1.0 / 0.01] * 4)

        self.uN = (1. / 3.) * np.linspace(1, 4, 4)

        self.phat = [6.24]

        self.bssetup = pecas.setups.BSsetup( \
            system = self.bsys, tu = self.tu, \
            u = self.uN)


class TestBasicSystemConstraints(unittest.TestCase, \
    test_set_bounds_initials.BSSetBoundsInitialsTest, \
    test_lsq_init.BSPESetupTest, \
    test_lsq_run.BSPERunTest):

    def setUp(self):

        # System

        self.u = ca.MX.sym("u", 2)
        self.p = ca.MX.sym("p", 2)

        self.y = self.u[0] * self.p[0] + self.u[1] * self.p[1]**2
        self.g = (2 - ca.mul(self.p.T, self.p))
        self.pinit = [1, 1]

        self.bsys = pecas.systems.BasicSystem(u = self.u, p = self.p, \
            y = self.y, g = self.g)

        # Inputs

        self.tu = np.linspace(0, 3, 4)

        self.invalidpargs = [[0, 1, 2], [[2, 2, 3], [2, 2, 3]], \
            np.asarray([1, 2, 2]), np.asarray([[2, 3, 3], [2, 3, 3]])]
        self.validpargs = [None, [0, 1], np.asarray([1, 1]), \
            np.asarray([1, 2]).T, np.asarray([[2], [2]])]

        self.invaliduargs = [np.ones((self.u.size(), \
            self.tu.size - 1)), \
            np.ones((self.tu.size - 1, self.u.size()))]
        self.validuargs = [None, np.ones((self.u.size(), self.tu.size)), \
            np.ones((self.tu.size, self.u.size()))]

        self.yN = np.asarray([2.23947, 2.84568, 4.55041, 5.08583])
        self.wv = np.asarray([1.0 / (0.5**2)] * 4)

        self.uN = np.vstack([np.ones(4), np.linspace(1, 4, 4)])

        self.phat = [0.961943, 1.03666]

        self.bssetup = pecas.setups.BSsetup( \
            system = self.bsys, tu = self.tu, \
            u = self.uN, \
            pinit = self.pinit)


class TestLotkaVolterra(unittest.TestCase, \
    test_set_bounds_initials.ODESetBoundsInitialsTest, \
    test_lsq_init.ODEPESetupTest, \
    test_lsq_run.ODEPERunTest):

    # (model and data taken from Bock, Sager et al.: Uebungen Numerische
    # Mathematik II, Blatt 9, IWR, Universitaet Heidelberg, 2006)

    def setUp(self):

        # System

        self.x = ca.MX.sym("x", 2)
        self.p = ca.MX.sym("p", 4)
        self.u = ca.MX.sym("u", 0)

        # self.v = ca.MX.sym("v", 2)
        self.we = ca.MX.sym("we", 2)

        self.f = ca.vertcat( \
            [-self.p[0] * self.x[0] + self.p[1] * self.x[0] * self.x[1], 
            self.p[2] * self.x[1] - self.p[3] * self.x[0] * self.x[1]]) + \
            self.we

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, we = self.we, f = self.f, y = self.y)

        # Inputs

        data = np.array(np.loadtxt("test/data_lotka_volterra.txt"))

        self.tu = data[:, 0]

        self.invalidpargs = [[0, 1, 2], [[2, 3], [2, 3]], \
            np.asarray([1, 2, 3]), np.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, [0, 1, 2, 3], np.asarray([1, 2, 3, 4]), \
            np.asarray([1, 2, 3, 4]).T, np.asarray([[2], [3], [2], [3]])]

        self.invalidxargs = [np.ones((self.x.size() - 1, self.tu.size)), \
            np.ones((self.tu.size - 1, self.x.size()))]
        self.validxargs = [None, np.ones((self.x.size(), self.tu.size)), \
            np.ones((self.tu.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2, 1], np.ones(3), \
            np.ones((1, 3)), np.ones((3, 1))]
        self.validxbvpargs = [None, [1, 1], [[1], [1]], np.ones((2,1)), \
            np.ones((1, 2)), np.ones(2)]

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
        self.wwe = [1.0 / 1e-4, 1.0 / 1e-4]
        self.wwu = None

        # self.phat = [1, 0.703278, 1, 0.342208]
        self.phat = [1, 0.703902, 1, 0.342233]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, tu = self.tu, \
            x0min = [self.yN[0,0], self.yN[0,1]], \
            x0max = [self.yN[0,0], self.yN[0,1]], \
            pmin = [1.0, -np.inf, 1.0, -np.inf], \
            pmax = [1.0, np.inf, 1.0, np.inf], \
            pinit = [1.0, 0.5, 1.0, 1.0])


class Test1DVehicle(unittest.TestCase, \
    test_set_bounds_initials.ODESetBoundsInitialsTest, \
    test_lsq_init.ODEPESetupTest, \
    test_lsq_run.ODEPERunTest, \
    # test_covmat.CovMatTest, \
    ):

    # (model and data taken from Diehl, Moritz: Course on System Identification,
    # Exercises 5 and 6, SYSCOP, IMTEK, University of Freiburg, 2014/2015)

    def setUp(self):

        # System

        self.x = ca.MX.sym("x", 1)
        self.p = ca.MX.sym("p", 3)
        self.u = ca.MX.sym("u", 1)
        self.we = ca.MX.sym("we", 1)

        self.f = self.p[0] * self.u - self.p[1] - self.p[2] * self.x + self.we

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, we = self.we, f = self.f, y = self.y)

        # Inputs

        data = np.array(np.loadtxt("test/data_1d_vehicle.txt"))

        self.tu = data[:, 0]

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            np.asarray([1, 2, 3, 4]), np.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, [0, 1, 2], np.asarray([3, 4, 5]), \
            np.asarray([1, 2, 1]).T, np.asarray([[2], [3], [4]])]

        self.invalidxargs = [np.ones((self.x.size() - 1, self.tu.size)), \
            np.ones((self.tu.size - 1, self.x.size()))]
        self.validxargs = [None, np.ones((self.x.size(), self.tu.size)), \
            np.ones((self.tu.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2], np.ones(3), \
            np.ones((1, 3)), np.ones((3, 1))]
        self.validxbvpargs = [None, [1], 1, np.ones(1), np.ones((1, 1))]

        self.invaliduargs = [np.ones((self.u.size(), self.tu.size)), \
            np.ones((self.tu.size, self.u.size()))]
        self.validuargs = [None, np.ones((self.u.size(), \
            self.tu.size - 1)), \
            np.ones((self.tu.size - 1, self.u.size()))]

        self.yN = data[:, 1]
        self.wv = 1 / (0.01**2) * np.ones(self.yN.shape)
        self.uN = data[:-1, 2]
        self.wwe = 1 / 1e-4
        self.wwu = None

        # self.phat = [10.0, 0.000236, 0.614818]
        self.phat = [10, 0.0299196, 0.604329]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, tu = self.tu,
            u = self.uN, \
            x0min = self.yN[0], x0max = self.yN[0], \
            xNmin = self.yN[-1:], xNmax = self.yN[-1:], \
            pmin = [10.0, 0.0, 0.4], pmax = [10.0, 2, 0.7], \
            pinit = [10.0, 0.08, 0.5])


class Test2DVehicle(unittest.TestCase, \
    test_set_bounds_initials.ODESetBoundsInitialsTest, \
    test_lsq_init.ODEPESetupTest, \
    test_lsq_run.ODEPERunTest, \
    ):

    # (model and data taken from Verschueren, Robin: Design and implementation 
    # of a time-optimal controller for model race cars, KU Leuven, 2014)

    def setUp(self):

        # System

        self.x = ca.MX.sym("x", 4)
        self.p = ca.MX.sym("p", 6)
        self.u = ca.MX.sym("u", 2)
        self.we = ca.MX.sym("we", 4)

        self.f = ca.vertcat( \

            [self.x[3] * np.cos(self.x[2] + self.p[0] * self.u[0] + self.we[0]),

            self.x[3] * np.sin(self.x[2] + self.p[0] * self.u[0] + self.we[1]),

            self.x[3] * self.u[0] * self.p[1] + self.we[2],

            self.p[2] * self.u[1] \
                - self.p[3] * self.u[1] * self.x[3] \
                - self.p[4] * self.x[3]**2 \
                - self.p[5] \
                - (self.x[3] * self.u[0])**2 * self.p[1] * self.p[0] \
                + self.we[3]])

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
            p = self.p, we = self.we, f = self.f, y = self.y)

        # Inputs

        data = np.array(np.loadtxt( \
            "test/controlReadings_ACADO_MPC_rates_Betterweights.dat", \
            delimiter = ", ", skiprows = 1))

        self.tu = data[200:250, 1]

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            np.asarray([1, 2, 3, 4, 5]), np.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, np.asarray([3, 4, 5, 5, 6, 7]), \
            np.asarray([1, 2, 1, 5, 6, 7]).T, \
            np.asarray([[2], [3], [4], [5], [6], [7]]), \
            [1, 2, 3, 4, 5, 6]]

        self.invalidxargs = [np.ones((self.x.size() - 1, self.tu.size)), \
            np.ones((self.tu.size - 1, self.x.size()))]
        self.validxargs = [None, np.ones((self.x.size(), self.tu.size)), \
            np.ones((self.tu.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2, 5], np.ones(5), \
            np.ones((1, 5)), np.ones((3, 1))]
        self.validxbvpargs = [None, [1] * 4, np.ones(4), np.ones((1, 4)), \
            np.ones((4, 1))]

        self.invaliduargs = [np.ones((self.u.size(), self.tu.size)), \
            np.ones((self.tu.size, self.u.size()))]
        self.validuargs = [None, np.ones((self.u.size(), \
            self.tu.size - 1)), \
            np.ones((self.tu.size - 1, self.u.size()))]

        self.yN = data[200:250, [2, 4, 6, 8]]
        self.wv = 1 / (0.1**2) * np.ones(self.yN.shape)
        self.uN = data[200:249, [9, 10]]
        self.wwe = [1 / 1e-4] * 4
        self.wwu = None

        # self.phat = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]
        self.phat = [0.5, 17.06, 3.98281, -10, -7.57932, 3]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, tu = self.tu,
            u = self.uN, \
            pmin = [0.5, 17.06, 0.0, -10.0, -1000.0, -10.0], \
            pmax = [0.5, 17.06, 13.2, 200, 500, 3], \
            pinit = [0.5, 17.06, 11.5, 5, 0.07, 0.70])


class PedulumBar(unittest.TestCase, \
    test_set_bounds_initials.ODESetBoundsInitialsTest, \
    test_lsq_init.ODEPESetupTest, \
    test_lsq_run.ODEPERunTest, \
    ):

    def setUp(self):

        # System

        m = 1
        L = 3
        g = 9.81
        psi = np.pi/2

        # System

        self.x = ca.MX.sym("x", 2)
        self.p = ca.MX.sym("p", 1)
        self.u = ca.MX.sym("u", 1)

        self.f = ca.vertcat([ \
            
            self.x[1], \
            self.p[0]/(m*(L**2))*(self.u-self.x[0]) - g/L * np.sin(self.x[0]) \

            ])

        self.y = self.x

        self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, p = self.p, \
            f = self.f, y = self.y)

        # Inputs

        data = np.loadtxt('test/ex6data.txt')

        self.tu = data[:50, 0]

        self.invalidpargs = [[0, 1], [[2, 3], [2, 3]], \
            np.asarray([1, 2]), np.asarray([[2, 3], [2, 3]])]
        self.validpargs = [None, 1, [0], np.asarray([1]), \
            np.asarray([1]).T, np.asarray([[2]])]

        self.invalidxargs = [np.ones((self.x.size() - 1, self.tu.size)), \
            np.ones((self.tu.size - 1, self.x.size()))]
        self.validxargs = [None, np.ones((self.x.size(), self.tu.size)), \
            np.ones((self.tu.size, self.x.size()))]

        self.invalidxbvpargs = [[3, 2, 1], np.ones(3), \
            np.ones((1, 3)), np.ones((3, 1))]
        self.validxbvpargs = [None, [1, 1], [[1], [1]], np.ones((2,1)), \
            np.ones((1, 2)), np.ones(2)]

        self.invaliduargs = [np.ones((self.u.size(), self.tu.size)), \
            np.ones((self.tu.size, self.u.size()))]
        self.validuargs = [None, np.ones((self.u.size(), \
            self.tu.size - 1)), \
            np.ones((self.tu.size - 1, self.u.size()))]

        N = self.tu.size
        phim = data[:50, 1]
        wm = data[:50, 2]

        self.yN = np.array([phim, wm])
        self.uN = [psi] * (N-1)

        self.wv = np.array([

                1.0 / (np.ones(N)*np.std(phim, ddof=1)**2),
                1.0 / (np.ones(N)*np.std(wm, ddof=1)**2)

            ])

        self.wwe = None
        self.wwu = None

        self.phat = [2.98427]

        self.odesetup = pecas.setups.ODEsetup( \
            system = self.odesys, tu = self.tu,
            u = self.uN, \
            pinit = 1, pmax = 50, pmin = 0 )
