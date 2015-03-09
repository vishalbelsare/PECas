#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test the ODE's setup mehthod (collocation struct builder)

import casadi as ca
import pylab as pl
import pecas

import unittest

class ODESetupTest(object):

    def test_valid_timegrid_inputs(self):

        # Test valid input dimensions for timegrid

        pecas.setupmethods.ODEsetup(system = self.odesys, \
            timegrid = self.timegrid)
        pecas.setupmethods.ODEsetup(system = self.odesys, timegrid = \
            self.timegrid.T)


    def test_invalid_systems_input(self):

        # Support an invalid systems-type

        bssys = pecas.systems.BasicSystem(p = self.p, y = self.p)
        self.assertRaises(TypeError, pecas.setupmethods.ODEsetup, \
            system = bssys, timegrid = self.timegrid)


    def test_invalid_parameter_bounds_and_initials(self):

        # Test some invalid values for p-arguments

        for parg in self.invalidpargs:

            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, pinit = parg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, pmin = parg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, pmax = parg)


    def test_valid_parameter_bounds_and_initials(self):

        # Test some valid values for p-arguments

        for parg in self.validpargs:

            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, pinit = parg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, pmin = parg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, pmax = parg)


    def test_invalid_state_bounds_and_initials(self):

        # Test some invalid values for x-arguments

        for xarg in self.invalidxargs:

            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xinit = xarg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xmin = xarg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xmax = xarg)


    def test_valid_state_bounds_and_initials(self):

        # Test some valid values for x-arguments

        for xarg in self.validxargs:

            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xinit = xarg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xmin = xarg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xmax = xarg)


    def test_invalid_state_bvp_inputs(self):

        # Test some invalid values for xbvp-arguments

        for xbvparg in self.invalidxbvpargs:

            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, x0min = xbvparg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, x0max = xbvparg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xNmin = xbvparg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, xNmax = xbvparg)
    

    def test_valid_state_bvp_inputs(self):

        # Test some valid values for xbvp-arguments

        for xbvparg in self.validxbvpargs:

            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, x0min = xbvparg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, x0max = xbvparg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xNmin = xbvparg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, xNmax = xbvparg)


    def test_invalid_control_bounds_and_initials_inputs(self):

        # Test some invalid values for u-arguments       

        for uarg in self.invaliduargs:

            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, uinit = uarg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, umin = uarg)
            self.assertRaises(ValueError, pecas.setupmethods.ODEsetup, \
                system = self.odesys, timegrid = self.timegrid, umax = uarg)
    

    def test_valid_control_bounds_and_initials_inputs(self):

        # Test some valid values for u-arguments

        for uarg in self.validuargs:

            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, uinit = uarg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, umin = uarg)
            pecas.setupmethods.ODEsetup( \
                system = self.odesys, timegrid = self.timegrid, umax = uarg)


# class TestLotkaVolterra(unittest.TestCase, BaseClassODESetupTest):

#     def setUp(self):

#         # System

#         self.x = ca.SX.sym("x", 2)
#         self.p = ca.SX.sym("p", 4)
#         self.u = ca.SX.sym("u", 0)

#         self.f = ca.vertcat( \
#             [-self.p[0] * self.x[0] + self.p[1] * self.x[0] * self.x[1], 
#             self.p[2] * self.x[1] - self.p[3] * self.x[0] * self.x[1]])

#         self.y = self.x

#         self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
#             p = self.p, f = self.f, y = self.y)

#         # Inputs

#         self.timegrid = pl.linspace(0, 10, 11)

#         self.invalidpargs = [[0, 1, 2], [[2, 3], [2, 3]], \
#             pl.asarray([1, 2, 3]), pl.asarray([[2, 3], [2, 3]])]
#         self.validpargs = [None, [0, 1, 2, 3], pl.asarray([1, 2, 3, 4]), \
#             pl.asarray([1, 2, 3, 4]).T, pl.asarray([[2], [3], [2], [3]])]

#         self.invalidxargs = [pl.ones((self.x.size() - 1, self.timegrid.size)), \
#             pl.ones((self.timegrid.size - 1, self.x.size()))]
#         self.validxargs = [None, pl.ones((self.x.size(), self.timegrid.size)), \
#             pl.ones((self.timegrid.size, self.x.size()))]

#         self.invalidxbvpargs = [[3, 2, 1], pl.ones(3), \
#             pl.ones((1, 3)), pl.ones((3, 1))]
#         self.validxbvpargs = [None, [1, 1], [[1], [1]], pl.ones((2,1)), \
#             pl.ones((1, 2)), pl.ones(2)]

#         # Since the supported values are never used, there is no case of
#         # invalid u-arguments in this testcase

#         self.invaliduargs = []
#         self.validuargs = [None, [], [1, 2, 3]]

#         # -- TODO! --
#         # None of the checks will detect an invalidxpvbarg of
#         # [[2, 1], [3]], since shape and size both fit.
#         # --> How to check for this?


# class Test1DVehicle(unittest.TestCase, BaseClassODESetupTest):

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

#         self.timegrid = pl.linspace(0, 6, 100)

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


# class Test2DVehicle(unittest.TestCase, BaseClassODESetupTest):

#     def setUp(self):

#         # System

#         self.x = ca.SX.sym("x", 6)
#         self.p = ca.SX.sym("p", 6)
#         self.u = ca.SX.sym("u", 2)

#         self.f = ca.vertcat( \
#             [self.x[3] * pl.cos(self.x[2] + self.p[0] * self.x[4]),
#             self.x[3] * pl.sin(self.x[2] + self.p[0] * self.x[4]),
#             self.x[3] * self.x[4] * self.p[1],
#             self.p[2] * self.x[5] - self.p[3] * self.x[5] * self.x[3] \
#                 - self.p[4] * self.x[3]**2 - self.p[5] \
#                 - (self.x[3]  * self.x[5])**2 * self.p[1] * self.p[0],
#             self.u[0],
#             self.u[1]
#             ])

#         self.y = self.x[:4]

#         self.odesys = pecas.systems.ExplODE(x = self.x, u = self.u, \
#             p = self.p, f = self.f, y = self.y)

#         # Inputs

#         self.timegrid = pl.linspace(0, 6, 100)

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
#             pl.ones((1, 5)), pl.ones((4, 1))]
#         self.validxbvpargs = [None, [1] * 6, pl.ones(6), pl.ones((1, 6)), \
#             pl.ones((6, 1))]

#         self.invaliduargs = [pl.ones((self.u.size(), self.timegrid.size)), \
#             pl.ones((self.timegrid.size, self.u.size()))]
#         self.validuargs = [None, pl.ones((self.u.size(), \
#             self.timegrid.size - 1)), \
#             pl.ones((self.timegrid.size - 1, self.u.size()))]

# if __name__ == '__main__':
#     unittest.main()

# def test_ode_setup():

#     x = ca.SX.sym("x", 2)
#     p = ca.SX.sym("p", 4)
#     u = ca.SX.sym("u", 0)

#     f = ca.vertcat( \
#         [-p[0] * x[0] + p[1] * x[0] * x[1], 
#         p[2] * x[1] - p[3] * x[0] * x[1]])

#     y = x

#     odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, y = y)

#     # Test valid input dimensions for timegrid

#     timegrid = pl.linspace(0, 10, 11)

#     pecas.setupmethods.ODEsetup(system = odesys, timegrid = timegrid)

#     timegrid = timegrid.T

#     pecas.setupmethods.ODEsetup(system = odesys, timegrid = timegrid)

#     # Support the invalid system type

#     bssys = pecas.systems.BasicSystem(p = p, y = p)
#     assert_raises(TypeError, pecas.setupmethods.ODEsetup, system = bssys, \
#         timegrid = timegrid)

#     # Test an invalid input dimension for timegrid

#     assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
#         system = odesys, timegrid = timegrid[:-1].reshape(2, 5))

    # Test some invalid values for p-arguments

    # pwarglist = [[0, 1, 2], [[2, 3], [2, 3]], pl.asarray([1, 2, 3]), \
    #     pl.asarray([[2, 3], [2, 3]])]

    # for parg in pwarglist:

    #     assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
    #         system = odesys, timegrid = timegrid, pinit = parg)
    #     assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
    #         system = odesys, timegrid = timegrid, pmin = parg)
    #     assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
    #         system = odesys, timegrid = timegrid, pmax = parg)

    # # Test some valid values for p-arguments

    # parglist = [None, [0, 1, 2, 3], pl.asarray([1, 2, 3, 4]), \
    #     pl.asarray([1, 2, 3, 4]).T, pl.asarray([[2], [3], [2], [3]])]

    # for parg in parglist:

    #     pecas.setupmethods.ODEsetup( \
    #         system = odesys, timegrid = timegrid, pinit = parg)
    #     pecas.setupmethods.ODEsetup( \
    #         system = odesys, timegrid = timegrid, pmin = parg)
    #     pecas.setupmethods.ODEsetup( \
    #         system = odesys, timegrid = timegrid, pmax = parg)



# data = pl.array(pl.loadtxt("data_ex33.txt"))

# timegrid = data[:, 0]
# yN = data[:, 1::2]
# stdyN = data[:, 2::2]

# pmin = pl.array([1.0, -pl.inf, 1.0, -pl.inf])
# print pmin

# xmin = -pl.inf*pl.ones((11, 2)).T

# U = pl.array([])

# odesol = pecas.setupmethods.ODEsetup( \
#     system = op, timegrid = timegrid, \
#     x0min = [yN[0,0], yN[0,1]], \
#     x0max = [yN[0,0], yN[0,1]], \
#     xmin = xmin, \
#     umin = U, umax = U, uinit = U, \
#     pmin = pmin, \
#     pmax = [1.0, pl.inf, 1.0, pl.inf], \
#     pinit = [1.0, 0.5, 1.0, 1.0])


# lsqprob = pecas.LSq(pesetup=odesol, yN=yN, stdyN = stdyN)
# lsqprob.run_parameter_estimation()

# phat = odesol.V()(lsqprob.Vhat)["P"]
# print phat

# pl.scatter(timegrid, yN[:,0], color = 'b')
# pl.scatter(timegrid, yN[:,1], color = 'r')

# pl.plot(timegrid, ca.vertcat(odesol.V()(lsqprob.Vhat)["X",:,0])[::2], \
#     color='b', ls = '--')
# pl.plot(timegrid, ca.vertcat(odesol.V()(lsqprob.Vhat)["X",:,0])[1::2], \
#     color='r', ls = '--')

# tgridint = pl.linspace(0,10,1000)
# ode = ca.SXFunction(ca.daeIn(x=x, p=p), ca.daeOut(ode=f))
# integrator = ca.Integrator("cvodes", ode)

# simulator = ca.Simulator(integrator, tgridint)
# simulator.init()

# simulator.setInput([1, 1], "x0")
# simulator.setInput(phat, "p")
# simulator.evaluate()

# pl.plot(tgridint, pl.squeeze(simulator.getOutput("xf")[0,:]), color='b')
# pl.plot(tgridint, pl.squeeze(simulator.getOutput("xf")[1,:]), color='r')
# pl.grid()
# pl.show()
