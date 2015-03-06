#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test the ODE's setup mehthod (collocation struct builder)

import casadi as ca
import pylab as pl
import pecas
from nose.tools import assert_raises

def test_ode_setup():

    x = ca.SX.sym("x", 2)
    p = ca.SX.sym("p", 4)
    u = ca.SX.sym("u", 0)

    f = ca.vertcat( \
        [-p[0] * x[0] + p[1] * x[0] * x[1], 
        p[2] * x[1] - p[3] * x[0] * x[1]])

    y = x

    odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, y = y)

    # Test valid input dimensions for timegrid

    timegrid = pl.linspace(0, 10, 11)

    pecas.setupmethods.ODEsetup(system = odesys, timegrid = timegrid)

    timegrid = timegrid.T

    pecas.setupmethods.ODEsetup(system = odesys, timegrid = timegrid)

    # Support the wrong system type

    bssys = pecas.systems.BasicSystem(p = p, y = p)
    assert_raises(TypeError, pecas.setupmethods.ODEsetup, system = bssys, \
        timegrid = timegrid)

    # Test an invalid input dimension for timegrid

    assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
        system = odesys, timegrid = timegrid[:-1].reshape(2, 5))

    # Test some wrong values for p-arguments

    pwarglist = [[0, 1, 2], [[2, 3], [2, 3]], pl.asarray([1, 2, 3]), \
        pl.asarray([[2, 3], [2, 3]])]

    for parg in pwarglist:

        assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
            system = odesys, timegrid = timegrid, pinit = parg)
        assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
            system = odesys, timegrid = timegrid, pmin = parg)
        assert_raises(ValueError, pecas.setupmethods.ODEsetup, \
            system = odesys, timegrid = timegrid, pmax = parg)

    # Test some correct values for p-arguments

    parglist = [None, [0, 1, 2, 3], pl.asarray([1, 2, 3, 4]), \
        pl.asarray([1, 2, 3, 4]).T, pl.asarray([[2], [3], [2], [3]])]

    for parg in parglist:

        pecas.setupmethods.ODEsetup( \
            system = odesys, timegrid = timegrid, pinit = parg)
        pecas.setupmethods.ODEsetup( \
            system = odesys, timegrid = timegrid, pmin = parg)
        pecas.setupmethods.ODEsetup( \
            system = odesys, timegrid = timegrid, pmax = parg)



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
