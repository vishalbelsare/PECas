#!/usr/bin/python

import pylab as pl
import casadi as ca

import pecas
# import colcas as co

# ex1.py: A first example to show the general usage of ColCas.

# (model and data taken from Diehl, Moritz: Course on System Identification,
# Exercises 5 and 6, IMTEK, University of Freiburg, 2014/2015)

data = pl.loadtxt('data_ex34.txt')

# Measurement times

timegrid = data[:, 0]

# Measured velocity

yN = data[:, 1]

# Control input (dutycycle)

uN = data[:-1, 2]

x = ca.SX.sym("x", 1)
p = ca.SX.sym("p", 3)
u = ca.SX.sym("u", 1)

f = p[0] * u - p[1] - p[2] * x

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, y = y)

odesetup = pecas.setups.ODEsetup(system = odesys, timegrid = timegrid,
    umin = uN, umax = uN, uinit = uN, \
    x0min = yN[0], x0max = yN[0], \
    xNmin = yN[-1:], xNmax = yN[-1:], \
    pmin = [10.0, 0.0, 0.4], pmax = [10.0, 2, 0.7], \
    pinit = [10.0, 0.08, 0.5])

stdyN = 0.01 * pl.ones((100, 1))

lsqprob = pecas.LSq(pesetup=odesetup, yN=yN, stdyN = stdyN, stds = [1e-3])
lsqprob.run_parameter_estimation()

phat = odesetup.V()(lsqprob.Vhat)["P"]
print phat

pl.scatter(timegrid, yN)
pl.plot(timegrid, ca.vertcat(odesetup.V()(lsqprob.Vhat)["X",:,0]))
pl.show()
