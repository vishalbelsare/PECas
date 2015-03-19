import casadi as ca
import pylab as pl
import pecas

import unittest
import test_lsq_run

import test_scenarios

import time

x = ca.SX.sym("x", 4)
u = ca.SX.sym("u", 2)

p = [0.5, 17.06, 0.00487815, 3.12938, -201.919, 0.000890558]

f = ca.vertcat( \

    [x[3] * pl.cos(x[2] + p[0] * u[0]),
    # [x[3] * pl.cos(x[2] + 0.5 * u[0]),

    x[3] * pl.sin(x[2] + p[0] * u[0]),
    # x[3] * pl.sin(x[2] + 0.5 * u[0]),

    x[3] * u[0] * p[1],
    # x[3] * u[0] * p[3],

    p[2] * u[1] \
        - p[3] * u[1] * x[3] \
        - p[4] * x[3]**2 \
        - p[5] \
        - (x[3] * u[0])**2 * p[1] * p[0]])

ode = ca.SXFunction(ca.daeIn(x=x, p=u), ca.daeOut(ode=f))
integrator = ca.Integrator("cvodes", ode)
integrator.init()

data = pl.array(pl.loadtxt( \
    "controlReadings_ACADO_MPC_Betterweights.dat", \
    delimiter = ", ", skiprows = 1))

timegrid = pl.linspace(1, 150, 150)

yN = data[150:300, [2, 4, 6, 8]]
uN = data[150:299, [9, 10]]

ysim = pl.zeros(yN.shape)
ysim[0, :] = yN[0, :]

integrator.setInput(yN[0,:], "x0")

for k in range(timegrid.size - 1):

    integrator.setInput(uN[k,:], "p")
    integrator.evaluate()

    ysim[k+1, :] = pl.squeeze(integrator.getOutput("xf"))

    integrator.setInput(ysim[k+1, :], "x0")

pl.figure()
pl.subplot(4, 1, 1)
pl.plot(ysim[:, 0])
pl.plot(yN[:,0])

pl.subplot(4, 1, 2)
pl.plot(ysim[:, 1])
pl.plot(yN[:,1])

pl.subplot(4, 1, 3)
pl.plot(ysim[:, 2])
pl.plot(yN[:, 2])

pl.subplot(4, 1, 4)
pl.plot(ysim[:, 3])
pl.plot(yN[:, 3])

pl.show()
