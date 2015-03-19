import casadi as ca
import pylab as pl
import pecas

import unittest
import test_lsq_run

import test_scenarios

# System

x = ca.SX.sym("x", 4)
p = ca.SX.sym("p", 6)
u = ca.SX.sym("u", 2)

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

    # p[0] * u[1] \
    #     - 2.17 * u[1] * x[3] \
    #     - p[1] * x[3]**2 - p[2] \
    #     - (x[3]  * u[0])**2 \
    #     * p[3] * 0.5])

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, \
    p = p, f = f, y = y)

# Inputs

data = pl.array(pl.loadtxt( \
    "controlReadings_ACADO_MPC_Betterweights.dat", \
    delimiter = ", ", skiprows = 1))

timegrid = data[200:300, 0]


yN = data[200:300, [2, 4, 6, 8]]
stdyN = 1e-2 * pl.ones(yN.shape)
stdyN[:, 2] = 1e-1
stdyN[:, 3] = 1
uN = data[200:299, [9, 10]]
stds = 1e-2

porig = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]
# phat = [12.0, 0.1, 0.6]

odesetup = pecas.setups.ODEsetup( \
    system = odesys, timegrid = timegrid,
    umin = uN, umax = uN, uinit = uN, \
    # x0min = yN[0,:], x0max = yN[0,:], \
    # pmin = [12.0, 0, -pl.inf, 0], \
    # pmax = [12.0, 10, pl.inf, pl.inf], \
    # pinit = [12.0, 0.1, 0.6, 17.06])
    # xmin = yN - 0.1 * abs(yN), xmax = yN + 0.1 * abs(yN), xinit = yN,
    pmin = [0.5, 17.06, 0.0, 0, 0.0, 0.0], \
    pmax = [0.5, 17.06, 13.2, 100, 400, 3], \
    pinit = [0.5, 17.06, 11.5, 5, 0.07, 0.7])
    # pmin = [0.1] * 6, \
    # pmax = [2] * 6, \
    # pinit = [1] * 6)

# Run parameter estimation and assure that the results is correct

lsqpe = pecas.LSq(pesetup =odesetup, yN =yN, \
            stdyN = stdyN, stds =stds)

lsqpe.run_parameter_estimation()
phat = odesetup.V()(lsqpe.Vhat)["P"]
print porig
print phat

xhat = odesetup.V()(lsqpe.Vhat)["X",:,0,0]
yhat = odesetup.V()(lsqpe.Vhat)["X",:,0,1]
psihat = odesetup.V()(lsqpe.Vhat)["X",:,0,2]
vhat = odesetup.V()(lsqpe.Vhat)["X",:,0,3]

pl.close("all")

pl.figure()
pl.plot(xhat)
pl.plot(yN[:,0])

pl.figure()
pl.plot(yhat)
pl.plot(yN[:,1])

pl.figure()
pl.plot(psihat)
pl.plot(yN[:, 2])

pl.figure()
pl.plot(vhat)
pl.plot(yN[:, 3])

pl.show()