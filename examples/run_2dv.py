import casadi as ca
import pylab as pl
import pecas

import time

tstart = time.time()

# System

x = ca.SX.sym("x", 4)
p = ca.SX.sym("p", 6)
u = ca.SX.sym("u", 2)
w = ca.SX.sym("w", 4)

f = ca.vertcat( \

    [x[3] * pl.cos(x[2] + p[0] * u[0]) + w[0],

    x[3] * pl.sin(x[2] + p[0] * u[0]) + w[1],

    x[3] * u[0] * p[1] + w[2],

    p[2] * u[1] \
        - p[3] * u[1] * x[3] \
        - p[4] * x[3]**2 \
        - p[5] \
        - (x[3] * u[0])**2 * p[1] * p[0] + w[3]])

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, w = w, f = f, y = y)

# Inputs

data = pl.array(pl.loadtxt( \
    "controlReadings_ACADO_MPC_Betterweights.dat", \
    delimiter = ", ", skiprows = 1))

timegrid = data[200:250, 1]


yN = data[200:250, [2, 4, 6, 8]]
wv = 1 / (0.1**2) * pl.ones(yN.shape)
uN = data[200:249, [9, 10]]
ww = [1 / 1e-4] * 4

porig = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]
# phat = [12.0, 0.1, 0.6]

odesetup = pecas.setups.ODEsetup( \
    system = odesys, timegrid = timegrid,
    umin = uN, umax = uN, uinit = uN, \
    pmin = [0.5, 17.06, 0.0, -10.0, -1000.0, -10.0], \
    pmax = [0.5, 17.06, 13.2, 200, 500, 3], \
    pinit = [0.5, 17.06, 11.5, 5, 0.07, 0.70])

# Run parameter estimation and assure that the results is correct

lsqpe = pecas.LSq(pesetup =odesetup, yN =yN, wv = wv, ww = ww)

lsqpe.run_parameter_estimation()
phat = lsqpe.Phat

print porig
print phat

Xhat = lsqpe.Xhat

xhat = lsqpe.Xhat[0]
yhat = lsqpe.Xhat[1]
psihat = lsqpe.Xhat[2]
vhat = lsqpe.Xhat[3]

pl.close("all")

pl.figure()
pl.subplot(4, 1, 1)
pl.plot(xhat)
pl.plot(yN[:,0])

pl.subplot(4, 1, 2)
pl.plot(yhat)
pl.plot(yN[:,1])

pl.subplot(4, 1, 3)
pl.plot(psihat)
pl.plot(yN[:, 2])

pl.subplot(4, 1, 4)
pl.plot(vhat)
pl.plot(yN[:, 3])

pl.figure()
pl.plot(xhat, yhat)
pl.plot(yN[:,0], yN[:, 1])

# pl.figure()
# pl.plot(sum(odesetup.V()(lsqpe.Vhat)["X",:,:,0], []))

pl.show()

tend = time.time()
dur = tend - tstart
print "started: " + time.ctime(tstart)
print "ended: " + time.ctime(tend)
print "duration: " + str(dur) + "sec"