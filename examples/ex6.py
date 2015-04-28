import casadi as ca
import pylab as pl
import pecas

import time

tstart = time.time()

#==============================================================================
# Defining constant problem parameters: 
#     - m: representing the ball of the mass in Kg
#     - L: the length of the pendulum bar in meters
#     - g: the gravity constant in m/s^2
#     - psi: the actuation angle of the manuver in radians, which stays
#     constant for this problem
#==============================================================================
m = 1
L = 3
g = 9.81
psi = pl.pi/2


# System

x = ca.SX.sym("x", 2)
p = ca.SX.sym("p", 1)
u = ca.SX.sym("u", 1)

f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * pl.sin(x[0])])

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, y = y)

#==============================================================================
# Loading data
#==============================================================================
data = pl.loadtxt('ex6data.txt')
timegrid = data[:50, 0]
phim = data[:50, 1]
wm = data[:50, 2]
N = timegrid.size
yN = pl.array([phim,wm])
uN = [psi] * (N-1)

#==============================================================================
# Definition of the standar deviations each of the measurements. Since no data
# is provided before hand, it is assumed that both, angular speed and rotational
# angle have i.i.d. noise, thus the measurement have identic standar deviation
# and they can be calculated. The vector that is provided to PECas must have
# the standar deviation of each measurement, and thus it is of size 2*N.
# Finally, the optimization problem is initialized using some random vector
# [phi,w,K] = [1,1,1], and the vector with all the measurement is created.
#==============================================================================

sigmaphi = 1.0 / (pl.ones(timegrid.size)*pl.std(phim, ddof=1)**2)
sigmaw = 1.0 / (pl.ones(timegrid.size)*pl.std(wm, ddof=1)**2)

wv = pl.array([sigmaphi, sigmaw])


odesetup = pecas.setups.ODEsetup( \
    system = odesys, timegrid = timegrid,
    u = uN, \
    pinit = 1, pmax = 50, pmin = 0 )

# Run parameter estimation and assure that the results is correct

lsqpe = pecas.LSq(pesetup =odesetup, yN =yN, wv = wv)

lsqpe.run_parameter_estimation()
phat = lsqpe.phat

print "Khat: " + str(phat)

phihat = lsqpe.Xhat[0]
what = lsqpe.Xhat[1]

print "Phi0hat: " + str(phihat[0])
print "w0hat: " + str(what[0])

pl.close("all")

pl.figure()
pl.subplot(2, 1, 1)
pl.plot(phihat)
pl.plot(phim)

pl.subplot(2, 1, 2)
pl.plot(what)
pl.plot(wm)

pl.figure()
pl.plot(phihat, what)
pl.plot(phim, wm)

# pl.figure()
# pl.plot(sum(odesetup.V()(lsqpe.Vhat)["X",:,:,0], []))

pl.show()

tend = time.time()
dur = tend - tstart
print "started: " + time.ctime(tstart)
print "ended: " + time.ctime(tend)
print "duration: " + str(dur) + "sec"