import casadi as ca
import pylab as pl
import pecas

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

x = ca.MX.sym("x", 2)
p = ca.MX.sym("p", 1)
u = ca.MX.sym("u", 1)

f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * pl.sin(x[0])])

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, y = y)

#==============================================================================
# Loading data
#==============================================================================
data = pl.loadtxt('ex6data.txt')
tu = data[:500, 0]
phim = data[:500, 1]
wm = data[:500, 2]
N = tu.size
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

sigmaphi = 1.0 / (pl.ones(tu.size)*pl.std(phim, ddof=1)**2)
sigmaw = 1.0 / (pl.ones(tu.size)*pl.std(wm, ddof=1)**2)

wv = pl.array([sigmaphi, sigmaw])

# odesetup = pecas.setups.ODEsetup( \
#     system = odesys, tu = tu,
#     u = uN, \
#     pinit = 1, pmax = 50, pmin = 0 )

# Run parameter estimation and assure that the results is correct

lsqpe = pecas.LSq( \
    system = odesys, tu = tu, \
    u = uN, \
    pinit = 1, \
    xinit = yN, 
    yN = yN, wv = wv)

lsqpe.show_system_information(showEquations = True)

lsqpe.run_parameter_estimation()
lsqpe.show_results()

lsqpe.compute_covariance_matrix()
# lsqpe.covmat_schur()
lsqpe.show_results()

phihat = lsqpe.Xhat[0]
what = lsqpe.Xhat[1]

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

pl.show()
