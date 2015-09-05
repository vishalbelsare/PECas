import casadi as ca
import pylab as pl
import pecas

# (Model and data taken from: Diehl, Moritz: Course on System Identification, 
# exercise 7, SYSCOP, IMTEK, University of Freiburg, 2014/2015)

# Defining constant problem parameters: 
#
#     - m: representing the ball of the mass in kg
#     - L: the length of the pendulum bar in meters
#     - g: the gravity constant in m/s^2
#     - psi: the actuation angle of the manuver in radians, which stays
#            constant for this problem

m = 1.0
L = 3.0
g = 9.81
psi = pl.pi / 2.0

# System

x = ca.MX.sym("x", 2)
p = ca.MX.sym("p", 1)
u = ca.MX.sym("u", 1)

f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * pl.sin(x[0])])

phi = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, phi = phi)
odesys.show_system_information(showEquations = True)

# Loading data

data = pl.loadtxt('data_pendulum.txt')
tu = data[:500, 0]
numeas = data[:500, 1]
wmeas = data[:500, 2]
N = tu.size
yN = pl.array([numeas,wmeas])
uN = [psi] * (N-1)

# Definition of the weightings for each of the measurements.

# Since no data is provided beforehand, it is assumed that both, angular
# speed and rotational angle have i.i.d. noise, thus the measurement have
# identic standard deviations that can be calculated.

sigmanu = pl.std(numeas, ddof=1)
sigmaw = pl.std(wmeas, ddof=1)

# The weightings for the measurements errors given to PECas are calculated
# from the standard deviations of the measurements, so that the least squares
# estimator ist the maximum likelihood estimator for the estimation problem.

wnu = 1.0 / (pl.ones(tu.size)*sigmanu**2)
ww = 1.0 / (pl.ones(tu.size)*sigmaw**2)

wv = pl.array([wnu, ww])

# Run parameter estimation and assure that the results is correct

lsqpe = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = 1, \
    xinit = yN, 
    linear_solver = "ma97", \
    yN = yN, wv = wv)

lsqpe.run_parameter_estimation()
lsqpe.show_results()

lsqpe.compute_covariance_matrix()
lsqpe.show_results()

lsqpe.run_simulation([numeas[0], wmeas[0]])
nusim = lsqpe.Xsim[0,:].T
wsim = lsqpe.Xsim[1,:].T

pl.close("all")

pl.figure()
pl.subplot2grid((2, 2), (0, 0))
pl.scatter(tu[::2], numeas[::2], \
    s = 10.0, color = 'k', marker = "x", label = r"$\nu_{meas}$")
pl.plot(tu, nusim, label = r"$\nu_{sim}$")

pl.xlabel("$t$")
pl.ylabel(r"$\nu$", rotation = 0)
pl.xlim(0.0, 4.2)

pl.legend(loc = "lower left")

pl.subplot2grid((2, 2), (1, 0))
pl.scatter(tu[::2], wmeas[::2], \
    s = 10.0, color = 'k', marker = "x", label = "$\omega_{meas}$")
pl.plot(tu, wsim, label = "$\omega_{sim}$")

pl.xlabel("$t$")
pl.ylabel("$\omega$", rotation = 0)
pl.xlim(0.0, 4.2)

pl.legend(loc = "lower right")

pl.subplot2grid((2, 2), (0, 1), rowspan = 2)
pl.scatter(numeas[::2], wmeas[::2], \
    s = 10.0, color = 'k', marker = "x", \
    label = r"$(\nu_{meas},\,\omega_{meas})$")
pl.plot(nusim, wsim, label = r"$(\nu_{sim},\,\omega_{sim})$")

pl.xlabel(r"$\nu$")
pl.ylabel("$\omega$", rotation = 0)
pl.xlim(-2.5, 3.0)
pl.ylim(-5.0, 5.0)

pl.legend()

pl.show()
