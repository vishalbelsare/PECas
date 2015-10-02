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

data = pl.loadtxt('data_pendulum.txt')
tu = data[:500, 0]
numeas = data[:500, 1]
wmeas = data[:500, 2]
yN = pl.array([numeas,wmeas])
N = tu.size
uN = [psi] * (N-1)

wv = pl.ones(yN.shape)

lsqpe_sim = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = 1, \
    xinit = yN, 
    # linear_solver = "ma97", \
    yN = yN, wv = wv)


lsqpe_sim.run_simulation(x0 = yN[:,0], psim = [3.0])

p_test = []

for k in range(100):

    y_test = lsqpe_sim.Xsim + 0.1 * (pl.rand(*lsqpe_sim.Xsim.shape) - 0.5)

    lsqpe_test = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = 1, \
    xinit = y_test, 
    # linear_solver = "ma97", \
    yN = y_test, wv = wv)

    lsqpe_test.run_parameter_estimation()

    p_test.append(lsqpe_test.phat)


p_std = pl.std(p_test, ddof=1)
p_mean = pl.mean(p_test)
print p_mean
print p_std

lsqpe_test.compute_covariance_matrix()
print ca.sqrt(lsqpe_test.Covp)
