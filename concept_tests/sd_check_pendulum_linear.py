import casadi as ca
import pylab as pl
import pecas

import os

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
# psi = pl.pi / 2.0
psi = pl.pi / (180.0 * 2)

# System

x = ca.MX.sym("x", 2)
p = ca.MX.sym("p", 1)
u = ca.MX.sym("u", 1)

# f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * pl.sin(x[0])])
f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * x[0]])

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

ptrue = [3.0]

lsqpe_sim.run_simulation(x0 = yN[:,0], psim = ptrue)

p_test = []

sigma = 0.1
wv = (1. / sigma**2) * pl.ones(yN.shape)

repetitions = 100

for k in range(repetitions):

    y_randn = lsqpe_sim.Xsim + sigma * (pl.randn(*lsqpe_sim.Xsim.shape))

    lsqpe_test = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = 1, \
    xinit = y_randn, 
    # linear_solver = "ma97", \
    yN = y_randn, wv = wv)

    lsqpe_test.run_parameter_estimation()

    p_test.append(lsqpe_test.phat)


p_mean = pl.mean(p_test)
p_std = pl.std(p_test, ddof=0)

lsqpe_test.compute_covariance_matrix()


# Generate report

print("\np_mean         = " + str(ca.DMatrix(p_mean)))
print("phat_last_exp  = " + str(ca.DMatrix(lsqpe_test.phat)))

print("\np_sd           = " + str(ca.DMatrix(p_std)))
print("sd_from_covmat = " + str(ca.diag(ca.sqrt(lsqpe_test.Covp))))
print("beta           = " + str(lsqpe_test.beta))

print("\ndelta_abs_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(lsqpe_test.Covp)))))
print("delta_rel_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(lsqpe_test.Covp))) / ca.DMatrix(p_std)))


fname = os.path.basename(__file__)[:-3] + ".rst"

report = open(fname, "w")
report.write( \
'''Concept test: covariance matrix computation
===========================================

Simulate system. Then: add gaussian noise N~(0, sigma^2), estimate,
store estimated parameter, repeat.

.. code-block:: python

    y_randn = lsqpe_sim.Xsim + sigma * \
(np.random.randn(*lsqpe_sim.Xsim.shape))

Afterwards, compute standard deviation of estimated parameters, 
and compare to single covariance matrix computation done in PECas.

''')

prob = "ODE, 2 states, 1 control, 1 param, (pendulum linear)"
report.write(prob)
report.write("\n" + "-" * len(prob) + "\n\n.. code-block:: python")

report.write( \
'''.. code-block:: python

    ------------------------ PECas system information ------------------------

    The system is a dynamic system defined by a set of
    explicit ODEs xdot which establish the system state x:
        xdot = f(t, u, x, p, we, wu)
    and by an output function phi which sets the system measurements:
        y = phi(t, x, p).

    Particularly, the system has:
        1 inputs u
        1 parameters p
        2 states x
        2 outputs phi

    Where xdot is defined by: 
    xdot[0] = x[1]
    xdot[1] = (((p/9)*(u-x[0]))-(3.27*x[0]))

    And where phi is defined by: 
    y[0] = x[0]
    y[1] = x[1]
''')

report.write("\n**Test results:**\n\n.. code-block:: python")

report.write("\n\n    repetitions    = " + str(repetitions))
report.write("\n    sigma          = " + str(sigma))

report.write("\n\n    p_orig         = " + str(ca.DMatrix(ptrue)))
report.write("\n\n    p_mean         = " + str(ca.DMatrix(p_mean)))
report.write("\n    phat_last_exp  = " + str(ca.DMatrix(lsqpe_test.phat)))

report.write("\n\n    p_sd           = " + str(ca.DMatrix(p_std)))
report.write("\n    sd_from_covmat = " + str(ca.diag(ca.sqrt(lsqpe_test.Covp))))
report.write("\n    beta           = " + str(lsqpe_test.beta))

report.write("\n\n    delta_abs_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(lsqpe_test.Covp)))))
report.write("\n    delta_rel_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(lsqpe_test.Covp))) / ca.DMatrix(p_std)) + "\n")

report.close()

os.system("rst2pdf " + fname)
