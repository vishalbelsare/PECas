#!/usr/bin/python

import pylab as pl
from scipy.constants import C2K
from scipy.integrate import quad

import casadi as ca
import pecas as pe


# Model

c = 4.2 * 1e3
rho = 1e3

h = 9.8 * 1e-2

delta1 = 1e-3
delta2 = ca.SX.sym("delta2", 1)
r_a = (6.3 * 1e-2) / 2.0
r_i = r_a - delta1

A_a = 2 * pl.pi * r_a * h
A_i = 2 * pl.pi * r_i * h

A_m = (A_a - A_i) / pl.log(A_a / A_i)

V = pl.pi * r_i**2 * h
m = rho * V

T_amb = C2K(22)

T2 = ca.SX.sym("T2", 1)
T1 = ca.SX.sym("T1", 1)

t2 = ca.SX.sym("t2", 1)
t1 = ca.SX.sym("t1", 1)

lambda_g = ca.SX.sym("lambda_g", 1)
lambda_l = 0.0262

# T2 = (((lambda_g * A_m * (t2 - t1)) / delta) * T_amb + m * c * T1) / \
#     (m * c + ((lambda_g * A_m * (t2 - t1)) / delta))

# T2 = (((lambda_g * A_m * (t2 - t1)) / delta) * T_amb + \
#     (m * c - ((lambda_g * A_m * (t2 - t1)) / (2 * delta))) * T1) / \
#     (m * c + ((lambda_g * A_m * (t2 - t1)) / (2 * delta)))

T2 = (((A_m * (t2 - t1)) / (delta1 / lambda_g + delta2 / lambda_l)) * T_amb + \
    (m * c - ((A_m * (t2 - t1)) / ((2 * delta1) / lambda_g + (2 * delta2) / lambda_l))) * T1) / \
    (m * c + ((A_m * (t2 - t1)) / ((2 * delta1) / lambda_g + (2 * delta2) / lambda_l)))


fT2 = ca.SXFunction([T1, t2, t1, lambda_g, delta2], [T2])
fT2.setOption("name", "fT2")
fT2.init()


# Experimental setup

# Load measurements data when available, and
# replace values of t by real measurement time when available

data = pl.loadtxt("meas_stirred_dunked.txt")
t = data[:,0]
t_end = t[-1:]
N = len(t)

Y = C2K(data[:,1])
T_start = Y[0]


# t_end = 600
# N = 1000
# t = pl.linspace(0,t_end,N)

lambda_g_lit = 0.76
sigma = 1 * pl.ones(N)

# Simulation

T_sim = ca.SX.sym("T", N)
T_sim[0] = T_start

for k in range(N-1):

    T_sim[k+1] = fT2([T_sim[k], t[k+1], t[k], lambda_g, delta2])[0]


xinit = pl.array([0.2, 0.0002])
# xmin = pl.array([0.01, 0.0002])
# xmax = pl.array([2, 0.0012])

Theta = ca.vertcat([lambda_g, delta2])

# Setup PECas and generate pseudo measurement data;
# insert data here when measurements are available
#
pep = pe.PECasLSq(Theta, T_sim, sigma, Y = Y, xinit = xinit)#, xmin = xmin, xmax = xmax)
#

# pep = pe.PECasLSq(lambda_g, T_sim, sigma, xtrue = pl.array(lambda_g_lit))
# pep.generate_pseudo_measurement_data()

pep.run_parameter_estimation()
pep.compute_covariance_matrix()

pep.print_results()


# Plot measurements and curve resulting from parameter estimation

fT_sim = ca.SXFunction(ca.nlpIn(x = Theta), ca.nlpOut(f = T_sim))
fT_sim.setOption("name", "fT_sim")
fT_sim.init()

fT_sim.setInput(pep.get_xhat(), "x")
fT_sim.evaluate()

f_sim = fT_sim.getOutput("f")

pl.plot(t, f_sim, color = "g")
pl.scatter(t, pep.get_Y(), s = 2)
pl.xlim((t[0], t[-1:]))
pl.show()