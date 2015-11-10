#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

# This example is an adapted version of the system identification example
# included in CasADi, for the original file see:
# https://github.com/casadi/casadi/blob/master/docs/examples/python/sysid.py

import casadi as ca
import pylab as pl
import pecas

import os

N = 1000
fs = 610.1

p_true = ca.DMatrix([5.625e-6,2.3e-4,1,4.69])
p_guess = ca.DMatrix([5,3,1,5])
scale = ca.vertcat([1e-6,1e-4,1,1])

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)
p = ca.MX.sym("p", 4)

f = ca.vertcat([
        x[1], \
        (u - scale[3] * p[3] * x[0]**3 - scale[2] * p[2] * x[0] - \
            scale[1] * p[1] * x[1]) / (scale[0] * p[0]), \
    ])

phi = x

odesys = pecas.systems.ExplODE( \
    x = x, u = u, p = p, f = f, phi = phi)

dt = 1.0 / fs
tu = pl.linspace(0, N, N+1) * dt

uN = ca.DMatrix(0.1*pl.random(N))

yN = pl.zeros((x.shape[0], N+1))
wv = pl.ones(yN.shape)

lsqpe_sim = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = p_guess, \
    xinit = yN, 
    # linear_solver = "ma97", \
    yN = yN, wv = wv)

lsqpe_sim.run_simulation(x0 = [0.0, 0.0], psim = p_true/scale)

p_test = []

sigma = 0.01
wv = (1. / sigma**2) * pl.ones(yN.shape)

repetitions = 100

for k in range(repetitions):

    y_randn = lsqpe_sim.Xsim + sigma * (pl.randn(*lsqpe_sim.Xsim.shape))

    lsqpe_test = pecas.LSq( \
    system = odesys, tu = tu, \
    uN = uN, \
    pinit = p_guess, \
    xinit = y_randn, 
    linear_solver = "ma97", \
    yN = y_randn, wv = wv)

    lsqpe_test.run_parameter_estimation()

    p_test.append(lsqpe_test.phat)


p_mean = []
p_std = []

for j, e in enumerate(p_true):

    p_mean.append(pl.mean([k[j] for k in p_test]))
    p_std.append(pl.std([k[j] for k in p_test], ddof = 0))

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

prob = "ODE, 2 states, 1 control, 1 param, (silverbox)"
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
        4 parameters p
        2 states x
        2 outputs phi

    Where xdot is defined by: 
    xdot[0] = x[1]
    xdot[1] = ((((u-(p[3]*pow(x[0],3)))-(p[2]*x[0]))- 
        ((0.0001*p[1])*x[1]))/(1e-06*p[0]))

    And where phi is defined by: 
    y[0] = x[0]
    y[1] = x[1]
''')

report.write("\n**Test results:**\n\n.. code-block:: python")

report.write("\n\n    repetitions    = " + str(repetitions))
report.write("\n    sigma          = " + str(sigma))

report.write("\n\n    p_orig         = " + str(ca.DMatrix(p_true/scale)))
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
