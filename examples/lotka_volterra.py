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

# Model and data taken from: Bock, Sager et al.: Uebungen zur Numerischen
# Mathematik II, sheet 9, IWR, Heidelberg university, 2006

import pylab as pl
import casadi as ca

import pecas

import pecas.system
import pecas.pe
import pecas.sim

T = pl.linspace(0, 10, 11)

yN = pl.array([[1.0, 0.9978287, 2.366363, 6.448709, 5.225859, 2.617129, \
           1.324945, 1.071534, 1.058930, 3.189685, 6.790586], \

           [1.0, 2.249977, 3.215969, 1.787353, 1.050747, 0.2150848, \
           0.109813, 1.276422, 2.493237, 3.079619, 1.665567]])

sigma_x1 = 0.1
sigma_x2 = 0.2

x = ca.MX.sym("x", 2)

alpha = 1.0
gamma = 1.0

p = ca.MX.sym("p", 2)

f = ca.vertcat( \
    [-alpha * x[0] + p[0] * x[0] * x[1], 
    gamma * x[1] - p[1] * x[0] * x[1]])

phi = x

system = pecas.system.System(x = x, p = p, f = f, phi = phi)

# The weightings for the measurements errors given to PECas are calculated
# from the standard deviations of the measurements, so that the least squares
# estimator ist the maximum likelihood estimator for the estimation problem.

wv = pl.zeros((2, yN.shape[1]))
wv[0,:] = (1.0 / sigma_x1**2)
wv[1,:] = (1.0 / sigma_x2**2)

pe = pecas.pe.LSq(system = system, time_points = T, xinit = yN, ydata = yN, wv = wv)

pe.run_parameter_estimation(solver_options = {"linear_solver": "ma97"})
pe.print_estimation_results()

pe.compute_covariance_matrix()

# T_sim = pl.linspace(0, 10, 101)
# x0 = yN[:,0]

# sim = pecas.sim.Simulation(system, pe.estimated_parameters)
# sim.run_system_simulation(time_points = T_sim, x0 = x0)

# pl.figure()

# pl.scatter(T, yN[0,:], color = "b", label = "$x_{1,meas}$")
# pl.scatter(T, yN[1,:], color = "r", label = "$x_{2,meas}$")

# pl.plot(T_sim, pl.squeeze(sim.simulation_results[0,:]), color="b", label = "$x_{1,sim}$")
# pl.plot(T_sim, pl.squeeze(sim.simulation_results[1,:]), color="r", label = "$x_{2,sim}$")

# pl.xlabel("$t$")
# pl.ylabel("$x_1, x_2$", rotation = 0)
# pl.xlim(0.0, 10.0)

# pl.legend(loc = "upper left")
# pl.grid()

# pl.show()
