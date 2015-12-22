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
# but WITHOUT ANY WARRANTY; without even the implied warrantime_points of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

# Model and data taken from: Verschueren, Robin: Design and implementation of a 
# time-optimal controller for model race cars, Master’s thesis, KU Leuven, 2014.

import casadi as ca
import pylab as pl
import pecas.system
import pecas.pe
import pecas.sim

# System

x = ca.MX.sym("x", 4)
p = ca.MX.sym("p", 6)
u = ca.MX.sym("u", 2)

f = ca.vertcat( \

    [x[3] * pl.cos(x[2] + p[0] * u[0]),

    x[3] * pl.sin(x[2] + p[0] * u[0]),

    x[3] * u[0] * p[1],

    p[2] * u[1] \
        - p[3] * u[1] * x[3] \
        - p[4] * x[3]**2 \
        - p[5] \
        - (x[3] * u[0])**2 * p[1]* p[0]])

phi = x

system = pecas.system.System(x = x, u = u, p = p, f = f, phi = phi)

data = pl.array(pl.loadtxt("data_2d_vehicle.dat", \
    delimiter = ", ", skiprows = 1))

data_opt = pl.loadtxt("results_doe_2d_vehicle.txt")

time_points = data[100:150, 1]

ydata = data[100:750, [2, 4, 6, 8]]

udata_opt = data_opt[:98].reshape(-1, 2)

pdata = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

# pe = pecas.pe.LSq(system = system, \
#     time_points = time_points, udata = udata, \
#     pinit = pinit, \
#     ydata = ydata, \
#     xinit = ydata)

# pe.run_parameter_estimation({"linear_solver": "ma97"})
# pe.print_estimation_results()

# pe.compute_covariance_matrix()
# pe.print_estimation_results()

sim = pecas.sim.Simulation(system, pdata)
sim.run_system_simulation(time_points = time_points, \
    x0 = ydata[0,:], udata = udata_opt)

xhat = sim.simulation_results[0,:].T
yhat = sim.simulation_results[1,:].T
psihat = sim.simulation_results[2,:].T
vhat = sim.simulation_results[3,:].T

pl.close("all")

pl.figure()

pl.subplot2grid((4, 2), (0, 0))
pl.plot(time_points, xhat, label = "$X_{sim}$")
# pl.plot(time_points, ydata[:,0], label = "$X_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$X$", rotation = 0)
pl.legend(loc = "upper right")

pl.subplot2grid((4, 2), (1, 0))
pl.plot(time_points, yhat, label = "$Y_{sim}$")
# pl.plot(time_points, ydata[:,1], label = "$Y_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$Y$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (2, 0))
pl.plot(time_points, psihat, label = "$\psi_{sim}$")
# pl.plot(time_points, ydata[:, 2], label = "$\psi_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$\psi$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (3, 0))
pl.plot(time_points, vhat, label = "$v_{sim}$")
# pl.plot(time_points, ydata[:, 3], label = "$v_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$v$", rotation = 0)
pl.legend(loc = "upper left")

pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
pl.plot(xhat, yhat, label = "$(X_{sim},\,Y_{sim})$")
# pl.plot(ydata[:,0], ydata[:, 1], label = "$(X_{meas},\,Y_{meas})$")
pl.xlabel("$X$")
pl.ylabel("$Y$", rotation = 0)
pl.legend(loc = "lower right")

pl.show()

# simulation_results_noisy = simulation_results + 0.01 * pl.randn(*simulation_results.shape)

