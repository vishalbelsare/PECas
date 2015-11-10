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

# Model and data taken from: Verschueren, Robin: Design and implementation of a 
# time-optimal controller for model race cars, Master’s thesis, KU Leuven, 2014.

import casadi as ca
import pylab as pl
import pecas

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

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, phi = phi)
odesys.show_system_information(showEquations = True)

# Inputs

data = pl.array(pl.loadtxt("data_2d_vehicle.dat", \
    delimiter = ", ", skiprows = 1))

ty = data[100:250, 1]

yN = data[100:250, [2, 4, 6, 8]]
wv = pl.ones(yN.shape)

uN = data[100:249, [9, 10]]

pinit = [0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

lsqpe = pecas.LSq(system = odesys, \
    tu = ty, uN = uN, \
    pinit = pinit, \
    ty = ty, yN =yN, \
    wv = wv, \
    xinit = yN, \
    linear_solver = "ma97", \
    scheme = "radau", \
    order = 3)

lsqpe.run_parameter_estimation(hessian = "exact-hessian")

lsqpe.run_simulation(x0 = yN[0,:])

xhat = lsqpe.Xsim[0,:].T
yhat = lsqpe.Xsim[1,:].T
psihat = lsqpe.Xsim[2,:].T
vhat = lsqpe.Xsim[3,:].T

pl.close("all")

pl.figure()

pl.subplot2grid((4, 2), (0, 0))
pl.plot(ty, xhat, label = "$X_{sim}$")
pl.plot(ty, yN[:,0], label = "$X_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$X$", rotation = 0)
pl.legend(loc = "upper right")

pl.subplot2grid((4, 2), (1, 0))
pl.plot(ty, yhat, label = "$Y_{sim}$")
pl.plot(ty, yN[:,1], label = "$Y_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$Y$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (2, 0))
pl.plot(ty, psihat, label = "$\psi_{sim}$")
pl.plot(ty, yN[:, 2], label = "$\psi_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$\psi$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (3, 0))
pl.plot(ty, vhat, label = "$v_{sim}$")
pl.plot(ty, yN[:, 3], label = "$v_{meas}$")
pl.xlabel("$t$")
pl.ylabel("$v$", rotation = 0)
pl.legend(loc = "upper left")

pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
pl.plot(xhat, yhat, label = "$(X_{sim},\,Y_{sim})$")
pl.plot(yN[:,0], yN[:, 1], label = "$(X_{meas},\,Y_{meas})$")
pl.xlabel("$X$")
pl.ylabel("$Y$", rotation = 0)
pl.legend(loc = "lower right")

pl.show()
