#!/usr/bin/python

import pylab as pl
import casadi as ca
import casadi.tools as cat

import pecas

# ex2.py: Usage of multiple states.

# (model and data taken from Bock, Sager et al.: Uebungen Numerische
# Mathematik II, Blatt 9, IWR, Universitaet Heidelberg, 2006)

T = pl.linspace(0, 10, 11)

yN = pl.array([[1.0, 0.9978287, 2.366363, 6.448709, 5.225859, 2.617129, \
           1.324945, 1.071534, 1.058930, 3.189685, 6.790586], \

           [1.0, 2.249977, 3.215969, 1.787353, 1.050747, 0.2150848, \
           0.109813, 1.276422, 2.493237, 3.079619, 1.665567]])

sigma_x1 = 0.1
sigma_x2 = 0.2

x = ca.MX.sym("x", 2)

alpha = 1
gamma = 1

p = ca.MX.sym("p", 2)

u = ca.MX.sym("u", 0)

f = ca.vertcat( \
    [-alpha * x[0] + p[0] * x[0] * x[1], 
    gamma * x[1] - p[1] * x[0] * x[1]])

y = x

odesys = pecas.systems.ExplODE(x = x, u = u, p = p, f = f, y = y)
odesys.show_system_information(showEquations = True)

sigma_Y = pl.zeros((2, yN.shape[1]))
sigma_Y[0,:] = (1.0 / sigma_x1**2)
sigma_Y[1,:] = (1.0 / sigma_x2**2)

lsqpe = pecas.LSq(system = odesys, \
    tu = T, \
    pinit = [0.5, 1.0], \
    xinit = yN, \
    yN = yN, \
    wv = sigma_Y, \
    # linear_solver = "mumps", \
    linear_solver = "ma97")

lsqpe.run_parameter_estimation()
lsqpe.compute_covariance_matrix()

lsqpe.show_results()

pl.scatter(T, yN[0,:], color = 'b')
pl.scatter(T, yN[1,:], color = 'r')

t = pl.linspace(0,10,1000)
lsqpe.run_simulation(x0 = yN[:,0], tsim = t)

pl.plot(t, pl.squeeze(lsqpe.Xsim[0,:]), color='b')
pl.plot(t, pl.squeeze(lsqpe.Xsim[1,:]), color='r')
pl.grid()
pl.show()
