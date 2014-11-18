#!/usr/bin/python

# NLPE-Tool

import casadi as ca
import numpy as np
import scipy as sp

# Problem setup

N = 4
d = 2
m = 1

sigma = [0.5, 0.5, 0.5, 0.5]
x = ca.MX.sym("x", d)

xstar_fixed = [1, 1]

M = ca.mul(np.transpose(np.matrix([np.ones(4), range(1,5)])),x)
G = 2 - ca.mul(x.trans(),x)

print M

# Generate Sigma

Sigma = np.diag(sigma)

# Generate pseudo-measurement data

Mx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=M))
Mx.init()

Mx.setInput(xstar_fixed, "x")
Mx.evaluate()
Y = Mx.getOutput("f")

Y_N = Y + np.random.normal(0, sigma, 4)
# Y_N = Y

# Set up cost function f

A = ca.mul(sp.linalg.inv(np.sqrt(Sigma)), M) - Y_N
# A = M - Y_N
f = ca.mul(A.T, A)

# Solve minimization problem for f

fx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=f, g=G))
fx.init()

solver = ca.NlpSolver("ipopt", fx)
solver.setOption("tol", 1e-10)
solver.init()

solver.setInput(np.zeros(G.size1()), "lbg")
solver.setInput(np.zeros(G.size1()), "ubg")

solver.evaluate()

xstar = solver.getOutput("x")
fxstar = solver.getOutput("f")

print xstar

# Mx.setInput(xstar, "x")
# Mx.evaluate()
# print Mx.getOutput("f")

beta = fxstar/(N-d)

print beta

J1 = ca.mul(sp.linalg.inv(np.sqrt(Sigma)), Mx.jac("x", "f"))

Gx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=G))
Gx.init()

J2 = Gx.jac("x", "f")

# Jplus1 = ca.vertcat((ca.mul(J1.T, J1), J2.T))
# Jplus2 = ca.vertcat((J2, np.zeros((J2.size1(), J2.size1()))))
# Jplus = ca.inv(ca.horzcat((Jplus1.T, Jplus2.T)).T)

