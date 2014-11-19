#!/usr/bin/python

# NLPE-Tool

import casadi as ca
import numpy as np

# Problem setup

N = 4
d = 2
m = 1

sigma = np.zeros(N)
sigma.fill(0.5)
x = ca.MX.sym("x", d)

xstar_fixed = [1, 1]

M = ca.mul(np.transpose(np.matrix([np.ones(4), range(1,5)])),x)
G = 2 - ca.mul(x.trans(),x)

# Generate Sigma

Sigma = np.diag(sigma)

# Generate pseudo-measurement data

Mx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=M))
Mx.init()

Mx.setInput(xstar_fixed, "x")
Mx.evaluate()

Y_N = Mx.getOutput("f") + np.random.normal(0, sigma, N)
# Y_N = Mx.getOutput("f")

# Set up cost function f

A = ca.mul(np.linalg.solve(np.sqrt(Sigma), np.eye(N)), M) - Y_N
# A = M - Y_N
f = ca.mul(A.T, A)

# Solve minimization problem for f

fx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=f, g=G))
fx.init()

solver = ca.NlpSolver("ipopt", fx)
solver.setOption("tol", 1e-10)
solver.init()

solver.setInput(np.zeros(m), "lbg")
solver.setInput(np.zeros(m), "ubg")

solver.evaluate()

xstar = solver.getOutput("x")
fxstar = solver.getOutput("f")

print xstar

# Mx.setInput(xstar, "x")
# Mx.evaluate()
# print Mx.getOutput("f")

beta = fxstar/(N + m -d)

print beta

J1 = ca.mul(np.linalg.solve(np.sqrt(Sigma), np.eye(N)), Mx.jac("x", "f"))

Gx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=G))
Gx.init()

J2 = Gx.jac("x", "f")

Jplus = ca.mul([ \

    ca.horzcat((np.ones((d,d)),np.zeros((d, m)))), \

    ca.solve(ca.vertcat(( \
    
        ca.horzcat((ca.mul(J1.T, J1), J2.T)), \
        ca.horzcat((J2, np.zeros((m, m)))) \
    
    )), np.eye(d+m)), \

    ca.vertcat((J1.T, np.zeros((m, N)))) \

    ])

Cov = ca.mul([Jplus, beta, np.eye(N), Jplus.T])

Covx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=Cov))
Covx.init()

Covx.setInput(xstar, "x")
Covx.evaluate()

print Covx.getOutput("f")