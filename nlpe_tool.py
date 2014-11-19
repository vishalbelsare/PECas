#!/usr/bin/python

# NLPE-Tool

import casadi as ca
import numpy as np

# Problem setup

N = 4
d = 2
m = 1

sigma = 0.5 * np.ones(N)
x = ca.MX.sym("x", d)

xstar_fixed = [1, 1]

M = ca.mul(np.matrix([np.ones(4), range(1,5)]).T,x)
G = 2 - ca.mul(x.T,x)

# Generate Sigma

Sigma = np.diag(sigma)**2

# Generate pseudo-measurement data

Mx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=M))
Mx.init()

Mx.setInput(xstar_fixed, "x")
Mx.evaluate()

Y_N = Mx.getOutput("f") + np.random.normal(0, sigma, N)
# Y_N = Mx.getOutput("f") + np.random.normal(0, 1, N)

# Set up cost function f

A = ca.mul(np.linalg.solve(np.sqrt(Sigma), np.eye(N)), (M - Y_N))
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

beta = fxstar / (N + m -d)

J1 = ca.mul(np.linalg.solve(np.sqrt(Sigma), np.eye(N)), Mx.jac("x", "f"))

Gx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=G))
Gx.init()

J2 = Gx.jac("x", "f")

Jplus = ca.mul([ \

    ca.horzcat((np.eye(d),np.zeros((d, m)))), \

    ca.solve(ca.vertcat(( \
    
        ca.horzcat((ca.mul(J1.T, J1), J2.T)), \
        ca.horzcat((J2, np.zeros((m, m)))) \
    
    )), np.eye(d+m)), \

    ca.vertcat((J1.T, np.zeros((m, N)))) \

    ])

Cov = beta * ca.mul([Jplus, Jplus.T])

Covx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=Cov))
Covx.init()

Covx.setInput(xstar, "x")
Covx.evaluate()

Covxstar = Covx.getOutput("f")

# Print the Output

print("\nResults:\n")

print("beta = {0}\n".format(beta))

for i, xi in enumerate(xstar):

    print("x{0:<3} = {1:10} +/- {2:10}".format(\
        i, xi, np.sqrt(Covxstar[i, i])))

print("\nCovariance matrix Cov(x):\n{0}".format(Covxstar))
