#!/usr/bin/python

# ex2.py: Show the usage of equality constraints and an initial guess in PECas.

# Again, import the necessary modules, and define the optimization variables x.

import pecas
import numpy as np
import casadi as ca

d = 2
x = ca.SX.sym("x", d)

# Then, define the model M from the descriptions above, e. g. by using the
# CasADi commands for matrix multiplication casadi.mul() and vertical
# concatenation casadi.vertcat().

M = ca.mul(np.matrix([np.ones(4), range(1,5)]).T, \
        ca.vertcat((x[0], x[1]**2)))

# Define the column vector G of type casadi.casadi_core.SX for the
# equality constraints using the optimization variables x.

G = 2 - ca.mul(x.T,x)

# The initial guess xinit has to be defined as a column vector of
# type numpy.ndarray.

xinit = np.array([1, 1])

# Finally, again create the vectors for the measurements Y and the
# standard deviations sigma.

Y = np.array([2.23947, 2.84568, 4.55041, 5.08583])
sigma = 0.5 * np.ones(M.shape[0])


# Now, an instance pep of the class PECasProb can be created by also
# adressing the equality constraints and the inital guess, and perform
# the least squares estimation.

pep = pecas.PECasProb(x, M, sigma, Y = Y, G = G, xinit = xinit)
pep.run_parameter_estimation()

# After the solver converged, the covariance matrix can be computed,
# and the results of the parameter estimation can be displayed.

pep.compute_covariance_matrix()
pep.print_results()
