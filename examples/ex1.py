#!/usr/bin/python

# ex1.py: A simple example to show the principal usage of PECas

# Import all modules necessary for use of PECas

import pecas
import numpy as np
import casadi as ca

# Define the d optimization variables, i. e. the parameters that
# will be estimated; in this example, there will be only on parameter;
# the variable has to be a column vector of type casadi.casadi.MX

d = 1
x = ca.MX.sym("x", d)

# Create the model using the optimization variables; with this, the
# model automatically becomes of type casadi.casadi.MX, which is
# necessary; the variable has also to be a column vector

M = np.array([1./3., 2./3., 3./3., 4./3.]) * x[0]

# Define the standard deviations of the measurements, and store
# them within a column vector of type np.ndarray; make sure to pass
# the standard deviations, not the variances of the measurements;
# the vector has to be of the same size as the model vector M

sigma = 0.1 * np.ones(M.shape[0])

# Store the measurement data in a variable of type np.ndarray that
# also has to be a column vector and of the same size as the model
# vector M and the vector for the standard deviations sigma

Y = np.array([2.5, 4.1, 6.3, 8.2])

# Create an instance of the class PECasProb to define the parameter
# estimation problem within PECas; while x, M and sigma, which are the
# first three arguments, are mandatory, the other variables are either
# optional or mutually substitutable, and therefor need to be adressed

pep = pecas.PECasProb(x, M, sigma, Y = Y)

# With this problem set up, you can now perform a least squares
# parameter estimation for the problem, compute the covariance matrix
# for the parameters (which in this case has only one entry), and
# display the results of the computations

pep.run_parameter_estimation()
pep.compute_covariance_matrix()
pep.print_results()
