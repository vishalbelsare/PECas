#!/usr/bin/python

# ex1.py: A simple example to show the general usage of PECas

# First, import all the modules that are necessary for using PECas

import pecas
import numpy as np
import casadi as ca

# Then, define the d optimization variables, i. e. the parameters that
# will be estimated. In this example, there will be only one parameter.
# The variable x has to be a column vector of type casadi.casadi.MX.

d = 1
x = ca.MX.sym("x", d)

# Afterwards, create the model using the optimization variables. By doing
# this, the model automatically becomes of the necessary type casadi.casadi.MX.
# The variable M has also to be a column vector and of size N.

M = np.array([1./3., 2./3., 3./3., 4./3.]) * x[0]

# Define the standard deviations of the measurements, and store them
# within a column vector sigma of type np.ndarray. Make sure to pass
# the standard deviations, not the variances of the measurements.
# The vector has to be of the same size N as the model vector M.

sigma = 0.1 * np.ones(M.shape[0])

# Store the measurement data in a variable of type np.ndarray that
# also has to be a column vector and of the same size as the model
# vector M and the vector for the standard deviations sigma.

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
