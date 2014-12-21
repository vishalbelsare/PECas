#!/usr/bin/python

# ex3.py: Generate and return pseudo measurement data in PECas.

# First, define the several components of the parameter estimation problem
# just as before, but without specifying the measurement values.

import pecas
import numpy as np
import casadi as ca 

d = 2
x = ca.SX.sym("x", d)   

M = ca.mul(np.matrix([np.ones(4), range(1,5)]).T, \
    ca.vertcat((x[0], x[1]**2)))

G = 2 - ca.mul(x.T,x)

sigma = 0.5 * np.ones(M.shape[0])

xinit = np.array([1, 1])

# Then, define a column vector xtrue of type numpy.ndarray for the true
# values of the parameters, and create an instance pep of the class
# PECasProb providing all information.

xtrue = np.array([1, 1])

pep = pecas.PECasProb(x, M, sigma, xtrue = xtrue, G = G, xinit = xinit)

# Afterwards, the class instance pep can be used to generate pseudo
# measurement data that will be stored as measurement data inside the object.

pep.generate_pseudo_measurement_data()

# The generated data can be returned using the get-method for the
# measurement data, and then e. g. be stored in another variable meas_data
# for further usage.

meas_data = pep.get_Y()
