#!/usr/bin/python

# ex4p1.py: Example 4 shows the optimization in PECas of a general maximum
#           likelihood problem, using as a model a one dimensional racing car.
#           Moreover it shows how to simulate the obtained model, compute
#           the covariance matrix and the confidence ellipsoids of the 
#           estimated parameters.
#           This example is divided in 2 problems. The first one deals with
#           a model where already one parameter is know (C1), and the input
#           of the system are constant on time (that arises an observability
#           problem on C1, forcing the pre-knowledge of its value). This
#           example also ilustrate the initialization of the matrix model M
#           when the size it is too big for doing it manually.  

#==============================================================================
# First, import all the modules that are necessary for using PECas. Note that
# pylab integrates the main numpy, scipy and matplotlib classes and methods,
# thus the import of numpy is implicit
#==============================================================================
import pecas as pc
import pylab as pl
import casadi as ca

#==============================================================================
# First, import the problem data. The data format is [time;velocity;dutycycle]
# where time, velocity and dutycycle are columm vectors, representing the
# velocity measurements of the the racing car due to variable car motor
# dutycycles as actuation at different times.
#==============================================================================
data = pl.loadtxt('ex4p1data.txt')

#==============================================================================
# Structure the data in columm vectors for better handling, and get data size M
#==============================================================================
t = data[:, 0]
vxm = data[:, 1]
Dk = data[:, 2]
N = t.size;

#==============================================================================
# Then, define the m non-optimization variables, i.e. the parameters that will
# represent numerical values on the model later on, but that initially must be 
# defined as column vector of the type casadi.casadi_core.SX, so that it is 
# possible later on to build the race car model as a casadi symbolic 
# expression (also of the type casadi.casadi_core.SX). In the model D will  the 
# represent actuation, dT the time interval between 2 actuations, and vxp the  of 
# value the velocity in a previous step. 
#==============================================================================
D = ca.SX.sym("D", 1)
dT = ca.SX.sym("dT", 1)
vxp = ca.SX.sym("vxb", 1)

#==============================================================================
# Define the the pre-known optimal value of the parameter C1.
#==============================================================================
C1 = 10

#==============================================================================
# Then, define the d optimization variables, i. e. the parameters that
# will be estimated. Here, there will be 3 parameters. C2, C3 representing
# the zeroth and first order friction constants respectively, and vx0
# the initial velocity of the car. All the variables have to be column vectors
# of type casadi.casadi_core.SX.
#==============================================================================
C2 = ca.SX.sym("C2", 1)
C3 = ca.SX.sym("C3", 1)
vx0 = ca.SX.sym("vx0", 1)

#==============================================================================
# A symbolic expression called vx and representing the car velocity model is
# built. This expression is still of the type casadi.casadi_core.SX
#==============================================================================
vx = (C1 * D - C2) * (1 / C3) * (1 - pl.exp(-C3 * dT)) + \
    vxp * pl.exp(-C3 * dT)

#==============================================================================
# A symbolic function that uses the previous symbolic expression vx is built.
# The function is needed to build the mandatory model M (which PECas uses to
# solve the problem) as a column vector of size N & type casadi.casadi_core.SX.
# Note that in example1 that was done manually because the model contained only
# 4 elements. In real problems, where the size is larger, the following method
# is better. The idea of the symbolic function is that it can be partially
# numerically evaluated respect some of its symbolic variables, producing as an
# output a column vector of the type casadi.casadi_core.SX. Therefore, with a
# iteration over the different values of the actuation and time vectors, the N
# elements of the matrix model M can be built as a iterative evaluation of the
# symbolic function on the discretize actuation and time values. The symbolic
# function has to be of the type casadi.casadi_core.SXFunction, and before it
# can be used it must be initialized. Note that the function name parameter is
# optional.
# SXfunction general constructor scheme f([input1,...,inpN],[output1,...,outN])
#==============================================================================
fvx = ca.SXFunction([C2, C3, vxp, dT, D], [vx])
fvx.setOption("name", "fvx")
fvx.init()

#==============================================================================
# Define the variable that PECas will use to optimize. It will be a column
# vector of the type casadi.casadi_core.SX. of size 3, representing the 3
# parameters to estimate: C2, C3 and vx0. Despite they were already declared
# before as 1x1 casadi.casadi_core.SX to build the symbolic function, they have
# to be re-declared as a single vector of size 3 to match the proper input 
# scheme of the PECas function PECasProb.
#==============================================================================
Theta = ca.SX.sym("Theta", 3)

#==============================================================================
# Define the model M that PECasProb will use as an input. As it was stated 
# before, M is a column vector of type casadi.casadi_core.SX and size
# N, which is created in a iterative fashion evaluating the symbolic function 
# fvx in a way that:
#     - Each element of M is a fvx evaluation or the corresponding actuation D 
#     and time values.
#     - The SX variables corresponding to C2 and C3 are constantly evaluated
#     as Theta[0] and Theta[1], so that Pecas optimize them.
#     - Theta[2] is included only int the first element of M as the initial 
#     velocity vx0.
# The model M is initialized as simvx. Then, its first value at time 0 computed,
# and finally the rest of the N-1 elements calculated with a loop. 
# Note that using the [0] at the end of the symbolic function fvx is necessary.
# Evaluation of a such a symbolic function returns several values and the 
# symbolic expression is just the first of them.
#==============================================================================
simvx = ca.SX.sym("simvx",N)
simvx[0] = fvx([Theta[0], Theta[1], Theta[2], 0, Dk[0]])[0]

for k in xrange(1, N):

    simvx[k] = fvx([Theta[0], Theta[1], simvx[k-1], t[k] - t[k-1], Dk[k-1]])[0]

#==============================================================================
# Since the standar deviation of the measurements is unknow, and a initial
# guess of the parameters is not known or provided, they are both initialized
# as a column vector of type numpy.ndarray.
#==============================================================================
sigma = pl.ones(N)
xinit = pl.ones(3)

#==============================================================================
# Finally, create an instance pep of the class PECasProb to define the parameter
# estimation problem within PECas. Remember that only x, M and sigma (first 
# three arguments) are mandatory, since the other variables are not, they
# need to be adressed. Then the least squares parameter estimation for the
# problem can be obtained.
#==============================================================================
pep = pc.PECasProb(Theta, simvx, sigma, Y = vxm, xinit = xinit)
pep.run_parameter_estimation()

#==============================================================================
# To simulate the system, the model M is evaluated using the optimal parameters
# obtained. For that, a new symbolic function to evaluate the symbolic
# expressions of simvx must be created. The input/output scheme used (nlpIn
# nlpOut) is pre-defined in Casadi for nonlinear programming, but helpful in
# this case just to generate the SX function and directly indicate the input and
# output as vectors of SX expressions. Before using it the function must be
# initialized.
#==============================================================================
fsimvx = ca.SXFunction(ca.nlpIn(x = Theta), ca.nlpOut(f =simvx))
fsimvx.init()

#==============================================================================
# Once created, the function must be initialized. Then the input is set as the
# Theta optimum obtained for PECas, so the optimized model can be simulated.
# Finally the function is evaluated, and the output of such evaluation stored in
# a variable called fsim. Note that since this time the SX expression is
# fully numerical evaluated, the type of fsim is casadi.casadi_core.DMatrix,
# the sort of casadi structure used as numerical output functions. Do not
# use directly such a numerical structure for heavy numerical computations,
# instead copy its content to a numpy array structure.
#==============================================================================
fsimvx.setInput(pep.get_xhat(), "x")
fsimvx.evaluate()
fsim = fsimvx.getOutput("f")

#==============================================================================
# Almost at the end, the comparision between the data provided and the model
# obtained by optimization has to be done. Using the scatter function of
# matplotlib the velocity data measurement data are plotted versus time
#==============================================================================
pl.scatter(t, vxm)

#==============================================================================
# The model obtained and simulated within fsim is also plotted versus time. To
# do it in a continuous fashion, the comand plot of matplotlib is used. Finally
# the show function creates one graph over the other, so that the comparision
# is available.
#==============================================================================
pl.plot(t, fsim)
pl.show()

#==============================================================================
# Right at the end, the covariance matrix of the estimators is computed.
# Then printing the results can be done. At this point you should be able to
# see the whole output of the parameter optimization, i.e. beta, the residual 
# Rhat, the estimated parameters + standard deviations and the covariance matrix.
#==============================================================================
pep.compute_covariance_matrix()
pep.print_results()


pep.plot_confidence_ellipsoids()