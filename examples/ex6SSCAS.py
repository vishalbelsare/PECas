#!/usr/bin/python
#
#==============================================================================
#  ex6.py: Example 6 shows how to estimate the parameters of a general system
#          described by an ODE with no analitical solution. To create a model
#          that the solver can handle, a single shooting approach using the
#          integrator class from casadi is used. Furthermore, the example
#          illustrates a parametric identification where two different kind of
#          measurements (angle and angular speed) are used, which have
#          different standar deviation.
#          The particular system is a pendulum, with a manuver as the actuator,
#          and a torsional spring attaching the pendulum to the manuver. The
#          problem tries to estimate the spring constant, and initial velocity
#          and pendulum angle.
#==============================================================================


#==============================================================================
# Importing neccesary modiles
#==============================================================================
import casadi as ca
import pylab as pl
import pecas as pc


#==============================================================================
# The example is writen as a function for profiling purpouses, using the two
# profiling tools line_profiler (https://github.com/rkern/line_profiler) and
# the memory_profiler (https://github.com/fabianp/memory_profiler)
#==============================================================================
def mainEx6SS():

#==============================================================================
# Loading data
#==============================================================================
    data = pl.loadtxt('ex6data.txt')
    t = data[:, 0]
    phim = data[:, 1]
    wm = data[:, 2]
    N = t.size

#==============================================================================
# Defining constant problem parameters: 
#     - m: representing the ball of the mass in Kg
#     - L: the length of the pendulum bar in meters
#     - g: the gravity constant in m/s^2
#     - psi: the actuation angle of the manuver in radians, which stays
#     constant for this problem
#==============================================================================
    m = 1
    L = 3
    g = 9.81
    psi = pl.pi/2

#==============================================================================
# Defining the 2 symbolic variables that represent the state of the system:
#     - phi: rotational angle
#     - w: angular speed
#==============================================================================
    phi = ca.SX.sym("phi", 1)
    w = ca.SX.sym("w", 1)

#==============================================================================
# Defining the 3 symbolic variables that represent the parameters to estimate:
#     - phi0: rotational angle at time t=0
#     - w0: angular speed at time t=0
#     - K: rotatational spring constant
#==============================================================================
    phi0 = ca.SX.sym("phi0", 1)
    w0 = ca.SX.sym("w0", 1)
    K = ca.SX.sym("K", 1)

#==============================================================================
# Defining the 2 symbolic ODE that determine the state:
#     - wdot: ODE for the angular speed
#     - phidot: ODE for the rotational angle
#==============================================================================
    wdot = K/(m*(L**2))*(psi-phi) - g/L * pl.sin(phi)
    phidot = w

#==============================================================================
# Grouping symbolic ODEs, states and parameter in columm vectors shape for 
# better and more elegant coding:
#     - fxs: vector with the 2 symbolic ODE
#     - xs: vector with the 2 symbolic states
#     - ps: vector with the 3 symbolic parameters
# Note that the problem could be solved just defining this 3 SX variables alone,
# the pre-definiton of individual ODEs, states and parameters was just done for
# easier interpretation of the problem model.
#==============================================================================
    fxs = ca.vertcat([phidot, wdot])
    xs = ca.vertcat([phi, w])
    ps = ca.vertcat([phi0, w0, K])

#==============================================================================
# The Model that will be provided to Pecas must contain the simulated values of
# the state x as a function of the parameters p. To obatin these values, the 
# ODE must be solved & evaluated for the different time steps defined in t. In 
# order to do that, several methods can be used (euler, RK4), in this example
# and for the sake of accuracy, the Casadi integrator class is used. The
# integrator solves an initial value problem defined by a differential
# algebraic equation (DAE).
# The line below defines the symbolic general DAE that the Casadi integrator
# requires as a input. The DAE input/output scheme of Casadi is used to
# define explicitly the state, parameter and ODE column vectors.
#==============================================================================
    ODE = ca.SXFunction(ca.daeIn(x=xs, p=ps), ca.daeOut(ode=fxs))
    

#==============================================================================
# Defining the Casadi integrator. The integrator has different plugins (cvodes,
# idas...), in this example the general cvodes for ODEs is used. Moreover the
# tolerances of the integrator are set and the initial time step for integration is
# defined. Before use it, the integrator function must be initialized.
#==============================================================================
    integrator = ca.Integrator("cvodes", ODE)
    integrator.setOption("abstol", 1e-6)
    integrator.setOption("reltol", 1e-6)
    integrator.setOption("tf", t[1]-t[0])
    integrator.init()
        
#==============================================================================
# Define the symbolic variable that PECas will use to optimize, as well as the
# Model vector that PECas will get as an input. Since the integrator output
# for symbolic expressions is  is of the type casadi.casadi_core.MX, and since
# the Cvodes interface requires that the input x0 and parameters p have to be 
# of the type casadi.casadi_core.MX to evaluate the sparsity, both (Model and
# variable Theta have to be also of type casadi.casadi_core.MX).
#       - Theta: matches the parameter structure used before to define the
#        input of the ODE, [phi0,w0,K]
#       - Model: storages the simulated values of the integrator at the
#        different times steps. Since it contains both, angular speed and
#        rotational angle values it must be formatted in a particular way,
#        [phi0,phi1.....,phiN,w0,w1,.......,wN].T
#==============================================================================
    Theta = ca.MX.sym("Theta", 3)
    Model = ca.MX.zeros(2*N, 1)

#==============================================================================
# Define the variable representing the current state X at any time k, and asigns
# its initial value at time t=0: X(t=0) = [phi0,w0] = [Theta[0], Theta[1]]. The
# first value of the Model are also assigned.
#==============================================================================
    X = ca.vertcat([Theta[0], Theta[1]])
    Model[0] = X[0]
    Model[t.size] = X[1]
    
#==============================================================================
# In a iterative fashion, the different discrete states X(k) = [phi(k), w(k)] 
# are obtained. For that, we use the Casadi integrator function, which 
# integrates the ODE of the system on a small time step ΔT, taking as initial
# state X(k-1) and ending at X(k). IThe first state has to be the unknown
# the X(t=0) = [phi0,w0] = [Theta[0], Theta[1]] that will be estimated.
# Using the Casadi integrator input scheme, the 2 required inputs are defined:
#     - x0: representing the integration initial state. It is the state at the 
#     previous time step, thus, x0 = X(k-1). x0 has to be of type 
#     casadi.casadi_core.MX, and of shape equal to the shape of xs (since xs 
#     represented the state on the ODE definition).
#     - p: the parameters of the ODE, and in turn, they have to take
#     take the value of Theta. Note that since only the spring constant K
#     appears explicitly in the ODE, the parameter vector ps (ps was the
#     variable defining the parameters on the ODE definition)  as well this 
#     input p could have been defined as ps = p = [K] = Theta[2], instead of 
#     the current [phi0,w0,k]. However, to use a single terminology por the 
#     estimated parameters, they can be all included in a single vector without 
#     any difference.The input p is also of type casadi.casadi_core.MX, and of 
#     shape equal to ps. 
# To obtain the different X(k), the value of the variable X is iteratively
# reassigned, and its value moved to the vector containing the Model.
#==============================================================================
    for k in xrange(1, N):
        integrator.setOption("tf", t[k]-t[k-1])
        X = ca.integratorOut(integrator(ca.integratorIn(x0=X,
                              p=Theta)), "xf")[0]
        Model[k] = X[0]
        Model[k+t.size] = X[1]
    
#==============================================================================
# Definition of the standar deviations each of the measurements. Since no data
# is provided before hand, it is assumed that both, angular speed and rotational
# angle have i.i.d. noise, thus the measurement have identic standar deviation
# and they can be calculated. The vector that is provided to PECas must have
# the standar deviation of each measurement, and thus it is of size 2*N.
# Finally, the optimization problem is initialized using some random vector
# [phi,w,K] = [1,1,1], and the vector with all the measurement is created.
#==============================================================================
    sigmaphi = pl.ones(t.size)*pl.std(phim, ddof=1)
    sigmaw = pl.ones(t.size)*pl.std(wm, ddof=1)
    sigma = pl.concatenate([sigmaphi, sigmaw])
    xinit = pl.ones(3)
    Y = pl.concatenate([phim, wm])

#==============================================================================
# Finally, using PECas as in the previous examples, the optimal parameters are
# calculated as well as the covariance matrix, the residual and the beta factor.
#==============================================================================
    pep = pc.PECasLSq(Theta, Model, sigma, Y=Y, xinit=xinit)
    pep.run_parameter_estimation()
    print pep.get_xhat()
    pep.compute_covariance_matrix()
    pep.print_results()

    return
    
if __name__ == "__main__":
    mainEx6SS()
