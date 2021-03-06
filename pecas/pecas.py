#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

import casadi as ca

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.misc import comb

import time
# import ipdb

import systems
import setups
import intro

from abc import ABCMeta, abstractmethod

class PECasBaseClass:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, system = None, \
        tu = None, uN = None, \
        ty = None, yN = None, \
        wv = None, weps_e = None, weps_u = None, \
        pinit = None, \
        xinit = None, \
        linear_solver = None, \
        scheme = None, \
        order = None):

        intro.pecas_intro()
        print('\n' + 22 * '-' + \
            ' PECas parameter estimation setup ' + 22 * '-')
        print('\nStarting parameter estimation problem setup ...') 


        self.linear_solver = linear_solver


        if type(system) is systems.BasicSystem:

            self.pesetup = setups.BSsetup(system = system, \
                tu = tu, uN = uN, \
                pinit = pinit)

        elif type(system) is systems.ExplODE:

            self.pesetup = setups.ODEsetup(system = system, \
                tu = tu, uN = uN, \
                ty = ty, yN = yN, \
                pinit = pinit, \
                xinit = xinit, \
                scheme = scheme, \
                order = order)

        else:

            raise NotImplementedError( \
                "The system type provided by the user is not supported.")

        # Store the parameter estimation problem setup

        # self.pesetup = pesetup

        # Check if the supported measurement data fits to the dimensions of
        # the output function

        yN = np.atleast_2d(yN)

        if yN.shape == (self.pesetup.tu.size, self.pesetup.nphi):

            yN = yN.T

        if not yN.shape == (self.pesetup.nphi, self.pesetup.tu.size):

            raise ValueError('''
The dimension of the measurement data given in yN does not match the
dimension of output function and/or tu.
Valid dimensions for yN for the given data are:
    {0} or {1},
but you supported yN of dimension:
    {2}.'''.format(str((self.pesetup.tu.size, self.pesetup.nphi)), \
    str((self.pesetup.nphi, self.pesetup.tu.size)), str(yN.shape)))

        # Check if the supported standard deviations fit to the dimensions of
        # the measurement data

        wv = np.atleast_2d(wv)

        if wv.shape == yN.T.shape:

            wv = wv.T

        if not wv.shape == yN.shape:

            raise ValueError('''
The dimension of weights of the measurement errors given in wv does not
match the dimensions of the measurement data.
Valid dimensions for wv for the given data are:
    {0} or {1},
but you supported wv of dimension:
    {2}.'''.format(str(yN.shape), str(yN.T.shape), str(wv.shape)))

        # Get the measurement values and standard deviations into the
        # necessary order of apperance and dimensions

        self.yN = np.zeros(np.size(yN))
        self.wv = np.zeros(np.size(yN))

        for k in range(yN.shape[0]):

            self.yN[k:yN.shape[0]*yN.shape[1]+1:yN.shape[0]] = \
                yN[k, :]
            self.wv[k:yN.shape[0]*yN.shape[1]+1:yN.shape[0]] = \
                wv[k, :]


        self.weps_e = []

        try:

            if self.pesetup.neps_e != 0:

                weps_e = np.atleast_2d(weps_e)

                try:

                    if weps_e.shape == (1, self.pesetup.neps_e):

                        weps_e = weps_e.T

                    if not weps_e.shape == (self.pesetup.neps_e, 1):

                        raise ValueError('''
The dimensions of the weights of the equation errors given in weps_e does not
match the dimensions of the equation errors given in eps_e.''')

                    self.weps_e = weps_e

                except AttributeError:

                    pass

                try:

                    self.weps_e = np.squeeze(ca.repmat(weps_e, self.pesetup.nsteps * \
                        (len(self.pesetup.tauroot)-1), 1))

                except AttributeError:

                    self.weps_e = []

        except AttributeError:

            pass


        self.weps_u = []

        try:

            if self.pesetup.neps_u != 0:

                weps_u = np.atleast_2d(weps_u)

                try:

                    if weps_u.shape == (1, self.pesetup.neps_u):

                        weps_u = weps_u.T

                    if not weps_u.shape == (self.pesetup.neps_u, 1):

                        raise ValueError('''
The dimensions of the weights of the input errors given in weps_u does not
match the dimensions of the input errors given in eps_u.''')

                    self.weps_u = weps_u

                except AttributeError:

                    pass

                try:

                    self.weps_u = np.squeeze(ca.repmat(weps_u, self.pesetup.nsteps * \
                        (len(self.pesetup.tauroot)-1), 1))

                except AttributeError:

                    self.weps_u = []

        except AttributeError:

            pass


        # Set up the covariance matrix for the measurements

        # self.w = ca.diag(np.concatenate((self.wv, self.weps_e,self.weps_u)))
        self.w = ca.veccat((self.wv, self.weps_e,self.weps_u))

        print('Setup of the parameter estimation problem sucessful.')        


    @property
    def phat(self):

        try:
            
            return np.array(self.Varshat[:self.pesetup.np])

        except AttributeError:
            

            raise AttributeError('''
The method run_parameter_estimation() must be run first, before trying to
obtain the optimal values.
''')


    @property
    def Xhat(self):

        xhat = np.ndarray((self.pesetup.nx, 0))

        try:

            for i in range (self.pesetup.nsteps + 1):

                xhat = np.append( \

                    xhat, \

                    self.Varshat[ \
                    self.pesetup.np + self.pesetup.nx * \
                        (self.pesetup.ntauroot+1) * i : \

                    self.pesetup.np + self.pesetup.nx * \
                        (self.pesetup.ntauroot+1) * i + self.pesetup.nx], \

                    axis = 1)

            return xhat


        except AttributeError:

            raise AttributeError('''
The method run_parameter_estimation() must be run first, before trying to
obtain the estimated state values.
''')     


class LSq(PECasBaseClass):

    '''The class :class:`LSq` is used to set up least squares parameter
    estimation problems for systems defined with one of the PECas systems
    classes, using a given set of user provided control 
    data, measurement data and different kinds of weightings.'''

    def __init__(self, system = None, \
        tu = None, uN = None, \
        ty = None, yN = None, \
        wv = None, weps_e = None, weps_u = None, \
        pinit = None, \
        xinit = None, \
        linear_solver = "mumps", \
        scheme = "radau", \
        order = 3):

        r'''
        :param system: system considered for parameter estimation, specified
                       using a PECas systems class
        :type system: pecas.systems

        :param tu: time points :math:`t_u \in \mathbb{R}^{N}`
                   for the controls (also used for
                   defining the collocation nodes)
        :type tu: numpy.ndarray, casadi.DMatrix, list

        :param uN: values for the controls at the switching time points 
                   :math:`u_N \in \mathbb{R}^{n_u \times N-1}`; note that the
                   the second dimension of :math:`u_N` is :math:`N-1` istead of
                   :math:`N`, since the control value at the last switching
                   point is never applied
        :type uN: numpy.ndarray, casadi.DMatrix

        :param ty: optional, time points :math:`t_y \in \mathbb{R}^{M}`
                   for the measurements; if no value is given, the time
                   points for the controls :math:`t_u` are used; if the values
                   in :math:`t_y` do not match with the values in :math:`t_u`,
                   a continuous
                   output approach will be used for setting up the
                   parameter estimation problem
        :type ty: numpy.ndarray, casadi.DMatrix, list

        :param yN: values for the measurements at the defined time points 
                   :math:`u_y \in \mathbb{R}^{n_y \times M}`
        :type yN: numpy.ndarray, casadi.DMatrix    

        :param wv: weightings for the measurements
                   :math:`w_v \in \mathbb{R}^{n_y \times M}`
        :type wv: numpy.ndarray, casadi.DMatrix    

        :param weps_e: weightings for equation errors
                   :math:`w_{\epsilon_e} \in \mathbb{R}^{n_{\epsilon_e}}`
                   (only necessary 
                   if equation errors are used within ``system``)
        :type weps_e: numpy.ndarray, casadi.DMatrix    

        :param weps_u: weightings for the input errors
                   :math:`w_{\epsilon_u} \in \mathbb{R}^{n_{\epsilon_u}}`
                   (only necessary
                   if input errors are used within ``system``)
        :type weps_u: numpy.ndarray, casadi.DMatrix    

        :param pinit: optional, initial guess for the values of the
                      parameters to be
                      estimated :math:`p_{init} \in \mathbb{R}^{n_p}`; if no
                      value is given, 0 will be used; a poorly or wrongly
                      chosen initial guess might cause the estimation to fail
        :type pinit: numpy.ndarray, casadi.DMatrix

        :param xinit: optional, initial guess for the values of the
                      states to be
                      estimated :math:`x_{init} \in \mathbb{R}^{n_x \times N}`;
                      if no value is given, 0 will be used; a poorly or wrongly
                      chosen initial guess might cause the estimation to fail
        :type xinit: numpy.ndarray, casadi.DMatrix

        :param linear_solver: set the linear solver for IPOPT; this option is
                              only interesting if HSL is installed
        :type linear_solver: str

        :param scheme: collocation scheme, possible values are `legendre` and
                       `radau`
        :type scheme: str

        :param order: order of collocation polynominals
                      :math:`d \in \mathbb{Z}`
        :type order: int

        '''

        super(LSq, self).__init__(system = system, \
            tu = tu, uN = uN, \
            ty = ty, yN = yN, \
            wv = wv, weps_e = weps_e, weps_u = weps_u, \
            pinit = pinit, \
            xinit = xinit, \
            linear_solver = linear_solver, \
            scheme = scheme, \
            order = order)


    def run_parameter_estimation(self, hessian = "gauss-newton"):

        r'''
        :param hessian: Method of hessian calculation/approximation; possible
                        values are `gauss-newton` and `exact-hessian`
        :type hessian: str

        This functions will run a least squares parameter estimation for the
        given problem and data set.
        For this, an NLP of the following
        structure is set up with a direct collocation approach and solved
        using IPOPT:

        .. math::

            \begin{aligned}
                & \text{arg}\,\underset{x, p, v, \epsilon_e, \epsilon_u}{\text{min}} & & \frac{1}{2} \| R(w, v, \epsilon_e, \epsilon_u) \|_2^2 \\
                & \text{subject to:} & & R(w, v, \epsilon_e, \epsilon_u) = w^{^\mathbb{1}/_\mathbb{2}} \begin{pmatrix} {v} \\ {\epsilon_e} \\ {\epsilon_u} \end{pmatrix} \\
                & & & w = \begin{pmatrix} {w_{v}}^T & {w_{\epsilon_{e}}}^T & {w_{\epsilon_{u}}}^T \end{pmatrix} \\
                & & & v_{l} + y_{l} - \phi(t_{l}, u_{l}, x_{l}, p) = 0 \\
                & & & (t_{k+1} - t_{k}) f(t_{k,j}, u_{k,j}, x_{k,j}, p, \epsilon_{e,k,j}, \epsilon_{u,k,j}) - \sum_{r=0}^{d} \dot{L}_r(\tau_j) x_{k,r} = 0 \\
                & & & x_{k+1,0} - \sum_{r=0}^{d} L_r(1) x_{k,r} = 0 \\
                & & & t_{k,j} = t_k + (t_{k+1} - t_{k}) \tau_j \\
                & & & L_r(\tau) = \prod_{r=0,r\neq j}^{d} \frac{\tau - \tau_r}{\tau_j - \tau_r}\\
                & \text{for:} & & k = 1, \dots, N, ~~~ l = 1, \dots, M, ~~~ j = 1, \dots, d, ~~~ r = 1, \dots, d \\
                & & & \tau_j = \text{time points w. r. t. scheme and order}
            \end{aligned}


        The status of IPOPT provides information whether the computation could
        be finished sucessfully. The optimal values for all optimization
        variables :math:`\hat{x}` can be accessed
        via the class variable ``LSq.Xhat``, while the estimated parameters
        :math:`\hat{p}` can also be accessed separately via the class attribute
        ``LSq.phat``.

        **Please be aware:** IPOPT finishing sucessfully does not necessarly
        mean that the estimation results for the unknown parameters are useful
        for your purposes, it just means that IPOPT was able to solve the given
        optimization problem.
        You have in any case to verify your results, e. g. by simulation using
        the class function :func:`run_simulation`.
        '''          

        intro.pecas_intro()
        print('\n' + 18 * '-' + \
            ' PECas least squares parameter estimation ' + 18 * '-')

        print('''
Starting least squares parameter estimation using IPOPT, 
this might take some time ...
''')

        self.tstart_estimation = time.time()

        g = ca.vertcat([ca.vec(self.pesetup.phiN) - self.yN + \
            ca.vec(self.pesetup.V)])

        self.R = ca.sqrt(self.w) * \
            ca.veccat([self.pesetup.V, self.pesetup.EPS_E, self.pesetup.EPS_U])

        if self.pesetup.g.size():

            g = ca.vertcat([g, self.pesetup.g])

        self.g = g

        self.Vars = ca.veccat([

                self.pesetup.P, \
                self.pesetup.X, \
                self.pesetup.XF, \
                self.pesetup.V, \
                self.pesetup.EPS_E, \
                self.pesetup.EPS_U, \

            ])


        nlp = ca.MXFunction("nlp", ca.nlpIn(x=self.Vars), \
            ca.nlpOut(f=(0.5 * ca.mul([self.R.T, self.R])), g=self.g))

        options = {}
        options["tol"] = 1e-10
        options["linear_solver"] = self.linear_solver

        if hessian == "gauss-newton":

            # ipdb.set_trace()

            gradF = nlp.gradient()
            jacG = nlp.jacobian("x", "g")

            # Can't the following be implemented more efficiently?!

            # gradF.derivative(0, 1)

            J = ca.jacobian(self.R, self.Vars)

            sigma = ca.MX.sym("sigma")
            hessLag = ca.MXFunction("H", \
                ca.hessLagIn(x = self.Vars, lam_f = sigma), \
                ca.hessLagOut(hess = sigma * ca.mul(J.T, J)))
        
            options["hess_lag"] = hessLag
            options["grad_f"] = gradF
            options["jac_g"] = jacG

        elif hessian == "exact-hessian":

            # let NlpSolver-class compute everything

            pass

        else:

            raise NotImplementedError( \
                "Requested method is not implemented. Availabe methods " + \
                "are 'gauss-newton' (default) and 'exact-hessian'.")

        # Initialize the solver, solve the optimization problem

        solver = ca.NlpSolver("solver", "ipopt", nlp, options)

        # Store the results of the computation

        Varsinit = ca.veccat([

                self.pesetup.Pinit, \
                self.pesetup.Xinit, \
                self.pesetup.XFinit, \
                self.pesetup.Vinit, \
                self.pesetup.EPS_Einit, \
                self.pesetup.EPS_Uinit, \

            ])  

        sol = solver(x0 = Varsinit, lbg = 0, ubg = 0)

        self.Varshat = sol["x"]

        R_squared_fcn = ca.MXFunction("R_squared_fcn", [self.Vars], 
            [ca.mul([ \
                ca.veccat([self.pesetup.V, self.pesetup.EPS_E, self.pesetup.EPS_U]).T, 
                ca.veccat([self.pesetup.V, self.pesetup.EPS_E, self.pesetup.EPS_U])])])

        [self.R_squared] = R_squared_fcn([self.Varshat])
        
        self.tend_estimation = time.time()
        self.duration_estimation = self.tend_estimation - \
            self.tstart_estimation

        print('''
Parameter estimation finished. Check IPOPT output for status information.''')


    def run_simulation(self, \
        x0 = None, tsim = None, usim = None, psim = None, method = "rk"):

        r'''
        :param x0: initial value for the states
                   :math:`x_0 \in \mathbb{R}^{n_x}`
        :type x0: list, numpy,ndarray, casadi.DMatrix

        :param tsim: optional, switching time points for the controls
                    :math:`t_{sim} \in \mathbb{R}^{L}` to be used for the
                    simulation
        :type tsim: list, numpy,ndarray, casadi.DMatrix        

        :param usim: optional, control values 
                     :math:`u_{sim} \in \mathbb{R}^{n_u \times L}`
                     to be used for the simulation
        :type usim: list, numpy,ndarray, casadi.DMatrix   

        :param psim: optional, parameter set 
                     :math:`p_{sim} \in \mathbb{R}^{n_p}`
                     to be used for the simulation
        :type psim: list, numpy,ndarray, casadi.DMatrix 

        :param method: optional, CasADi integrator to be used for the
                       simulation
        :type method: str

        This function performs a simulation of the system for a given
        parameter set :math:`p_{sim}`, starting from a user-provided initial
        value for the states :math:`x_0`. If the argument ``psim`` is not
        specified, the estimated parameter set :math:`\hat{p}` is used.
        For this, a parameter
        estimation using :func:`run_parameter_estimation()` has to be
        done beforehand, of course.

        By default, the switching time points for
        the controls :math:`t_u` and the corresponding controls 
        :math:`u_N` will be used for simulation. If desired, other time points
        :math:`t_{sim}` and corresponding controls :math:`u_{sim}`
        can be passed to the function.

        For the moment, the function can only be used for systems of type
        :class:`pecas.systems.ExplODE`.

        '''

        intro.pecas_intro()
        print('\n' + 27 * '-' + \
            ' PECas system simulation ' + 26 * '-')
        print('\nPerforming system simulation, this might take some time ...') 

        if not type(self.pesetup.system) is systems.ExplODE:

            raise NotImplementedError("Until now, this function can only " + \
                "be used for systems of type ExplODE.")


        if x0 == None:

            raise ValueError("You have to provide an initial value x0 " + \
                "to run the simulation.")


        x0 = np.squeeze(np.asarray(x0))

        if np.atleast_1d(x0).shape[0] != self.pesetup.nx:

            raise ValueError("Wrong dimension for initial value x0.")


        if tsim == None:

            tsim = self.pesetup.tu


        if usim == None:

            usim = self.pesetup.uN


        if psim == None:

            try:

                psim = self.phat

            except AttributeError:

                errmsg = '''
You have to either perform a parameter estimation beforehand to obtain a
parameter set that can be used for simulation, or you have to provide a
parameter set in the argument psim.
'''
                raise AttributeError(errmsg)

        else:

            if not np.atleast_1d(np.squeeze(psim)).shape[0] == self.pesetup.np:

                raise ValueError("Wrong dimension for parameter set psim.")


        fp = ca.MXFunction("fp", \
            [self.pesetup.system.t, self.pesetup.system.u, \
            self.pesetup.system.x, self.pesetup.system.eps_e, \
            self.pesetup.system.eps_u, self.pesetup.system.p], \
            [self.pesetup.system.f])

        fpeval = fp([\
            self.pesetup.system.t, self.pesetup.system.u, \
            self.pesetup.system.x, np.zeros(self.pesetup.neps_e), \
            np.zeros(self.pesetup.neps_u), psim])[0]

        fsim = ca.MXFunction("fsim", \
            ca.daeIn(t = self.pesetup.system.t, \
                x = self.pesetup.system.x, \
                p = self.pesetup.system.u), \
            ca.daeOut(ode = fpeval))


        Xsim = []
        Xsim.append(x0)

        u0 = ca.DMatrix()

        for k,e in enumerate(tsim[:-1]):

            try:

                integrator = ca.Integrator("integrator", method, \
                    fsim, {"t0": e, "tf": tsim[k+1]})

            except RuntimeError as err:

                errmsg = '''
It seems like you want to use an integration method that is not currently
supported by CasADi. Please refer to the CasADi documentation for a list
of supported integrators, or use the default RK4-method by not setting the
method-argument of the function.
'''
                raise RuntimeError(errmsg)


            if not self.pesetup.nu == 0:

                u0 = usim[:,k]


            Xk_end = itemgetter('xf')(integrator({'x0':x0,'p':u0}))

            Xsim.append(Xk_end)
            x0 = Xk_end


        self.Xsim = ca.horzcat(Xsim)

        print( \
'''System simulation finished.''')


    def compute_covariance_matrix(self):

        r'''
        This function computes the covariance matrix of the estimated
        parameters from the inverse of the KKT matrix for the
        parameter estimation problem. This allows then for statements on the
        quality of the values of the estimated parameters.

        For efficiency, only the inverse of the relevant part of the matrix
        is computed using the Schur complement.

        A more detailed description of this function will follow in future
        versions.

        '''

        intro.pecas_intro()
        
        print('\n' + 20 * '-' + \
            ' PECas covariance matrix computation ' + 21 * '-')

        print('''
Computing the covariance matrix for the estimated parameters, 
this might take some time ...
''')

        self.tstart_cov_computation = time.time()

        try:

            N1 = ca.MX(self.Vars.shape[0] - self.w.shape[0], \
                self.w.shape[0])

            N2 = ca.MX(self.Vars.shape[0] - self.w.shape[0], \
                self.Vars.shape[0] - self.w.shape[0])

            hess = ca.blockcat([[N2, N1], [N1.T, ca.diag(self.w)],])

            # hess = hess + 1e-10 * ca.diag(self.Vars)
            
            # J2 can be re-used from parameter estimation, right?

            J2 = ca.jacobian(self.g, self.Vars)

            kkt = ca.blockcat( \

                [[hess, \
                    J2.T], \

                [J2, \
                    ca.MX(self.g.size1(), self.g.size1())]] \

                    )

            B1 = kkt[:self.pesetup.np, :self.pesetup.np]
            E = kkt[self.pesetup.np:, :self.pesetup.np]
            D = kkt[self.pesetup.np:, self.pesetup.np:]

            Dinv = ca.solve(D, E, "csparse")

            F11 = B1 - ca.mul([E.T, Dinv])

            self.fbeta = ca.MXFunction("fbeta", [self.Vars], 
                [ca.mul([self.R.T, self.R]) / \
                (self.yN.size + self.g.size1() - self.Vars.size())])

            [self.beta] = self.fbeta([self.Varshat])

            self.fcovp = ca.MXFunction("fcovp", [self.Vars], \
                [self.beta * ca.solve(F11, ca.MX.eye(F11.size1()))])

            [self.Covp] = self.fcovp([self.Varshat])

            print( \
'''Covariance matrix computation finished, run show_results() to visualize.''')


        except AttributeError as err:

            errmsg = '''
You must execute run_parameter_estimation() first before the covariance
matrix for the estimated parameters can be computed.
'''

            raise AttributeError(errmsg)


        finally:

            self.tend_cov_computation = time.time()
            self.duration_cov_computation = self.tend_cov_computation - \
                self.tstart_cov_computation


    def show_results(self):

        r'''
        :raises: AttributeError

        This function displays the results of the parameter estimation
        computations. It can not be used before function
        :func:`run_parameter_estimation()` has been used. The results
        displayed by the function contain:
        
          - the values of the estimated parameters :math:`\hat{p}`
            and their corresponding standard deviations
            (the values of the standard deviations are presented
            only if the covariance matrix had already been computed),
          - the values of the covariance matrix
            :math:`\Sigma_{\hat{p}}` for the
            estimated parameters (if it had already been computed),
          - in the case of the estimation of a dynamic
            system, the estimated value of the first state 
            :math:`\hat{x}(t_{0})` and the estimated value 
            of the last state :math:`\hat{x}(t_{N})`,
          - the value of :math:`R^2` measuring the goodness of fit
            of the estimated parameters, and
          - the durations of the setup and the estimation.
        '''

        intro.pecas_intro()

        np.set_printoptions(linewidth = 200, \
            formatter={'float': lambda x: format(x, ' 10.8e')})

        try:

            print('\n' + 21 * '-' + \
                ' PECas Parameter estimation results ' + 21 * '-')
             
            print("\nEstimated parameters p_i:")

            for i, xi in enumerate(self.phat):
            
                try:

                    print("    p_{0:<3} = {1: 10.8e} +/- {2: 10.8e}".format(\
                         i, xi[0], ca.sqrt(abs(self.Covp[i, i]))))

                except AttributeError:

                    print("    p_{0:<3} = {1: 10.8e}".format(\
                        i, xi[0]))
            
            try:

                print("\nInitial states value estimated from collocation:  ")
                print("    x(t_0) = {0}".format(self.Xhat[:,0]))
                
                print("\nFinal states value estimated from collocation:  ")
                print("    x(t_N) = {0}".format(self.Xhat[:,-1]))
            
            except AttributeError:

                pass

            print("\nCovariance matrix for the estimated parameters:")

            try:

                print(np.atleast_2d(self.Covp))

            except AttributeError:

                print( \
'''    Covariance matrix for the estimated parameters not yet computed.
    Run class function compute_covariance_matrix() to do so.''')

            print( \
                "\nGoodness of fit R^2" + 30 * "." + ": {0:10.8e}".format(\
                    float(self.R_squared)))

            print("\nDuration of the problem setup"+ 20 * "." + \
                ": {0:10.8e} s".format(self.pesetup.duration_setup))
            
            print("Duration of the estimation" + 23 * "." + \
                ": {0:10.8e} s".format(self.duration_estimation))

            try:

                print("Duration of the covariance matrix computation...." + \
                    ": {0:10.8e} s".format(self.duration_cov_computation))

            except AttributeError:

                pass

        except AttributeError:

            raise AttributeError('''
You must execute at least run_parameter_estimation() to obtain results,
and compute_covariance_matrix() before all results can be displayed.
''')   

        finally:

            np.set_printoptions()


    def plot_confidence_ellipsoids(self, indices = []):

        r'''
        :param indices: List of the indices of the parameters in
                        :math:`\hat{p}` for
                        which the confidence ellipsoids shall be plotted.
                        The indices must be defined by list entries of type
                        *int*. If an empty list is supported (default),
                        the ellipsoids for all parameters are plotted.

        :type indices: list
        :raises: AttributeError, ValueError, TypeError

        This function plots the confidence ellipsoids pairwise for all
        parameters defined in ``indices``. The plots are displayed in subplots
        inside of one plot window.
        '''

        if not hasattr(self, "Covp"):

            raise AttributeError('''
You must execute both run_parameter_estimation() and
compute_covariance_matrix() before the confidece ellipsoids can be plotted.
''')

        if type(indices) is not list:

            raise TypeError('''
The variable containing the indices of the parameters has to be of type list.
''')

        # If the list of indices is empty, create a list that contains
        # all indices

        if len(indices) == 0:
            indices = range(0, self.pesetup.np)

        if len(indices) == 1:
            raise ValueError('''
A confidence ellipsoid can not be plotted for only one single parameter. The
list of indices must therefor contain more than only one entry.
''')            

        for ind in indices:

            if type(ind) is not int:
                
                raise TypeError('''
All list entries for the indices have to be of type int.
''')

        nplots = int(round(comb(len(indices), 2)))

        # TODO! Don't plot! Save every figure, system temp folder as default, 
        # set different location by argument

        plotfig = plt.figure()
        plcount = 1

        xy = np.array([np.cos(np.linspace(0,2*np.pi,100)), \
                    np.sin(np.linspace(0,2*np.pi,100))])

        for j, ind1 in enumerate(indices):

            for k, ind2 in enumerate(indices[j+1:]):

                covs = np.array([ \

                        [self.Covp[ind1, ind1], self.Covp[ind1, ind2]], \
                        [self.Covp[ind2, ind1], self.Covp[ind2, ind2]] \

                    ])

                w, v = np.linalg.eig(covs)

                ellipse = ca.mul(np.array([self.phat[ind1], \
                    self.phat[ind2]]), \
                    np.ones([1,100])) + ca.mul([v, np.diag(w), xy])

                ax = plotfig.add_subplot(nplots, 1, plcount)
                ax.plot(np.array(ellipse[0,:]).T, np.array(ellipse[1,:]).T, \
                    label = 'p' + str(ind1) + ' - p' + str(ind2))
                ax.scatter(self.phat[ind1], self.phat[ind2])
                ax.legend(loc="upper left")

                plcount += 1

        plt.show()
