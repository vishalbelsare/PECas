#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat

import numpy as np
from operator import itemgetter
# from scipy.misc import comb

import time

import systems
import setups
import intro

from abc import ABCMeta, abstractmethod

class PECasBaseClass:

    __metaclass__ = ABCMeta

    # @profile
    @abstractmethod
    def __init__(self, system = None, \
        tu = None, uN = None, \
        ty = None, yN = None, \
        wv = None, wwe = None, wwu = None, \
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


        self.wwe = []

        try:

            if self.pesetup.nwe != 0:

                wwe = np.atleast_2d(wwe)

                try:

                    if wwe.shape == (1, self.pesetup.nwe):

                        wwe = wwe.T

                    if not wwe.shape == (self.pesetup.nwe, 1):

                        raise ValueError('''
The dimensions of the weights of the equation errors given in wwe does not
match the dimensions of the equation errors given in we.''')

                    self.wwe = wwe

                except AttributeError:

                    pass

                try:

                    # if self.ww is not None:

                        self.wwe = np.squeeze(ca.repmat(wwe, self.pesetup.nsteps * \
                            (len(self.pesetup.tauroot)-1), 1))

                except AttributeError:

                    self.wwe = []

        except AttributeError:

            pass


        self.wwu = []

        try:

            if self.pesetup.nwu != 0:

                wwu = np.atleast_2d(wwu)

                try:

                    if wwu.shape == (1, self.pesetup.nwu):

                        wwu = wwu.T

                    if not wwu.shape == (self.pesetup.nwu, 1):

                        raise ValueError('''
The dimensions of the weights of the input errors given in wwu does not
match the dimensions of the input errors given in wu.''')

                    self.wwu = wwu

                except AttributeError:

                    pass

                try:

                    # if self.ww is not None:

                        self.wwu = np.squeeze(ca.repmat(wwu, self.pesetup.nsteps * \
                            (len(self.pesetup.tauroot)-1), 1))

                except AttributeError:

                    self.wwu = []

        except AttributeError:

            pass


        # Set up the covariance matrix for the measurements

        self.W = ca.diag(np.concatenate((self.wv, self.wwe,self.wwu)))

        print('Setup of the parameter estimation problem sucessful.')        


    @property
    def phat(self):

        try:
            
            return np.array(self.pesetup.Vars()(self.Varshat)["P"])

        except AttributeError:
            

            raise AttributeError('''
The method run_parameter_estimation() must be run first, before trying to
obtain the optimal values.
''')


    @property
    def Xhat(self):

        xhat = []

        try:

            for i in range (self.pesetup.nx):
                T = self.pesetup.Vars()(self.Varshat)["X",:,0,i]
                T.append(self.pesetup.Vars()(self.Varshat)["XF",i])
                xhat.append(T)
            return np.array(xhat)

        except AttributeError:

            raise AttributeError('''
The method run_parameter_estimation() must be run first, before trying to
obtain the optimal values.
''')     


class LSq(PECasBaseClass):

    '''The class :class:`LSq` is used to set up least squares parameter
    estimation problems for systems defined with one of the PECas systems
    classes, using a given set of user provided control 
    data, measurement data and different kinds of weightings.'''

    def __init__(self, system = None, \
        tu = None, uN = None, \
        ty = None, yN = None, \
        wv = None, wwe = None, wwu = None, \
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

        :param wwe: weightings for equation errors
                   :math:`w_{w_e} \in \mathbb{R}^{n_{w_e}}` (only necessary 
                   if equation errors are used within `system`)
        :type wwe: numpy.ndarray, casadi.DMatrix    

        :param wwu: weightings for the input errors
                   :math:`w_{w_u} \in \mathbb{R}^{n_{w_u}}` (only necessary
                   if input errors are used within `system`)
        :type wwu: numpy.ndarray, casadi.DMatrix    

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
            wv = wv, wwe = wwe, wwu = wwu, \
            pinit = pinit, \
            xinit = xinit, \
            linear_solver = linear_solver, \
            scheme = scheme, \
            order = order)


    # @profile
    def run_parameter_estimation(self):

        r'''
        This functions will run a least squares parameter estimation for the
        given problem and data set.
        For this, an NLP of the following
        structure is set up with a direct collocation approach and solved
        using IPOPT:

        .. math::

            \begin{aligned}
                & \text{arg}\,\underset{x, p, v, w_e, w_u}{\text{min}} & & \| v \|_{W_{v}}^{2} + \| w_{e} \|_{W_{w_{e}}}^{2} + \| w_{u} \|_{W_{w_{u}}}^{2}\\
                & \text{subject to:} & & y_{l} - \phi(t_{l}, u_{l}, x_{l}, p) + v_{l} = 0 \\
                & & & (t_{k+1} - t_{k}) f(t_{k,j}, u_{k,j}, x_{k,j}, p, w_{e,k,j}, w_{u,k,j}) - \sum_{r=0}^{d} \dot{L}_r(\tau_j) x_{k,r} = 0 \\
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

        g = ca.vertcat([self.pesetup.phiN - self.yN + \
                ca.vertcat(self.pesetup.Vars["V"])])

        A = self.pesetup.Vars["V"]

        if "WE" in self.pesetup.Vars.keys():

            # W = []

            # for k, elem in enumerate(self.pesetup.Vars["WE"]):

            #     W.append(elem)

            # A = A + sum(W, [])

            A = A + sum(self.pesetup.Vars["WE"], [])


        if "WU" in self.pesetup.Vars.keys():

            # W = []

            # for k, elem in enumerate(self.pesetup.Vars["WU"]):

            #     W.append(elem)

            # A = A + sum(W, [])

            A = A + sum(self.pesetup.Vars["WU"], [])

        A = ca.vertcat(A)

        self.reslsq = ca.mul([A.T, self.W, A])

        self.A = A


        if self.pesetup.g.size():

            g = ca.vertcat([g, self.pesetup.g])

        self.g = g


        reslsqfcn = ca.MXFunction("reslsqfcn", ca.nlpIn(x=self.pesetup.Vars), \
            ca.nlpOut(f=self.reslsq, g=g))

        reslsqfcn = reslsqfcn.expand()

        # Initialize the solver, solve the optimization problem

        solver = ca.NlpSolver("solver", "ipopt", reslsqfcn, \
            {"tol":1e-1, "linear_solver":self.linear_solver})

        # Store the results of the computation

        sol = solver(x0 = self.pesetup.Varsinit, \
            lbg = 0, ubg = 0)

        self.Varshat = sol["x"]
        self.rhat = sol["f"]
        self.dv_lambdahat = sol["lam_g"]
        
        # Ysim = self.pesetup.phiNfcn([self.Varshat])[0]
        # Ym = np.reshape(self.yN.T,(Ysim.shape))
        # res = Ym-Ysim
        # self.residual = []
        # self.Rsquared = []

        # for i in range(self.pesetup.ny):   

        #     self.residual.append(\
        #         (np.linalg.norm(res[i:-1:self.pesetup.ny]))**2)
        #     self.Rsquared.append(1 - self.residual[i] / \
        #         (np.linalg.norm(Ym[i:-1:self.pesetup.ny]))**2)
        
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

        if not type(self.pesetup.system) is systems.ExplODE:

            raise NotImplementedError("Until now, this function can only " + \
                "be used for systems of type ExplODE.")


        if x0 == None:

            raise ValueError("You have to provide an initial value x0 " + \
                "to run the simulation.")


        x0 = np.squeeze(np.asarray(x0))

        if x0.shape[0] != self.pesetup.nx:

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

            if not np.squeeze(psim).shape[0] == self.pesetup.np:

                raise ValueError("Wrong dimension for parameter set psim.")


        fp = ca.MXFunction("fp", \
            [self.pesetup.system.t, self.pesetup.system.u, \
            self.pesetup.system.x, self.pesetup.system.we, \
            self.pesetup.system.wu, self.pesetup.system.p], \
            [self.pesetup.system.f])

        fpeval = fp([\
            self.pesetup.system.t, self.pesetup.system.u, \
            self.pesetup.system.x, np.zeros(self.pesetup.nwe), \
            np.zeros(self.pesetup.nwu), psim])[0]

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


    def compute_covariance_matrix(self):

        r'''
        This function is yet experimental, and will be presented in detail in
        a future version of PECas.
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

            N1 = ca.MX(self.pesetup.Vars.shape[0] - self.W.shape[0], \
                self.W.shape[0])

            N2 = ca.MX(self.pesetup.Vars.shape[0] - self.W.shape[0], \
                self.pesetup.Vars.shape[0] - self.W.shape[0])

            hess = ca.blockcat([[N2, N1], [N1.T, self.W],])

            hess = hess + 1e-10 * ca.diag(self.pesetup.Vars)
            
            J2 = ca.jacobian(self.g, self.pesetup.Vars)

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

            self.beta = self.rhat / (self.yN.size + self.g.size1() - \
                    self.pesetup.Vars.size)

            self.fcovp = ca.MXFunction([self.pesetup.Vars], \
                [self.beta * ca.solve(F11, ca.MX.eye(F11.size1()))])

            self.fcovp.init()
            [self.Covp] = self.fcovp([self.Varshat])

            print( \
'''Covariance matrix computation finished, run show_results() to visualize.''')

        except AttributeError:

            print( \
'''You must execute run_parameter_estimation() first before the covariance
matrix for the estimated parameters can be computed.''')

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
            of the last state :math:`\hat{x}(t_{N})`, and
          - the durations of the setup and the estimation.
        '''

        # - the value of :math:`R^2` measuring the goodness of fit
        #   of the estimated parameters,

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

            # print( \
            #     "\nGoodness of fit R^2" + 30 * "." + ": {0:10.8e}".format(\
            #         self.Rsquared))
            
            # print("\nGoodness of fit R-squared:  ")
            # for i in range(self.pesetup.ny):
            #     print("R^2 - Y_{0} = {1: 10.8e}".format(i,self.Rsquared[i]))

            # print("Residual" + 41 * "." + ": {0:10.8e}".format(self.residual))

            # print("\nResidual:  ")
            # for i in range(self.pesetup.ny):
            #     print("R - Y_{0} = {1: 10.8e}".format(i,self.residual[i]))

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


#     def plot_confidence_ellipsoids(self, indices = []):

#         r'''
#         --- docstring tbd ---
        
#         '''

#         raise NotImplementedError( \
# '''
# This feature of PECas is currently disabled, but will be available again in a
# future version of PECas.
# ''')

#         '''
#         This function plots the confidence ellipsoids pairwise for all
#         parameters defined in ``indices``. The plots are displayed in subplots
#         inside of one plot window. For naming the plots, the variable names
#         defined within the SX/MX-variables that contain the parameters are used.

#         :param indices: List of the indices of the parameters in :math:`x` for
#                         which the confidence ellipsoids shall be plotted.
#                         The indices must be defined by list entries of type
#                         *int*. If an empty list is supported (which is also
#                         the default case),
#                         the ellipsoids for all parameters are plotted.
#         :type indices: list
#         :raises: AttributeError, ValueError, TypeError

#         '''

#         if (self.get_xhat(msg = False) is None) or \
#             (self.get_Covx(msg = False) is None):

#             raise AttributeError('''
# You must execute both run_parameter_estimation() and
# compute_covariance_matrix() before the confidece ellipsoids can be plotted.
# ''')

#         if type(indices) is not list:
#             raise TypeError('''
# The variable containing the indices of the parameters has to be of type list.
# ''')

#         # If the list of indices is empty, create a list that contains
#         # all indices

#         if len(indices) == 0:
#             indices = range(0, self.__d)

#         if len(indices) == 1:
#             raise ValueError('''
# A confidence ellipsoid can not be plotted for only one single parameter. The
# list of indices must therefor contain more than only one entry.
# ''')            

#         for ind in indices:
#             if type(ind) is not int:
#                 raise TypeError('''
# All list entries for the indices have to be of type int.
# ''')

#         nplots = int(round(comb(len(indices), 2)))
#         plotfig = np.figure()
#         plcount = 1

#         xy = np.array([np.cos(np.linspace(0,2*np.pi,100)), \
#                     np.sin(np.linspace(0,2*np.pi,100))])

#         for j, ind1 in enumerate(indices):

#             for k, ind2 in enumerate(indices[j+1:]):

#                 covs = np.array([ \

#                         [self.__Covx[ind1, ind1], self.__Covx[ind1, ind2]], \
#                         [self.__Covx[ind2, ind1], self.__Covx[ind2, ind2]] \

#                     ])

#                 w, v = np.linalg.eig(covs)

#                 ellipse = ca.mul(np.array([self.__xhat[ind1], \
#                     self.__xhat[ind2]]), \
#                     np.ones([1,100])) + ca.mul([v, np.diag(w), xy])

#                 ax = plotfig.add_subplot(nplots, 1, plcount)
#                 ax.plot(np.array(ellipse[0,:]).T, np.array(ellipse[1,:]).T, \
#                     label = str(self.__x[ind1].getName()) + ' - ' + \
#                     str(self.__x[ind2].getName()))
#                 ax.scatter(self.__xhat[ind1], self.__xhat[ind2])
#                 ax.legend(loc="upper left")

#                 plcount += 1

#         np.show()