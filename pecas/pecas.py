##!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat

import numpy as np
# from scipy.misc import comb

import pdb
import time

import systems
import setups
import intro

from abc import ABCMeta, abstractmethod

class PECasBaseClass:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pesetup = None, yN = None, wv = None, wwe = None,\
    wwu = None):

        intro.pecas_intro()
        print('\n' + 22 * '-' + \
            ' PECas parameter estimation setup ' + 22 * '-')
        print('\nStarting parameter estimation problem setup ...') 

        # Store the parameter estimation problem setup

        self.pesetup = pesetup

        # Check if the supported measurement data fits to the dimensions of
        # the output function

        yN = np.atleast_2d(yN)

        if yN.shape == (pesetup.timegrid.size, pesetup.ny):

            yN = yN.T

        if not yN.shape == (pesetup.ny, pesetup.timegrid.size):

            raise ValueError('''
The dimension of the measurement data given in yN does not match the
dimension of output function and/or timegrid.
Valid dimensions for yN for the given data are:
    {0} or {1},
but you supported yN of dimension:
    {2}.'''.format(str((pesetup.timegrid.size, pesetup.ny)), \
    str((pesetup.ny, pesetup.timegrid.size)), str(yN.shape)))

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

    '''The class :class:`LSq` is used to solve least squares parameter
    estimation problems for previously defined systems using a given set
    of measurement data and weightings.'''

    def __init__(self, pesetup = None, yN = None, wv = None, wwe = None,\
    wwu = None):

        super(LSq, self).__init__(pesetup = pesetup, yN = yN, wv = wv, \
        wwe = wwe, wwu = wwu)


    def run_parameter_estimation(self):

        r'''
        This functions will run a least squares parameter estimation for the
        given problem and data set.
        For this, the least squares parameter estimation problem

        .. math::
            ~ & \hat{x} = \text{arg}\, & \underset{x}{\text{min}}\|M(x)-Y\|_{\Sigma_{\epsilon}^{-1}}^{2}\\
            \text{s. t.}&~&~\\
            ~ & ~ & G = 0\\
            ~ & ~ & x_{0} = x_{init}

        will be set up, and solved using IPOPT. Afterwards,

        - the value of :math:`\hat{x}`
          can be returned using the function :func:`get_xhat()`, and
        - the value of the residual :math:`\hat{R}`
          can be returned using the function :func:`get_Rhat()`.
        '''          

        intro.pecas_intro()
        print('\n' + 18 * '-' + \
            ' PECas least squares parameter estimation ' + 18 * '-')

        print('''
Starting least squares parameter estimation using IPOPT, 
this might take some time ...
''')

        self.tstart_estimation = time.time()

        A = []

        for k, elem in enumerate(self.pesetup.Vars["V"]):
        
            A.append(elem)

        g = ca.vertcat([self.pesetup.phiN - self.yN + ca.vertcat(A)])


        if "WE" in self.pesetup.Vars.keys():

            W = []

            for k, elem in enumerate(self.pesetup.Vars["WE"]):

                W.append(elem)

            A = A + sum(W, [])


        if "WU" in self.pesetup.Vars.keys():

            W = []

            for k, elem in enumerate(self.pesetup.Vars["WU"]):

                W.append(elem)

            A = A + sum(W, [])


        A = ca.vertcat(A)

        self.reslsq = ca.mul([A.T, self.W, A])

        self.A = A


        if self.pesetup.g.size():

            g = ca.vertcat([g, self.pesetup.g])

        self.g = g


        reslsqfcn = ca.MXFunction(ca.nlpIn(x=self.pesetup.Vars), \
            ca.nlpOut(f=self.reslsq, g=g))

        reslsqfcn.init()

        # Initialize the solver

        solver = ca.NlpSolver("ipopt", reslsqfcn)
        solver.setOption("tol", 1e-10)
        solver.setOption("linear_solver", "ma97")
        # solver.setOption("expand", True)
        solver.init()

        # Set equality constraints

        solver.setInput(np.zeros(g.size()), "lbg")
        solver.setInput(np.zeros(g.size()), "ubg")

        # Set the initial guess and bounds for the solver]

        solver.setInput(self.pesetup.Varsinit, "x0")
        solver.setInput(self.pesetup.Varsmin, "lbx")
        solver.setInput(self.pesetup.Varsmax, "ubx")

        # Solve the optimization problem

        solver.evaluate()

        # Store the results of the computation

        self.Varshat = solver.getOutput("x")
        self.rhat = solver.getOutput("f")
        self.dv_lambdahat = solver.getOutput("lam_g")
        
        Ysim = self.pesetup.phiNfcn([self.Varshat])[0]
        Ym = np.reshape(self.yN.T,(Ysim.shape))
        res = Ym-Ysim
        self.residual = []
        self.Rsquared = []

        for i in range(self.pesetup.ny):   

            self.residual.append((np.linalg.norm(res[i:-1:self.pesetup.ny]))**2)
            self.Rsquared.append(1 - self.residual[i]/(np.linalg.norm(Ym[i:-1:self.pesetup.ny]))**2)
        
        self.tend_estimation = time.time()
        self.duration_estimation = self.tend_estimation - \
            self.tstart_estimation

        print('''
Parameter estimation finished. Check IPOPT output for status information.''')


    def compute_covariance_matrix(self):

        r'''
        --- docstring tbd ---
        
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

            self.beta = self.rhat / (self.yN.size + self.g.size1() - \
                    self.pesetup.Vars.size)

            dv_lambda = ca.MX.sym("dv_lambda", self.g.size1())

            L = self.reslsq + ca.mul((dv_lambda.T, self.g))

            kkt = ca.blockcat( \

                [[ca.hessian(L, self.pesetup.Vars), \
                    ca.jacobian(self.g, self.pesetup.Vars).T], \

                [ca.jacobian(self.g, self.pesetup.Vars), \
                    ca.MX(self.g.size1(), self.g.size1())]] \

                    )

            B1 = kkt[:self.pesetup.np, :self.pesetup.np]
            E = kkt[self.pesetup.np:, :self.pesetup.np]
            # ET = kkt[:self.pesetup.np, self.pesetup.np:]
            D = kkt[self.pesetup.np:, self.pesetup.np:]

            # Dinv = ca.solve(D, ca.MX.eye(D.size1()), "csparse")
            # F11 = B1 - ca.mul([E.T, Dinv, E])

            Dinv = ca.solve(D, E, "csparse")

            F11 = B1 - ca.mul([E.T, Dinv])

            self.fcovp = ca.MXFunction([self.pesetup.Vars, dv_lambda], \
                [self.beta * ca.solve(F11, ca.MX.eye(F11.size1()))])

            self.fcovp.init()
            [self.Covp] = self.fcovp([self.Varshat, self.dv_lambdahat])

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


        # r'''
        # :raises: AttributeError

        # This function will compute the covariance matrix
        # :math:`\Sigma_{\hat{x}} \in \mathbb{R}^{d\,x\,d}` for the
        # estimated parameters :math:`\hat{x}` and the residual
        # :math:`\hat{R}`. It can not be used before function
        # :func:`run_parameter_estimation()` has been used.


        # :math:`\Sigma_{\hat{x}}` is then computed as

        # .. math::

        #     \Sigma_{\hat{x}} = \beta \cdot J^{+}
        #         \begin{pmatrix} J^{+} \end{pmatrix}^{T}

        # with

        # .. math::

        #     \beta = \frac{\hat{R}}{N + m - d}

        # and

        # .. math::

        #     J^{+} = \begin{pmatrix} {I} & {0} \end{pmatrix}
        #         \begin{pmatrix} {J_{1}^{T} J_{1}} & {J_{2}^{T}} \\
        #         {J_{2}} & {0} \end{pmatrix}^{-1}
        #         \begin{pmatrix} {J_{1}^{T}} \\ {0} \end{pmatrix}

        # while

        # .. math::

        #     J_{1} = \Sigma_{\epsilon}^{\mathbf{^{-1}/_{2}}} \frac{\partial M}{\partial x}

        # and

        # .. math::

        #     J_{2} = \frac{\partial G}{\partial x} .

        # If the number of equality constraints :math:`m = 0`, computation of
        # :math:`J^{+}` simplifies to

        # .. math::

        #     J^{+} = \begin{pmatrix} {J_{1}^{T} J_{1}}\end{pmatrix}^{-1}
        #         {J_{1}^{T}}

        # Afterwards,

        #   - the value of :math:`\beta` 
        #     can be returned using the function :func:`get_beta()`, and
        #   - the matrix :math:`\Sigma_{\hat{x}}`
        #     can be returned using the function :func:`get_Covx()`.
        # '''

        # # Compute beta

        # if self.get_m() is not None:

        #     self.__beta = self.__Rhat / (self.__N + self.__m - self.__d)

        # else:

        #     self.__beta = self.__Rhat / (self.__N - self.__d)

        # ... 

        # self.__fCov = self.__beta * ca.mul([self.__Jplus, self.__Jplus.T])

        # ...


    def show_results(self):

        r'''
        :raises: AttributeError

        This function displays the results of the parameter estimation
        computations. It can not be used before function
        :func:`run_parameter_estimation()` has been used. The results
        displayed by the function contain:
        
          - the value of :math:`R^2` measuring the goodness of fit
            of the estimated parameters,
          - the values of the estimated parameters :math:`\hat{p}`
            and their corresponding standard deviations
            (the value of the standar deviation is represented
            only if the covariance matrix is computed),
          - the values of the covariance matrix
            :math:`\Sigma_{\hat{x}}` for the
            estimated parameters (if is computed),
          - in the case of the estimation of a dynamic
            system, the optimal value of the first state 
            :math:`\hat{x}(t_{0})` and the optimal value 
            of the last state :math:`\hat{x}(t_{N})`, and
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
            
                print("    p_{0:<3} = {1: 10.8e} +/- {2: 10.8e}".format(\
                     i, xi[0], ca.sqrt(abs(self.Covp[i, i]))))
            
            try:

                print("\nEstimated initial states value:  ")
                print("    x(t_0) = {0}".format(self.Xhat[:,0]))
                
                print("\nEstimated final states value:  ")
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
            
            print("\nGoodness of fit R-squared:  ")
            for i in range(self.pesetup.ny):
                print("R^2 - Y_{0} = {1: 10.8e}".format(i,self.Rsquared[i]))

            # print("Residual" + 41 * "." + ": {0:10.8e}".format(self.residual))

            print("\nResidual:  ")
            for i in range(self.pesetup.ny):
                print("R - Y_{0} = {1: 10.8e}".format(i,self.residual[i]))

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


    def show_system_information(self, showEquations = False):

        r'''
        --- docstring tbd ---
        '''
        
        intro.pecas_intro()

        print('\n' + 26 * '-' + \
            ' PECas system information ' + 26 * '-')

        if isinstance(self.pesetup.system, systems.BasicSystem):
            
            print("""\The system is a non-dynamic systems with the general 
input-output structure and contrain equations: """)
            
            print("phi = y(t, u, p), g(t, u, p) = 0 ")
            
            print("""\nWith {0} inputs u, {1} parameters p and {2} outputs y
            """.format(self.pesetup.nu,self.pesetup.np,self.pesetup.ny))


            if showEquations:
                
                print("\nAnd where Phi is defined by: ")
                for i, yi in enumerate(self.pesetup.system.fcn['y']):         
                    print("y[{0}] = {1}".format(\
                         i, yi))
                         
                print("\nAnd where g is defined by: ")
                for i, gi in enumerate(self.pesetup.system.fcn['g']):              
                    print("g[{0}] = {1}".format(\
                         i, gi))

        elif isinstance(self.pesetup.system, systems.ExplODE):

            print("""\nThe system is a dynamic system defined by a set of
explicit ODEs xdot which establish the system state x:
    xdot = f(t, u, x, p, w)
and by an output function y which sets the system measurements:
    phi = y(t, x, p).
""")
            
            
            print("""Particularly, the system has:
    {0} inputs u
    {1} parameters p
    {2} states x
    {3} outputs y""".format(self.pesetup.nu,self.pesetup.np,\
                                self.pesetup.nx, self.pesetup.ny))

            if showEquations:
                
                print("\nWhere xdot is defined by: ")
                for i, xi in enumerate(self.pesetup.system.fcn['f']):         
                    print("xdot[{0}] = {1}".format(\
                         i, xi))
                         
                print("\nAnd where y is defined by: ")
                for i, yi in enumerate(self.pesetup.system.fcn['y']):              
                    print("y[{0}] = {1}".format(\
                         i, yi))   
        else:
            raise NotImplementedError('''
This feature of PECas is currently disabled, but will be 
available when the DAE systems are implemented.
''')


    def plot_confidence_ellipsoids(self, indices = []):

        r'''
        --- docstring tbd ---
        
        '''

        raise NotImplementedError( \
'''
This feature of PECas is currently disabled, but will be available again in a
future version of PECas.
''')

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