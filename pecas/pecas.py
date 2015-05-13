#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat

import pylab as pl
from scipy.misc import comb

import systems
import setups

import time

import pdb

from abc import ABCMeta, abstractmethod

class PECasBaseClass:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pesetup = None, yN = None, wv = None, ww = None):

        # Store the parameter estimation problem setup

        self.pesetup = pesetup

        # Check if the supported measurement data fits to the dimensions of
        # the output function

        yN = pl.atleast_2d(yN)

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

        wv = pl.atleast_2d(wv)

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

        self.yN = pl.zeros(pl.size(yN))
        self.wv = pl.zeros(pl.size(yN))

        for k in range(yN.shape[0]):

            self.yN[k:yN.shape[0]*yN.shape[1]+1:yN.shape[0]] = \
                yN[k, :]
            self.wv[k:yN.shape[0]*yN.shape[1]+1:yN.shape[0]] = \
                wv[k, :]


        self.ww = []

        try:

            if self.pesetup.nw != 0:

                ww = pl.atleast_2d(ww)

                try:

                    if ww.shape == (1, self.pesetup.nw):

                        ww = ww.T

                    if not ww.shape == (self.pesetup.nw, 1):

                        raise ValueError('''
The dimensions of the weights of the equation errors given in ww does not
match the dimensions of the differential equations.''')

                    self.ww = ww

                except AttributeError:

                    pass

                try:

                    # if self.ww is not None:

                        self.ww = pl.squeeze(ca.repmat(ww, self.pesetup.nsteps * \
                            (len(self.pesetup.tauroot)-1), 1))

                except AttributeError:

                    self.ww = []

        except AttributeError:

            pass

        # Set up the covariance matrix for the measurements

        self.W = pl.diag(pl.concatenate((self.wv, self.ww)))

        self.EstimationDone = False
        self.CovCal = False

    @property
    def phat(self):

        try:
            
            return self.pesetup.Vars()(self.Varshat)["P"]

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

                xhat.append(self.pesetup.Vars()(self.Varshat)["X",:,0,i])
            
            return xhat

        except AttributeError:

            raise AttributeError('''
The method run_parameter_estimation() must be run first, before trying to
obtain the optimal values.
''')     


class LSq(PECasBaseClass):

    '''The class :class:`LSq` is used to solve least squares parameter
    estimation problems with PECas for a given set of measurement data.'''

    def __init__(self, pesetup = None, yN = None, wv = None, ww = None):

        super(LSq, self).__init__(pesetup = pesetup, yN = yN, wv = wv, ww = ww)


    def run_parameter_estimation(self):

        r'''
        This functions will run a least sqaures parameter estimation for the
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

        self.tstart = time.time()

        A = []

        for k, elem in enumerate(self.pesetup.Vars["V"]):
        
            A.append(elem)

        g = ca.vertcat([self.pesetup.phiN - self.yN + ca.vertcat(A)])


        if "W" in self.pesetup.Vars.keys():


            W = []

            for k, elem in enumerate(self.pesetup.Vars["W"]):

                W.append(elem)

            A = A + sum(W, [])


        A = ca.vertcat(A)

        reslsq = ca.mul([A.T, self.W, A])


        if self.pesetup.g.size():

            g = ca.vertcat([g, self.pesetup.g])

        reslsqfcn = ca.SXFunction(ca.nlpIn(x=self.pesetup.Vars), \
            ca.nlpOut(f=reslsq, g=g))

        reslsqfcn.init()

        # Initialize the solver

        solver = ca.NlpSolver("ipopt", reslsqfcn)
        solver.setOption("tol", 1e-10)
        solver.init()

        # Set equality constraints

        solver.setInput(pl.zeros(g.size()), "lbg")
        solver.setInput(pl.zeros(g.size()), "ubg")

        # Set the initial guess and bounds for the solver]

        solver.setInput(self.pesetup.Varsinit, "x0")
        solver.setInput(self.pesetup.Varsmin, "lbx")
        solver.setInput(self.pesetup.Varsmax, "ubx")

        # Solve the optimization problem

        solver.evaluate()

        # Store the results of the computation

        self.Varshat = solver.getOutput("x")
        self.rhat = solver.getOutput("f")
        
        self.EstimationDone = True
        
        Ysim = self.pesetup.phiNfcn([self.Varshat])[0]
        Ym = pl.reshape(self.yN.T,(Ysim.shape))
        res = Ym-Ysim
        self.residual = (pl.norm(res))**2

        self.est_duration = time.time() - self.tstart

        self.Rsquared = 1 - self.residual/(pl.norm(Ym))**2
        
    @property        
    def Phat(self):       
        if self.EstimationDone == False:

            raise AttributeError('''
            The method run_parameter_estimation
            must be run first, before trying to
            obtain the optimal values''')

        return pl.array(self.pesetup.Vars()(self.Varshat)["P"])

    @property    
    def Xhat(self):
        
        if self.EstimationDone == False:

            raise AttributeError('''
            The method run_parameter_estimation
            must be run first, before trying to
            obtain the optimal values''')
        
        xhat = []

        for i in range (self.pesetup.nx):
            xhat.append(self.pesetup.Vars()(self.Varshat)["X",:,0,i])
            
        return pl.array(xhat)

    def compute_covariance_matrix(self):

        raise NotImplementedError( \
'''
This feature of PECas is currently disabled, but will be available again in a
future version of PECas.
''')


        self.CovCal = True
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

        # # Compute J1, J2

        # Mx = self.__CasADiFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__M))
        # Mx.init()

        # self.__J1 = ca.mul(ca.solve(pl.sqrt(self.__Sigma_eps), \
        #     pl.eye(self.__N)), Mx.jac("x", "f"))

        # # Compute Jplus and covariance matrix

        # if self.get_G(msg = False) is not None:

        #     Gx = self.__CasADiFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__G))
        #     Gx.init()

        #     self.__J2 = Gx.jac("x", "f")

        #     self.__Jplus = ca.mul([ \

        #         ca.horzcat((pl.eye(self.__d),pl.zeros((self.__d, self.__m)))),\

        #         ca.solve(ca.vertcat(( \
                
        #             ca.horzcat((ca.mul(self.__J1.T, self.__J1), self.__J2.T)),\
        #             ca.horzcat((self.__J2, pl.zeros((self.__m, self.__m)))) \
                
        #         )), pl.eye(self.__d + self.__m)), \

        #         ca.vertcat((self.__J1.T, pl.zeros((self.__m, self.__N)))) \

        #         ])

        # else:

        #     self.__Jplus = ca.mul(ca.solve(ca.mul(self.__J1.T, self.__J1), \
        #         pl.eye(self.__d)), self.__J1.T)

        # self.__fCov = self.__beta * ca.mul([self.__Jplus, self.__Jplus.T])

        # # Evaluate covariance matrix for xhat

        # self.__fCovx = self.__CasADiFunction(ca.nlpIn(x=self.__x), \
        #     ca.nlpOut(f=self.__fCov))
        # self.__fCovx.init()

        # self.__fCovx.setInput(self.__xhat, "x")
        # self.__fCovx.evaluate()

        # # Store the covariance matrix in Covx

        # self.__Covx = self.__fCovx.getOutput("f")


    def print_results(self):

         r'''
         :raises: AttributeError

         This function displays the results of the parameter estimation
         computations. It can not be used before function
         :func:`run_parameter_estimation()` has been used. The results
         displayed by the function contain

           - the value of :math:`R^2` measuring the goodness of fit
             of the estimated parameters,
           - the values of the estimated parameters :math:`\hat{p}`
             and their corresponding standard deviations
             (the value of the standar deviation is represented
             only if the covariance matrix is computed)
           - the values of the covariance matrix
             :math:`\Sigma_{\hat{x}}` for the
             estimated parameters (if is computed),
           - in the case of the estimation of a dynamic
             system, the optimal value of the first state 
             :math:`\Sigma_{\hat{X0}}` and the optimal value 
             of the last state :math:`\Sigma_{\hat{XN}}`
           - the running time of the estimation 
         '''

         if (self.EstimationDone == False):
             raise AttributeError('''
 You must execute both run_parameter_estimation() and
 compute_covariance_matrix() before all results can be displayed.
 ''')

         if self.CovCal == False:
            print('\n\n## Begin of parameter estimation results ##')
             
            print("\nEstimated parameters pi:")
            for i, xi in enumerate(self.Phat):
            
                print("p{0:<3} = {1:10}".format(\
                     i, xi[0]))
            
            print("\nEstimated initial value x0:  ")
            print self.Xhat[:,0]             
            
            print("\nEstimated final value xN:  ")
            print self.Xhat[:,-1]  
                     
            print("\nGoodness of fit R-squared:  ")
            print("R^2 = {0}".format(self.Rsquared))

            print("\nResidual:  ")
            print("Residual = {0}".format(self.residual))


            print("\nDuration of the problem setup:  ")
            print self.pesetup.setupDura
            
            print("\nDuration of the simulation:  ")
            print self.est_duration
             
            print('\n\n##  End of parameter estimation results  ## \n')


#         print("\nEstimated parameters xi:\n")
#         for i, xi in enumerate(self.get_xhat()):

#             print("x{0:<3} = {1:10} +/- {2:10}".format(\
#                 i, xi, pl.sqrt(self.get_Covx()[i, i])))

#         print("\n\nCovariance matrix Cov(x):\n")

#         print(pl.vectorize("%03.05e".__mod__)(self.get_Covx()))

#         print('\n\n##  End of parameter estimation results  ## \n')


    def show_system_information(self, showEquations = False):
        
        print('\n\n## Begin of system information ##')

        if isinstance(self.pesetup.system, systems.BasicSystem):
            
            print("""\The system is a non-dynamic systems with the general 
input-output structure and contrain equations: """)
            
            print("Phi = y(t, u, p), g(t, u, p)=0 ")
            
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

            print("""\nThe system is a dynamic defined by a set of explicit ODEs 
xdot which stablish the system state x:
    xdot = f(t, u, x, p, w) )
and by an output function y which sets the system measurements:
    Phi = y(t, x, p)
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

        print('\n\n##  End of system information  ## \n')

    def plot_confidence_ellipsoids(self, indices = []):

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
#         plotfig = pl.figure()
#         plcount = 1

#         xy = pl.array([pl.cos(pl.linspace(0,2*pl.pi,100)), \
#                     pl.sin(pl.linspace(0,2*pl.pi,100))])

#         for j, ind1 in enumerate(indices):

#             for k, ind2 in enumerate(indices[j+1:]):

#                 covs = pl.array([ \

#                         [self.__Covx[ind1, ind1], self.__Covx[ind1, ind2]], \
#                         [self.__Covx[ind2, ind1], self.__Covx[ind2, ind2]] \

#                     ])

#                 w, v = pl.linalg.eig(covs)

#                 ellipse = ca.mul(pl.array([self.__xhat[ind1], \
#                     self.__xhat[ind2]]), \
#                     pl.ones([1,100])) + ca.mul([v, pl.diag(w), xy])

#                 ax = plotfig.add_subplot(nplots, 1, plcount)
#                 ax.plot(pl.array(ellipse[0,:]).T, pl.array(ellipse[1,:]).T, \
#                     label = str(self.__x[ind1].getName()) + ' - ' + \
#                     str(self.__x[ind2].getName()))
#                 ax.scatter(self.__xhat[ind1], self.__xhat[ind2])
#                 ax.legend(loc="upper left")

#                 plcount += 1

#         pl.show()
