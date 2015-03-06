#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat

import pylab as pl
from scipy.misc import comb

import systems
import setupmethods

from abc import ABCMeta, abstractmethod

class PECasBaseClass:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pesetup = None, yN = None, stdyN = 1, stds = 1e-2):

        # Store the parameter estimation problem setup

        self.pesetup = pesetup

        # Get the measurement values and standard deviations into the
        # necessary order of apperance and dimensions

        self.yN = pl.zeros(pl.size(yN))
        self.stdyN = pl.zeros(pl.size(yN))

        for k in range(yN.shape[1]):

            self.yN[k:yN.shape[1]*yN.shape[0]+1:yN.shape[1]] = \
                yN[:, k]
            self.stdyN[k:yN.shape[1]*yN.shape[0]+1:yN.shape[1]] = \
                stdyN[:, k]

        self.stds = stds * pl.ones(pesetup.s.shape[0])

        # Set up the covariance matrix for the measurements

        self.CovyNs = pl.square(pl.diag(pl.concatenate((self.stdyN, self.stds))))



class LSq(PECasBaseClass):

    '''The class :class:`LSq` is used to define and solve least
    squares parameter estimation problems with PECas.'''

    def __init__(self, pesetup = None, yN = None, stdyN = 1, stds = 10e-2):

        super(LSq, self).__init__(pesetup = pesetup, yN = yN, stdyN = stdyN, stds = stds)


    def run_parameter_estimation(self):

        r'''
        This functions will run a least sqaures parameter estimation for the
        given problem.

        For this, the least squares parameter estimation problem

        .. math::

            ~ & \hat{x} = \text{arg}\, & \underset{x}{\text{min}}\|M(x)-Y\|_{\Sigma_{\epsilon}^{-1}}^{2}\\
            \text{s. t.}&~&~\\
            ~ & ~ & G = 0\\
            ~ & ~ & H \leq 0\\
            ~ & ~ & x_{0} = x_{init}


        will be set up, and solved using IPOPT. Afterwards,

        - the value of :math:`\hat{x}`
          can be returned using the function :func:`get_xhat()`, and

        - the value of the residual :math:`\hat{R}`
          can be returned using the function :func:`get_Rhat()`.
        '''          

        # Set up the residual function reslsq

        A = ca.mul(pl.linalg.solve(pl.sqrt(self.CovyNs), \
            pl.eye(pl.size(self.yN) + pl.size(self.stds))), \
            ca.vertcat([self.pesetup.phiN - self.yN, self.pesetup.s]))

        reslsq = ca.mul(A.T, A)

        # If equality constraints exists, set then for the solver as well
        # as the cost function

        if not self.pesetup.g.size():

            reslsqfcn = ca.MXFunction(ca.nlpIn(x=self.pesetup.V), \
                ca.nlpOut(f=reslsq))

        else:

            reslsqfcn = ca.MXFunction(ca.nlpIn(x=self.pesetup.V), \
                ca.nlpOut(f=reslsq, g=self.pesetup.g))

        reslsqfcn.init()

        solver = ca.NlpSolver("ipopt", reslsqfcn)
        solver.setOption("tol", 1e-10)
        solver.init()

        # If equality constraints exist, set the bounds for the solver

        if self.pesetup.g.size():

            solver.setInput(pl.zeros(self.pesetup.g.size()), "lbg")
            solver.setInput(pl.zeros(self.pesetup.g.size()), "ubg")

        # Set the initial guess and bounds for the solver

        solver.setInput(self.pesetup.Vinit, "x0")
        solver.setInput(self.pesetup.Vmin, "lbx")
        solver.setInput(self.pesetup.Vmax, "ubx")

        # Run the optimization problem

        solver.evaluate()

        # Store the results of the computation

        self.Vhat = solver.getOutput("x")
        self.rhat = solver.getOutput("f")


    ##########################################################################
    ########## 5. Functions for parameter estimation interpretation ##########
    ##########################################################################

    def compute_covariance_matrix(self):
        
        r'''
        :raises: AttributeError

        This function will compute the covariance matrix
        :math:`\Sigma_{\hat{x}} \in \mathbb{R}^{d\,x\,d}` for the
        estimated parameters :math:`\hat{x}` and the residual
        :math:`\hat{R}`. It can not be used before function
        :func:`run_parameter_estimation()` has been used.


        :math:`\Sigma_{\hat{x}}` is then computed as

        .. math::

            \Sigma_{\hat{x}} = \beta \cdot J^{+}
                \begin{pmatrix} J^{+} \end{pmatrix}^{T}

        with

        .. math::

            \beta = \frac{\hat{R}}{N + m - d}

        and

        .. math::

            J^{+} = \begin{pmatrix} {I} & {0} \end{pmatrix}
                \begin{pmatrix} {J_{1}^{T} J_{1}} & {J_{2}^{T}} \\
                {J_{2}} & {0} \end{pmatrix}^{-1}
                \begin{pmatrix} {J_{1}^{T}} \\ {0} \end{pmatrix}

        while

        .. math::

            J_{1} = \Sigma_{\epsilon}^{\mathbf{^{-1}/_{2}}} \frac{\partial M}{\partial x}

        and

        .. math::

            J_{2} = \frac{\partial G}{\partial x} .

        If the number of equality constraints :math:`m = 0`, computation of
        :math:`J^{+}` simplifies to

        .. math::

            J^{+} = \begin{pmatrix} {J_{1}^{T} J_{1}}\end{pmatrix}^{-1}
                {J_{1}^{T}}

        Afterwards,

          - the value of :math:`\beta` 
            can be returned using the function :func:`get_beta()`, and
          - the matrix :math:`\Sigma_{\hat{x}}`
            can be returned using the function :func:`get_Covx()`.
        '''

        # Compute beta

        if self.get_m() is not None:

            self.__beta = self.__Rhat / (self.__N + self.__m - self.__d)

        else:

            self.__beta = self.__Rhat / (self.__N - self.__d)

        # Compute J1, J2

        Mx = self.__CasADiFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__M))
        Mx.init()

        self.__J1 = ca.mul(ca.solve(pl.sqrt(self.__Sigma_eps), \
            pl.eye(self.__N)), Mx.jac("x", "f"))

        # Compute Jplus and covariance matrix

        if self.get_G(msg = False) is not None:

            Gx = self.__CasADiFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__G))
            Gx.init()

            self.__J2 = Gx.jac("x", "f")

            self.__Jplus = ca.mul([ \

                ca.horzcat((pl.eye(self.__d),pl.zeros((self.__d, self.__m)))),\

                ca.solve(ca.vertcat(( \
                
                    ca.horzcat((ca.mul(self.__J1.T, self.__J1), self.__J2.T)),\
                    ca.horzcat((self.__J2, pl.zeros((self.__m, self.__m)))) \
                
                )), pl.eye(self.__d + self.__m)), \

                ca.vertcat((self.__J1.T, pl.zeros((self.__m, self.__N)))) \

                ])

        else:

            self.__Jplus = ca.mul(ca.solve(ca.mul(self.__J1.T, self.__J1), \
                pl.eye(self.__d)), self.__J1.T)

        self.__fCov = self.__beta * ca.mul([self.__Jplus, self.__Jplus.T])

        # Evaluate covariance matrix for xhat

        self.__fCovx = self.__CasADiFunction(ca.nlpIn(x=self.__x), \
            ca.nlpOut(f=self.__fCov))
        self.__fCovx.init()

        self.__fCovx.setInput(self.__xhat, "x")
        self.__fCovx.evaluate()

        # Store the covariance matrix in Covx

        self.__Covx = self.__fCovx.getOutput("f")


    def print_results(self):

        r'''
        :raises: AttributeError

        This function displays the results of the parameter estimation
        computations. It can not be used before function
        :func:`compute_covariance_matrix()` has been used. The results
        displayed by the function contain

          - the value of :math:`\beta`,
          - the values of the estimated parameters :math:`\hat{x}`
            and their corresponding standard deviations, and
          - the values of the covariance matrix
            :math:`\Sigma_{\hat{x}}` for the
            estimated parameters.
        '''

        if (self.get_xhat(msg = False) is None) or \
            (self.get_Covx(msg = False) is None):

            raise AttributeError('''
You must execute both run_parameter_estimation() and
compute_covariance_matrix() before all results can be displayed.
''')


        print('\n\n## Begin of parameter estimation results ## \n')

        print("Factor beta and residual Rhat:\n")
        print("beta = {0}".format(self.get_beta()))
        print("Rhat = {0}\n".format(self.get_Rhat()))

        print("\nEstimated parameters xi:\n")
        for i, xi in enumerate(self.get_xhat()):

            print("x{0:<3} = {1:10} +/- {2:10}".format(\
                i, xi, pl.sqrt(self.get_Covx()[i, i])))

        print("\n\nCovariance matrix Cov(x):\n")

        print(pl.vectorize("%03.05e".__mod__)(self.get_Covx()))

        print('\n\n##  End of parameter estimation results  ## \n')


    def plot_confidence_ellipsoids(self, indices = []):

        '''
        This function plots the confidence ellipsoids pairwise for all
        parameters defined in ``indices``. The plots are displayed in subplots
        inside of one plot window. For naming the plots, the variable names
        defined within the SX/MX-variables that contain the parameters are used.

        :param indices: List of the indices of the parameters in :math:`x` for
                        which the confidence ellipsoids shall be plotted.
                        The indices must be defined by list entries of type
                        *int*. If an empty list is supported (which is also
                        the default case),
                        the ellipsoids for all parameters are plotted.
        :type indices: list
        :raises: AttributeError, ValueError, TypeError

        '''

        if (self.get_xhat(msg = False) is None) or \
            (self.get_Covx(msg = False) is None):

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
            indices = range(0, self.__d)

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
        plotfig = pl.figure()
        plcount = 1

        xy = pl.array([pl.cos(pl.linspace(0,2*pl.pi,100)), \
                    pl.sin(pl.linspace(0,2*pl.pi,100))])

        for j, ind1 in enumerate(indices):

            for k, ind2 in enumerate(indices[j+1:]):

                covs = pl.array([ \

                        [self.__Covx[ind1, ind1], self.__Covx[ind1, ind2]], \
                        [self.__Covx[ind2, ind1], self.__Covx[ind2, ind2]] \

                    ])

                w, v = pl.linalg.eig(covs)

                ellipse = ca.mul(pl.array([self.__xhat[ind1], \
                    self.__xhat[ind2]]), \
                    pl.ones([1,100])) + ca.mul([v, pl.diag(w), xy])

                ax = plotfig.add_subplot(nplots, 1, plcount)
                ax.plot(pl.array(ellipse[0,:]).T, pl.array(ellipse[1,:]).T, \
                    label = str(self.__x[ind1].getName()) + ' - ' + \
                    str(self.__x[ind2].getName()))
                ax.scatter(self.__xhat[ind1], self.__xhat[ind2])
                ax.legend(loc="upper left")

                plcount += 1

        pl.show()