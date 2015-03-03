import casadi as ca
import casadi.tools as cat
import numpy as np
import pylab as pl
from scipy.misc import comb
import sys

class PECasBaseClass:

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pep = None, Ymeas = None, sigmaY = 1, sigmaS = 10e-4):

        self.pep = pep

        self.Ymeas = pl.zeros(pl.size(Ymeas))
        self.sigmaY = pl.zeros(pl.size(Ymeas))

        for k in Ymeas.shape[1]:

            self.Ymeas[k:Ymeas.shape[1]*Ymeas.shape[0]+1:Ymeas.shape[1]] = \
                Ymeas[k, :]
            self.Ymeas[k:Ymeas.shape[1]*Ymeas.shape[0]+1:Ymeas.shape[1]] = \
                sigmaY[k, :]

        self.sigmaS = [sigmaS] * pep.S.shape[0]

        selfSigma = pl.square(pl.diag(pl.concatenate((sigmaY, sigmaS))))



class LSq(PECasBaseClass):

    '''The class :class:`LSq` is used to define and solve least
    squares parameter estimation problems with PECas.'''

    def __init__(\
        self, x, M, sigma, Y=None, xtrue=None, G=None, H=None, xinit=None, \
        xmin = None, xmax = None):

        '''

        **Mandatory information for constructing the class**

        :param x: Column vector :math:`x \in \mathbb{R}^{d}` for the
                  parameters.
        :type x: casadi.casadi_core.SX/.MX,
                 casadi.tools.structure.ssymStruct/.msymStruct

        :param M: Column vector :math:`M \in \mathbb{R}^{N}` for the model.
        :type M: casadi.casadi_core.SX/.MX

        :param sigma: Column vector :math:`\sigma \in \mathbb{R}^{N}` for
                      the standard deviations.
        :type sigma: numpy.ndarray


        **Mutually substitutable information for constructing the class**

        *One of the two variables Y and xtrue has to be set!*

        :param Y: Column vector :math:`Y \in \mathbb{R}^{N}` for the
                  measurements.
        :type Y: numpy.ndarray

        :param xtrue: Column vector :math:`x_{true} \in \mathbb{R}^{d}` 
                      containing the true value of :math:`x` to
                      generate pseudo measurement data using
                      the vectors :math:`M` and :math:`\sigma`.
        :type xtrue: numpy.ndarray


        **Optional information for constructing the class**

        :param G: Column vector :math:`G \in (0)^{m}` for the
                  equality constraints.
        :type G: casadi.casadi_core.SX/.MX

        :param xinit: Column vector :math:`x_{init} \in \mathbb{R}^{d}` for the initial guess of the parameter values.
        :type xinit: numpy.ndarray, casadi.tools.structure.DMatrixStruct

        :param xmin: Column vector :math:`x_{min} \in \mathbb{R}^{d}` 
                    for the lower bounds of :math:`x`.
        :type xmin: numpy.ndarray, casadi.tools.structure.DMatrixStruct

        :param xmax: Column vector :math:`x_{max} \in \mathbb{R}^{d}` 
                    for the upper bounds of :math:`x`.
        :type xmax: numpy.ndarray, casadi.tools.structure.DMatrixStruct

        |

        '''

        # Parameters

        self.set_x(x)

        # Depending on the type of the parameter variable, determine whether
        # the SX or the MX class is used, and with this, whether SXFunction or
        # MXFunction is used to set up the CasADi functions within the class

        if type(self.__x) is ca.casadi_core.SX or \
            type(self.__x) is cat.structure.ssymStruct:

            self.__CasADiFunction = ca.SXFunction

        else:

            self.__CasADiFunction = ca.MXFunction

        # Model

        self.set_M(M)

        # Standard deviations

        self.set_sigma(sigma)

        # Measurements

        if Y is not None:
            self.set_Y(Y)

        # True value of x

        if xtrue is not None:
            self.set_xtrue(xtrue)

        # Assure that only one of the variables Y and xtrue is provided

        if (Y is None) and (xtrue is None):
            raise ValueError('''
If no measurement data Y is provided, the true value of x must be provided
in xtrue so pseudo measurement data can be created for parameter estimation.
''')

        # Optional information

        # Equality constrains

        if G is not None:
            self.set_G(G)
        else:
            self.__m = 0

        # Initial guess

        if xinit is not None:
            self.set_xinit(xinit)

        # Bounds

        if xmin is not None:
            self.set_xmin(xmin)

        if xmax is not None:
            self.set_xmax(xmax)


    ##########################################################################
    ############# 4. Functions for running parameter estimation ##############
    ##########################################################################

    def generate_pseudo_measurement_data(self):

        r'''
        :raises: AttributeError
        
        This functions generates "random" pseudo measurement data in
        :math:`Y` for a parameter estimation from :math:`M`,
        :math:`\sigma` and :math:`x_{true}`. If measurement data had been
        stored already, it will be overwritten.

        For obtaining :math:`Y`, at first the expression

        .. math::
            Y_{true} = M(x_{true})

        is evaluated, and afterwards, normally distributed measurement noise
        :math:`\epsilon \in \mathbb{R}^{N}` with zeros mean
        and standard deviation :math:`\sigma` is added to the computed values
        of :math:`\hat{Y}`, so that

        .. math::
            Y = Y_{true} + \epsilon,~ \epsilon \sim
                \mathcal{N}(0, \sigma^{2}).


        Afterwards,

          - the vector :math:`Y`
            can be returned using the function :func:`get_Y()`.
        '''

        if self.get_xtrue(msg = False) is None:

            raise AttributeError('''
Pseudo measurement data can only be generated if the true value of x, xtrue,
is known. You can set xtrue manually using the function set_xtrue().
''')

        self.__Mx = self.__CasADiFunction(ca.nlpIn(x=self.__x), \
            ca.nlpOut(f=self.__M))
        self.__Mx.setOption("name", "Model function Mx")
        self.__Mx.init()

        self.__Mx.setInput(self.__xtrue, "x")
        self.__Mx.evaluate()

        self.__Y = self.__Mx.getOutput("f") + \
            np.random.normal(0, self.__sigma, self.__N)


    def run_parameter_estimation(self):

        r'''
        This functions will run the parameter estimation for the given problem.        
        If measurement data is not yet existing, it will be generated using
        the function :func:`generate_pseudo_measurement_data()`.

        Then, the parameter estimation problem

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

        # First, check if measurement data exists; if not, generate it

        if self.get_Y(msg = False) is None:

            self.generate_pseudo_measurement_data()            

        # Set up the cost function f

        A = ca.mul(np.linalg.solve(np.sqrt(self.__Sigma_eps), np.eye(self.__N)), \
            (self.__M - self.__Y))
        self.__f = ca.mul(A.T, A)

        # Solve the minimization problem for f

        if self.get_G(msg = False) is None:

            self.__fx = self.__CasADiFunction(ca.nlpIn(x=self.__x), \
                ca.nlpOut(f=self.__f))

        else:

            self.__fx = self.__CasADiFunction(ca.nlpIn(x=self.__x), \
                ca.nlpOut(f=self.__f, g=self.__G))

        self.__fx.init()

        solver = ca.NlpSolver("ipopt", self.__fx)
        solver.setOption("tol", 1e-10)
        solver.init()

        # If equality constraints exist, set the bounds for the solver

        if self.get_m() is not 0:

            solver.setInput(np.zeros(self.__m), "lbg")
            solver.setInput(np.zeros(self.__m), "ubg")

        # If an initial guess was given, set the initial guess for the solver
        
        if self.get_xinit(msg = False) is not None:

            solver.setInput(self.__xinit, "x0")

        # If given, set the bounds for the parameter values

        if self.get_xmin(msg = None) is not None:

            solver.setInput(self.__xmin, "lbx")

        if self.get_xmax(msg = None) is not None:

            solver.setInput(self.__xmax, "ubx")

        # Run the optimization problem

        solver.evaluate()

        # Store the results of the computation

        self.__xhat = solver.getOutput("x")
        self.__Rhat = solver.getOutput("f")


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

        if self.get_xhat(msg = False) is None:

            raise AttributeError('''
Execute run_parameter_estimation() before computing the covariance matrix.
''')

        # Compute beta

        if self.get_m() is not None:

            self.__beta = self.__Rhat / (self.__N + self.__m - self.__d)

        else:

            self.__beta = self.__Rhat / (self.__N - self.__d)

        # Compute J1, J2

        Mx = self.__CasADiFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__M))
        Mx.init()

        self.__J1 = ca.mul(ca.solve(np.sqrt(self.__Sigma_eps), \
            np.eye(self.__N)), Mx.jac("x", "f"))

        # Compute Jplus and covariance matrix

        if self.get_G(msg = False) is not None:

            Gx = self.__CasADiFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__G))
            Gx.init()

            self.__J2 = Gx.jac("x", "f")

            self.__Jplus = ca.mul([ \

                ca.horzcat((np.eye(self.__d),np.zeros((self.__d, self.__m)))),\

                ca.solve(ca.vertcat(( \
                
                    ca.horzcat((ca.mul(self.__J1.T, self.__J1), self.__J2.T)),\
                    ca.horzcat((self.__J2, np.zeros((self.__m, self.__m)))) \
                
                )), np.eye(self.__d + self.__m)), \

                ca.vertcat((self.__J1.T, np.zeros((self.__m, self.__N)))) \

                ])

        else:

            self.__Jplus = ca.mul(ca.solve(ca.mul(self.__J1.T, self.__J1), \
                np.eye(self.__d)), self.__J1.T)

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
                i, xi, np.sqrt(self.get_Covx()[i, i])))

        print("\n\nCovariance matrix Cov(x):\n")

        print(np.vectorize("%03.05e".__mod__)(self.get_Covx()))

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

                covs = np.array([ \

                        [self.__Covx[ind1, ind1], self.__Covx[ind1, ind2]], \
                        [self.__Covx[ind2, ind1], self.__Covx[ind2, ind2]] \

                    ])

                w, v = pl.linalg.eig(covs)

                ellipse = ca.mul(np.array([self.__xhat[ind1], \
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
