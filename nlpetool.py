import casadi as ca
import numpy as np

class NLPETool:

    '''Here to be the docstring for the class.'''

    ##########################################################################
    #### 1. Functions for checking validity and consistency of the inputs ####
    ##########################################################################

    def __check_variable_validity(self, var, varname, dtype, ddim):

        '''
        :param var: Variable that's validity shall be checked.
        :type var: dtype
        :param varname: Name of the variable that shall be checked.
        :type varname: str
        :param dtype: Description of the type that var has to be.
        :type dtype: ca.casadi_core.MX, numpy.ndarray
        :param ddim: Description of the second dimension var has to have.
        :param ddim: int
        :raises: IndexError, ValueError
        :catches: IndexError

        Check the validity of an input variable by checking the type of the
        varriable and it's shape properties.
        '''

        # Check variable type

        if type(var) is not dtype:
            raise ValueError(\
                '"{0}" needs to be of type "{1}".'.format(varname, dtype))

        # Check if the variable contains a (column) vector

        try:
            if var.shape[1] is not ddim:
                raise ValueError(\
                    '"{0}" needs to be a (column) vector.'.format(varname))
        except IndexError:

            # If there is no second dimension given, the variable is also
            # a column vector

            pass
    
    # -----------------------------------------------------------------------#


    def __check_variable_consistency(self, varname1, ddim1, varname2, ddim2):

        '''
        :param varname1: The name of the first variable that shall be compared.
        :type varname1: str
        :param ddim1: The dimension for the first variable to be compared.
        :type ddim1: int
        :param varname2: The name of the second variable that shall be
                         compared.
        :type varname2: str
        :param ddim2: The dimension for the second variable to be compared.
        :type ddim2: int

        Check the consistency for two input variables by comparing the
        relevant dimensions of the variables provided with the function call.
        '''

        if ddim1 != ddim2:
            raise ValueError('''
The dimensions of the variables "{0}" and "{2}" do not match, since
"{0}" has {1} entries, while "{2}"" has {4} entries.'''.format(\
                varname1, ddim1, varname2, ddim2))


    ##########################################################################
    #### 2. Functions for getting and setting input and output variables #####
    ##########################################################################

    def set_x(self, x):

        '''
        :param x: Column vector :math:`x \in \mathbb{R}^{d}` for the
                  parameters.
        :type x: casadi.casadi_core.MX
        :raises: ValueError

        *This function is called automatically at the initialization of the
        object.*

        Set the column vector :math:`x` for the parameters.
        '''

        self.__check_variable_validity(x, "x", ca.casadi_core.MX, 1)
        self.__x = x


    def get_x(self):

        '''
        :returns: casadi.casadi_core.MX - the column vector
                  :math:`x \in \mathbb{R}^{d}` for the parameters.

        Get the column vector :math:`x` for the parameters.
        '''

        return self.__x

    # -----------------------------------------------------------------------#


    def set_M(self, M):

        '''
        :param M: Column vector :math:`M \in \mathbb{R}^{N}` for the
                  model.
        :type M: casadi.casadi_core.MX
        :raises: AttributeError, ValueError
        :catches: AttributeError

        *This function is called automatically at the initialization of the
        object.*

        Set the column vector :math:`M` for the model. If the
        dimensions of :math:`\sigma` and :math:`M` are not consistent,
        an exception will be raised.
        '''

        self.__check_variable_validity(M, "M", ca.casadi_core.MX, 1)

        try:

            self.__check_variable_consistency("M", self.__M.shape[0], \
            "sigma", self.__sigma.shape[0])

        # If a variable in comparison has not been set up so far, an
        # AttributeError exception will be thrown

        except AttributeError:

            pass

        self.__M = M


    def get_M(self):

        '''
        :returns: casadi.casadi_core.MX - the column vector
                  :math:`M  \in \mathbb{R}^{N}` for the model.

        Get the column vector :math:`M` for the model.
        '''

        return self.__M

    # -----------------------------------------------------------------------#


    def set_sigma(self, sigma):

        '''
        :param sigma: Column vector :math:`\sigma \in \mathbb{R}^{N}` for the
                  standard deviations.
        :type sigma: numpy.ndarray
        :raises: AttributeError, ValueError
        :catches: AttributeError

        *This function is called automatically at the initialization of the
        object.*

        Set the column vector :math:`\sigma` for the standard deviation. If the
        dimensions of :math:`\sigma` and :math:`M` or :math:`\sigma` and 
        :math:`Y` are not consistent, an exception will be raised.

        Also set up the covariance matrix :math:`\Sigma_{\epsilon}` of the
        error using :math:`\sigma` as its diagonal entries, i. e.
        :math:`\sigma = diag(\Sigma_{\epsilon})`.
        '''

        self.__check_variable_validity(sigma, "sigma", np.ndarray, 1)

        try:

            self.__check_variable_consistency("M", self.__M.shape[0], \
            "sigma", self.__sigma.shape[0])

            self.__check_variable_consistency("Y", self.__Y.shape[0], \
                "sigma", self.__sigma.shape[0])

        # If a variable in comparison has not been set up so far, an
        # AttributeError exception will be thrown

        except AttributeError:

            pass

        self.__sigma = sigma

        # Set up the covariance matrix of the error

        self.__Sigma = np.diag(sigma)


    def get_sigma(self):

        '''
        :returns: numpy.ndarray - the column vector
                  :math:`\sigma  \in \mathbb{R}^{N}` for the
                  standard deviations.

        Get the column vector :math:`\sigma` for the standard deviations.
        '''

        return self.__sigma


    def get_Sigma(self):

        '''
        :returns: numpy.ndarray - the covariance matrix
                  of the error
                  :math:`\Sigma_{\epsilon} \in \mathbb{R}^{N\,x\,N}`.

        Get the covariance matrix of the error :math:`\Sigma_{\epsilon}.`
        '''

        return self.__Sigma

    # -----------------------------------------------------------------------#


    def set_Y(self, Y):

        '''
        :param Y: Column vector :math:`Y \in \mathbb{R}^{N}` for the
                  measurements.
        :type Y: numpy.ndarray
        :raises: AttributeError, ValueError
        :catches: AttributeError

        *If data is provided, this function is called automatically at the
        initialization of the object.*

        Set the column vector :math:`Y` for the measurements. If the
        dimensions of :math:`\sigma` and :math:`Y` are not consistent,
        an exception will be raised.

        For generation of "random" pseudo measurement data, see
        :func:`generate_pseudo_measurement_data`.
        '''

        self.__check_variable_validity(Y, "Y", np.ndarray, 1)

        try:

            self.__check_variable_consistency("Y", self.__Y.shape[0], \
                "sigma", self.__sigma.shape[0])

        # If a variable in comparison has not been set up so far, an
        # AttributeError exception will be thrown

        except AttributeError:

            pass

        self.__Y = Y


    def get_Y(self):

        '''
        :returns: numpy.ndarray - the column vector
                  :math:`Y  \in \mathbb{R}^{N}` for the measurements.
        :raises: AttributeError
        :catches: AttributeError

        Get the column vector :math:`Y` for the measurements. If no data
        has been provided so far either by manual input or via
        :func:`generate_pseudo_measurement_data`, the function will raise and
        catch an exception and display possible solutions to the user.
        '''
        try:
            return self.__Y
        except AttributeError:
            print('''
No data for Y has been provided so far. Try set_Y() for manual setting, or
generate_pseudo_measurement_data() for "random" pseudo measurement data.
''')

    # -----------------------------------------------------------------------#


    def set_xtrue(self, xtrue):

        '''
        :param xtrue: Column vector :math:`x_{true} \in \mathbb{R}^{d}` 
                      containing the true value of :math:`x`.
        :type xtrue: numpy.ndarray
        :raises: ValueError

        *If data is provided, this function is called automatically at the
        initialization of the object.*

        Set the column vector :math:`x_{true}` for the true values of :math:`x`
        that can be used for generating "random" pseudo measurement data, see
        :func:`generate_pseudo_measurement_data`.
        '''

        self.self.__check_variable_validity(xtrue, "xtrue", np.ndarray, 1)

        self.__xtrue = xtrue


    def get_xtrue(self):

        '''
        :returns: numpy.ndarray - the column vector
                  :math:`x_{true} \in \mathbb{R}^{d}` for the true values
                  of :math:`x`.
        :raises: AttributeError
        :catches: AttributeError

        Get the column vector :math:`x_{true}` for the true values of
        :math:`x`. If no data has been provided, the function will raise and
        catch an exception and display possible solutions to the user.
        '''
        try:
            return self.__xtrue
        except AttributeError:
            print('''
No data for xtrue has been provided so far. Try set_xtrue() for manual setting.
''')

    # -----------------------------------------------------------------------#


    def set_G(self, G):

        '''
        :param G: Column vector :math:`G \in \mathbb{R}^{m}` for the
                  equality constraints.
        :type G: casadi.casadi_core.MX
        :raises: ValueError

        *If data is provided, this function is called automatically at the
        initialization of the object.*

        Set the column vector :math:`G` for the equality constraints.
        '''

        self.__check_variable_validity(G, "G", ca.casadi_core.MX, 1)

        self.__G = G


    def get_G(self):

        '''
        :returns: casadi.casadi_core.MX - the column vector
                  :math:`G \in \mathbb{R}^{m}` for the equality constraints.
        :raises: AttributeError
        :catches: AttributeError

        Get the column vector :math:`G` for the equality constraints.
        If no data has been provided, the function will raise and
        catch an exception and display possible solutions to the user.
        '''

        try:
            return self.__G
        except AttributeError:
            print('''
No data for G has been provided so far. Try set_G() for manual setting.
''')

    # -----------------------------------------------------------------------#


    def set_H(self, H):

        '''
        :param H: Column vector :math:`H` for the
                  inequality constraints.
        :type H: casadi.casadi_core.MX
        :raises: ValueError

        *If data is provided, this function is called automatically at the
        initialization of the object.*

        Set the column vector :math:`H` for the inequality constraints.
        '''

        self.__check_variable_validity(H, "H", ca.casadi_core.MX, 1)

        self.__H = H


    def get_H(self):

        '''
        :returns: casadi.casadi_core.MX - the column vector
                  :math:`H` for the inequality constraints.
        :raises: AttributeError
        :catches: AttributeError

        Get the column vector :math:`H` for the inequality constraints.
        If no data has been provided, the function will raise and
        catch an exception and display possible solutions to the user.
        '''

        try:
            return self.__H
        except AttributeError:
            print('''
No data for G has been provided so far. Try set_H() for manual setting.
''')

    # -----------------------------------------------------------------------#


    def set_xinit(self, xtrue):

        '''
        :param xinit: Column vector :math:`x_{init} \in \mathbb{R}^{d}` 
                      containing the initial guess for :math:`x`.
        :type xtrue: numpy.ndarray
        :raises: ValueError

        *If data is provided, this function is called automatically at the
        initialization of the object.*

        Set the column vector :math:`x_{init}` for the initial guess
        of :math:`x`.
        '''

        self.self.__check_variable_validity(xinit, "xinit", np.ndarray, 1)

        self.__xinit = xinit


    def get_xinit(self):

        '''
        :returns: numpy.ndarray - the column vector
                  :math:`x_{init} \in \mathbb{R}^{d}` 
                  containing the initial guess for :math:`x`.
        :raises: AttributeError
        :catches: AttributeError

        Get the column vector :math:`x_{init}` for the initial guess
        of :math:`x`.
        '''

        try:
            return self.__xinit
        except AttributeError:
            print('''
No data for xinit has been provided so far. Try set_xinit() for manual setting.
''')

    # -----------------------------------------------------------------------#


    def get_xhat(self):

        '''
        :returns: numpy.ndarray - the column vector
                  :math:`\hat{x} \in \mathbb{R}^{d}` 
                  containing the estimated value for :math:`x`.
        :raises: AttributeError
        :catches: AttributeError

        Get the column vector :math:`\hat{x}` for the estimated value
        of :math:`x`.
        '''

        try:
            return self.__xhat
        except AttributeError:
            print('''
No data for xhat has been provided so far. You have to call
run_parameter_estimation() first.
''')    


    def get_Rhat(self):

        '''
        :returns: float - the scalar value
                  :math:`\hat{R} \in \mathbb{R}` 
                  containing the residual for the estimated value
                  :math:`\hat{x}`.
        :raises: AttributeError
        :catches: AttributeError

        Get the scalar value :math:`\hat{R}` containing the residual
        for the estimated value :math:`\hat{x}`.
        '''

        try:
            return self.__Rhat
        except AttributeError:
            print('''
No data for Rhat has been provided so far. You have to call
run_parameter_estimation() first.
''')

    # -----------------------------------------------------------------------#


    def get_beta(self):

        '''
        :returns: float - the scalar value for :math:`\beta`.
        :raises: AttributeError
        :catches: AttributeError

        Get the scalar value for :math:`\beta`. For information about
        computation of :math:`\beta`, see :func:`compute_covariance_matrix()`.
        '''

        try:
            return self.__beta
        except AttributeError:
            print('''
No data for beta has been provided so far. You have to call
compute_covariance_matrix() first.
''')


    def get_Covx(self):

        '''
        :returns: numpy.ndarray - the covariance matrix
        :math:`\Sigma_{\hat{x}} \in \mathbb{R}^{d\,x\,d}` for the
        estimated parameters :math:`\hat{x}`.
        :raises: AttributeError
        :catches: AttributeError

        Get the covariance matrix
        :math:`\Sigma_{\hat{x}}}` for the
        estimated parameters :math:`\hat{x}`. For information about
        computation of :math:`\beta`, see :func:`compute_covariance_matrix()`.
        '''

        try:
            return self.__Covx
        except AttributeError:
            print('''
No data for beta has been provided so far. You have to call
compute_covariance_matrix() first.
''')


    ##########################################################################
    ################## 3. Constructor for the class NLPETool #################
    ##########################################################################

    def __init__(\
        self, x, M, sigma, Y=None, xtrue=None, G=None, H=None, xinit=None):

        '''

        **Mandatory information for constructing the class**

        :param x: Column vector :math:`x \in \mathbb{R}^{d}` for the
                  parameters.
        :type x: casadi.casadi_core.MX

        :param M: Column vector :math:`M \in \mathbb{R}^{N}` for the model.
        :type M: casadi.casadi_core.MX

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

        :param G: Column vector :math:`G \in \mathbb{R}^{m}` for the
                  equality constraints.
        :type G: casadi.casadi_core.MX

        :param H: Column vector :math:`H` for the inequality constraints.
        :type H: casadi.casadi_core.MX

        :param xinit: Column vector :math:`x_{init} \in \mathbb{R}^{d}` for the initial guess of the parameter values.
        :type xinit: numpy.ndarray

        |

        '''

        # Parameters

        self.set_x(x)

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
            self.set_G()

        # Inequality constrains

        if H is not None:
            self.set_H()

        # Initial guess

        if xinit is not None:
            self.__xinit = xinit

        # Get the number of measurements N from M

        self.__N = self.__M.shape[0]

        # Get the number of parameters d from x

        self.__d = self.__x.shape[0]

        # If they exist, get the number of equality constraints from G
        if G is not None:
            self.__m = self.__G.shape[0]
        else:
            self.__m = 0


    ##########################################################################
    ############# 4. Functions for running parameter estimation ##############
    ##########################################################################

    def generate_pseudo_measurement_data(self):

        r'''

        This functions generates "random" pseudo measurement data in
        :math:`Y` for a parameter estimation from :math:`M`,
        :math:`\sigma` and :math:`x_{true}`.

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
        '''

        self.__Mx = ca.MXFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__M))
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

            \hat{x} &= arg\,\underset{x}{min}\|M(x)-Y\|_{2}^{2}\\
            s.\,t.&~\\
            G &= 0\\
            H &\leq 0\\
            x_{0} &= x_{init}


        will be set up, and solved using IPOPT. Afterwards,

        - the value of :math:`\hat{x}` is stored inside the variable xhat,
          which can be returned using the function :func:`get_xhat()`,

        - the value of the residual :math:`\hat{R}` will be stored inside
          the variable Rhat,
          which can be returned using the function :func:`get_Rhat()`.
        '''

        # First, check if measurement data exists; if not, generate it

        if not hasattr(self, '__Y'):

            self.generate_pseudo_measurement_data()            

        # Set up the cost function f

        A = ca.mul(np.linalg.solve(np.sqrt(self.__Sigma), np.eye(self.__N)), \
            (self.__M - self.__Y))
        self.__f = ca.mul(A.T, A)

        # Solve the minimization problem for f

        self.__fx = ca.MXFunction(ca.nlpIn(x=self.__x), \
            ca.nlpOut(f=self.__f, g=self.__G))
        self.__fx.init()

        solver = ca.NlpSolver("ipopt", self.__fx)
        solver.setOption("tol", 1e-10)
        solver.init()

        # If equality constraints exist, set the bounds for the solver

        if hasattr(self, '__m'):
            solver.setInput(np.zeros(self.__m), "lbg")
            solver.setInput(np.zeros(self.__m), "ubg")

        # If an initial guess was given, set the initial guess for the solver
        
        if hasattr(self, '__xinit'):
            solver.setInput(self.__xinit, "xinit")

        solver.evaluate()

        # Store the results of the computation

        self.__xhat = solver.getOutput("x")
        self.__Rhat = solver.getOutput("f")


    ##########################################################################
    ########## 5. Functions for parameter estimation interpretation ##########
    ##########################################################################

    def compute_covariance_matrix(self):
        
        r'''
        This function will compute the covariance matrix
        :math:`\Sigma_{\hat{x}} \in \mathbb{R}^{d\,x\,d}` for the
        estimated parameters :math:`\hat{x}`. It is computed from

        .. math::

            \Sigma_{\hat{x}} = \beta * J^{+, T} * J

        with

        .. math::

            \beta = \frac{\hat{R}}{N + m - d}

        and

        .. math::

            J^{+} = ...

        while

        .. math::

            J_{1} = \Sigma_{\epsilon}^{-1} * \frac{\partial M}{\partial x}

        and

        .. math::

            J_{2} = \frac{\partial G}{\partial x} .

        If the number of equality constraints is 0, copmutation of
        :math:`J_{plus}` simplifies to

        .. math::

            J_{plus} = J_{1}

        Afterwards,

          - the value of :math:`\beta` will be stored inside the variable beta,
            which can be returned using the function :func:`get_beta()`, and
          - the value of :math:`\Sigma_{\hat{x}}` will be stored inside the
            variable Covx which can be returned using the function
            :func:`get_Covx()`.
        '''

        # Compute beta

        if hasattr(self, '__m'):
            self.__beta = self.Rhat / (self.__N + self.__m - self.__d)
        else:
            self.__beta = self.Rhat / (self.__N - self.__d)

        # Compute J1, J2

        Mx = ca.MXFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__M))
        Mx.init()

        self.__J1 = ca.mul(np.linalg.solve(np.sqrt(self.__Sigma), \
            np.eye(self.__N)), Mx.jac("x", "f"))

        Gx = ca.MXFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__G))
        Gx.init()

        self.__J2 = Gx.jac("x", "f")

        # Compute Jplus; this simplifies to Jplus = J1 when m = 0

        if hasattr(self, '__m'):

            self.__Jplus = ca.mul([ \

                ca.horzcat((np.eye(self.__d),np.zeros((self.__d, self.__m)))),\

                ca.solve(ca.vertcat(( \
                
                    ca.horzcat((ca.mul(self.__J1.T, self.__J1), self.__J2.T)),\
                    ca.horzcat((self.__J2, np.zeros((self.__m, self.__m)))) \
                
                )), np.eye(self.__d + self.__m)), \

                ca.vertcat((self.__J1.T, np.zeros((self.__m, self.__N)))) \

                ])

        else:

            self.__Jplus = self.__J1

        # Compute the covariance matrix, and evaluate for xhat

        self.__fCov = beta * ca.mul([self.__Jplus, self.__Jplus.T])

        self.__fCovx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=Cov))
        self.__fCovx.init()

        self.__fCovx.setInput(xhat, "x")
        self.__fCovx.evaluate()

        # Store the covariance matrix in Covx

        self.__Covx = self.__fCovx.getOutput("f")
