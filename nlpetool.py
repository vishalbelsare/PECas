import casadi as ca
import numpy as np

class NLPETool:

    '''Here to be the docstring for the class.'''

    ##########################################################################
    #### 1. Functions for checking validity and consistency of the inputs ####
    ##########################################################################

    def __check_variable_validity(self, var, varname, dtype, ddim):

        '''
        Check the validity of an input variables.
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


    def __check_variable_consistency(self, var1, ddim1, var2, ddim2):

        '''
        Check the consistency for two input variables.
        '''

        if ddim1 != ddim2:
            raise ValueError('''
The dimensions of the variables "{0}" and "{2}" do not match, since
"{0}" has {1} entries, while "{2}"" has {4} entries.'''.format(\
                var1, ddim1, var2, ddim2))


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


    def get_sigma(self):

        '''
        :returns: numpy.ndarray - the column vector
                  :math:`\sigma  \in \mathbb{R}^{N}` for the model.

        Get the column vector :math:`\sigma` for the standard deviations.
        '''

        return self.__sigma

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

    ##########################################################################
    ################### Constructor for the class NLPETool ###################
    ##########################################################################

    def __init__(\
        self, x, M, sigma, Y=None, xtrue=None, G=None, H=None, xinit=None):

        '''

        **Mandatory information for constructing the class**

        :param x: Column vector :math:`x \in \mathbb{R}^{d}` for the
                  parameters.
        :type x: casadi.casadi_core.MX.

        :param M: Column vector :math:`M \in \mathbb{R}^{N}` for the model.
        :type M: casadi.casadi_core.MX.

        :param sigma: Column vector :math:`\sigma \in \mathbb{R}^{N}` for
                      the standard deviations.
        :type sigma: numpy.ndarray.


        **Mutually substitutable information for constructing the class**

        *One of the two variables Y and xtrue has to be set!*

        :param Y: Column vector :math:`Y \in \mathbb{R}^{N}` for the
                  measurements.
        :type Y: numpy.ndarray.

        :param xtrue: Column vector :math:`x_{true} \in \mathbb{R}^{d}` 
                      containing the true value of :math:`x` to
                      generate pseudo measurement data using
                      the vectors :math:`M` and :math:`\sigma`.
        :type xtrue: numpy.ndarray.


        **Optional information for constructing the class**

        :param G: Column vector :math:`G \in \mathbb{R}^{m}` for the
                  equality constraints.
        :type G: casadi.casadi_core.MX.

        :param H: Column vector :math:`H` for the inequality constraints.
        :type H: casadi.casadi_core.MX.

        :param xinit: Column vector :math:`x_{init} \in \mathbb{R}^{d}` for the initial guess of the parameter values.
        :type xinit: numpy.ndarray.

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


        ## Check variable consistency ##

        # Check whether M and sigma, and Y if already given, have the same
        # number of entries

        self.__check_variable_consistency("M", self.__M.shape[0], \
            "sigma", self.__sigma.shape[0])

        if Y is not None:
            self.__check_variable_consistency("Y", self.__Y.shape[0], \
                "sigma", self.__sigma.shape[0])


        ## Get problem dimension ##

        # Get the number of measurements N from M

        self.__N = self.__M.shape[0]

        # Get the number of parameters d from x

        self.__d = self.__x.shape[0]

        # If they exist, get the number of equality constraints from G
        if G is not None:
            self.__m = self.__G.shape[0]
        else:
            self.__m = 0


    def generate_pseudo_measurement_data(self):

        '''
        This functions generates pseudo measurement data for the parameter
        estimation from M and sigma.

        '''

        self.__Mx = ca.MXFunction(ca.nlpIn(x=self.__x), ca.nlpOut(f=self.__M))
        self.__Mx.setOption("name", "Model function Mx")
        self.__Mx.init()

        self.__Mx.setInput(self.__xtrue, "x")
        self.__Mx.evaluate()

        self.__Y = self.__Mx.getOutput("f") + \
            np.random.normal(0, self.__sigma, self.__N)


    def run_parameter_estimation(self):

        '''
        This functions will run the parameter estimation for the given model.
        If measurement data is not yet existing, it will be generate using
        the class function generate_pseudo_measurement_data().

        '''

        ## Check variable consistency again ##

        # Check whether M, sigma and Y have the same number of entries

        # self.__check_variable_consistency("M", self.__M.shape[0], \
        #     "sigma", self.__sigma.shape[0])

        # # With this, also check if measurements data exists; if not,
        # # generate it

        # try:
        #     self.__check_variable_consistency("Y", self.__Y.shape[0], \
        #         "sigma", self.__sigma.shape[0])
        # except NameError:
        #     self.generate_pseudo_measurement_data()            

        # Set up cost function f

        pass

        # A = ca.mul(np.linalg.solve(np.sqrt(Sigma), np.eye(N)), (M - Y_N))
        # f = ca.mul(A.T, A)

        # Solve minimization problem for f

        # fx = ca.MXFunction(ca.nlpIn(x=x), ca.nlpOut(f=f, g=G))
        # fx.init()

        # solver = ca.NlpSolver("ipopt", fx)
        # solver.setOption("tol", 1e-10)
        # solver.init()

        # solver.setInput(np.zeros(m), "lbg")
        # solver.setInput(np.zeros(m), "ubg")
        # solver.setInput([1, 1], "xinit")

        # solver.evaluate()