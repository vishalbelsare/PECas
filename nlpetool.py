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
        Set the column vector for the parameters (type casadi.casadi_core.MX).
        This function is automatically called at the initialization of the
        object and only needed to change parameters after the initialization.
        '''

        self.__check_variable_validity(x, "x", ca.casadi_core.MX, 1)
        self.__x = x


    def get_x(self):

        '''
        Get the column vector for the parameters (type casadi.casadi_core.MX).
        '''

        return self.__x

    # -----------------------------------------------------------------------#


    def set_M(self, M):

        '''
        Set the column vector for the model (type casadi.casadi_core.MX).
        This function is automatically called at the initialization of the
        object and only needed to change parameters after the initialization.
        '''

        self.__check_variable_validity(M, "M", ca.casadi_core.MX, 1)
        self.__M = M


    def get_M(self):

        '''
        Get the column vector for the model (type casadi.casadi_core.MX).
        '''

        return self.__M

    # -----------------------------------------------------------------------#


    def set_sigma(self, sigma):

        '''
        Set the column vector for the standard deviation (type numpy.ndarray).
        This function is automatically called at the initialization of the
        object and only needed to change parameters after the initialization.
        '''

        self.__check_variable_validity(sigma, "sigma", np.ndarray, 1)
        self.__sigma = sigma


    def get_sigma(self):

        '''
        Get the column vector for the standard deviation (type numpy.ndarray).
        '''

        return self.__sigma

    # -----------------------------------------------------------------------#




    ##########################################################################
    ################### Constructor for the class NLPETool ###################
    ##########################################################################

    def __init__(\
        self, x, M, sigma, Y=None, xtrue=None, G=None, H=None, x0=None):

        '''
        The inputs for the constructor are as follows:

        Necessary information:

        x:          Column vector for the parameters
                    (type casadi.casadi_core.MX)
        M:          Column vector for the model
                    (type casadi.casadi_core.MX)
        sigma:      Column vector for the standard deviations
                    (type numpy.ndarray)

        Mutually substitutable information:

        (exactly one of the two variables Y and xtrue has to be set)

        Y:          Column vector for the Measurements
                    (type numpy.ndarray)
        xtrue:      Column vector containing the true value of x to generate
                    pseudo measurement data using M and sigma
                    (type numpy.ndarray)

        Optional information:

        G:          Column vector for the equality constraints
                    (type casadi.casadi_core.MX)

        H:          Column vector for the inequality constraints
                    (type casadi.casadi_core.MX)

        x0:         Column vector for the initial guess x0
                    (type numpy.ndarray)

        '''

        ## Check variable validity ##

        # Necessary information

        # Parameters

        # self.__check_variable_validity(x, "x", ca.casadi_core.MX, 1)
        # self.__x = x
        self.set_x(x)

        # Model

        # self.__check_variable_validity(M, "M", ca.casadi_core.MX, 1)
        # self.__M = M
        self.set_M(M)

        # Standard deviations

        # self.__check_variable_validity(sigma, "sigma", np.ndarray, 1)
        # self.__sigma = sigma

        self.set_sigma(sigma)

        # Mutually substitutable information

        # Measurements

        if Y is not None:
            self.__check_variable_validity(Y, "Y", np.ndarray, 1)
            self.__Y = Y

        # True value of x

        if xtrue is not None:
            self.__check_variable_validity(xtrue, "xtrue", np.ndarray, 1)
            self.__xtrue = xtrue

        # Assure that only one of the variables Y and xtrue is provided

        if (Y is None) and (xtrue is None):
            raise ValueError('''
If no measurement data "Y" is provided, the true value of x must be provided
in "xtrue" so pseudo measurement data can be created for parameter estimation.
''')
        elif (Y is not None) and (xtrue is not None):
            raise ValueError('''
You can only either directly provide measurement data in "Y", or or the true
value of x in "xtrue" to generate pseudo measurement data, not both.
''')

        # Optional information

        # Equality constrains

        if G is not None:
            self.__check_variable_validity(G, "G", ca.casadi_core.MX, 1)
            self.__G = G

        # Inequality constrains

        if H is not None:
            self.__check_variable_validity(H, "H", ca.casadi_core.MX, 1)
            self.__H = H

        # Initial guess

        if x0 is not None:
            self.__check_variable_validity(x0, "x0", np.ndarray, 1)
            self.__x0 = x0


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
        # solver.setInput([1, 1], "x0")

        # solver.evaluate()