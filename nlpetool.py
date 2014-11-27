import casadi as ca
import numpy as np

class NLPETool:

    '''Here to be the docstring for the class.'''

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


        def check_variable_validity(var, varname, dtype, ddim):

            # Check validity of input variables

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
        

        def check_variable_consistency(var1, ddim1, var2, ddim2):

            # Check consistency of input variables

            if ddim1 != ddim2:
                raise ValueError('''
The dimensions of the variables "{0}" and "{2}" do not match, since
"{0}" has {1} entries, while "{2}"" has {4} entries.'''.format(\
                    var1, ddim1, var2, ddim2))


        ## Check variable validity ##

        # Necessary information

        # Parameters

        check_variable_validity(x, "x", ca.casadi_core.MX, 1)
        self.x = x

        # Model

        check_variable_validity(M, "M", ca.casadi_core.MX, 1)
        self.M = M

        # Standard deviations

        check_variable_validity(sigma, "sigma", np.ndarray, 1)
        self.sigma = sigma

        # Mutually substitutable information

        # Measurements

        if Y is not None:
            check_variable_validity(Y, "Y", np.ndarray, 1)
            self.Y = Y

        # True value of x

        if xtrue is not None:
            check_variable_validity(xtrue, "xtrue", np.ndarray, 1)
            self.xtrue = xtrue

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
            check_variable_validity(G, "G", ca.casadi_core.MX, 1)
            self.G = G

        # Inequality constrains

        if H is not None:
            check_variable_validity(H, "H", ca.casadi_core.MX, 1)
            self.H = H

        # Initial guess

        if x0 is not None:
            check_variable_validity(x0, "x0", np.ndarray, 1)
            self.x0 = x0


        ## Check variable consistency ##

        # Check whether M and sigma, and Y if already given, have the same
        # number of entries

        check_variable_consistency("M", self.M.shape[0], \
            "sigma", self.sigma.shape[0])

        if Y is not None:
            check_variable_consistency("Y", self.Y.shape[0], \
                "sigma", self.sigma.shape[0])


        ## Get problem dimension ##

        # Get the number of measurements N from M

        self.N = self.M.shape[0]

        # Get the number of parameters d from x

        self.d = self.x.shape[0]

        # If they exist, get the number of equality constraints from G
        if G is not None:
            self.m = self.G.shape[0]
        else:
            self.m = 0


    def generate_pseudo_measurement_data(self):

        '''
        This functions generates pseudo measurement data for the parameter
        estimation from M and sigma.

        '''

        self.Mx = ca.MXFunction(ca.nlpIn(x=self.x), ca.nlpOut(f=self.M))
        self.Mx.setOption("name", "Model function Mx")
        self.Mx.init()

        self.Mx.setInput(self.xtrue, "x")
        self.Mx.evaluate()

        self.Y = self.Mx.getOutput("f") + \
            np.random.normal(0, self.sigma, self.N)


    def run_parameter_estimation(self):

        '''
        This functions will run the parameter estimation for the given model.
        If measurement data is not yet existing, it will be generate using
        the class function generate_pseudo_measurement_data().

        '''

        ## Check variable consistency again ##

        # Check whether M, sigma and Y have the same number of entries

        # check_variable_consistency("M", self.M.shape[0], \
        #     "sigma", self.sigma.shape[0])

        # # With this, also check if measurements data exists; if not,
        # # generate it

        # try:
        #     check_variable_consistency("Y", self.Y.shape[0], \
        #         "sigma", self.sigma.shape[0])
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