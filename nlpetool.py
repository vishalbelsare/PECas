import casadi as ca
import numpy as np

class NLPETool:

    '''Here to be the docstring for the class.'''

    def __init__(\
        self, x, M, sigma, Y=None, xtrue=None, G=None, H=None, x0=None):
    
        '''Here to be the docstring for the initialization.'''


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

        if (self.Y is None) and (self.xtrue is None):
            
            raise ValueError('''
If no measurement data "Y" is provided, the true value of x must be provided
in "xtrue" so pseudo measurement data can be created for parameter estimation.
''')
        elif (self.Y is not None) and (self.xtrue is not None):
            
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

        # ... tbd ...