import casadi as ca
import casadi.tools as cat

class BasicProblem(object):


    __v_struct = cat.struct_symMX(["p", "x", "u", "z", "t"])
    __fg_struct = cat.struct_symMX(["f", "g"])
    __meas = cat.struct_symMX(["tg", "meas"])


    def __check_for_valid_input(self, inp):

        if inp is not None:

            return inp

        else:

            raise ValueError('''
You have to specify a value for setting up this class.
''')


    def __init__(self, \
                 p = None, pmin = None, pmax = None, \
                 t = None, \
                 f = None, g = None, \
                 tg = None, meas = None):

        self.v = self.__v_struct()
        self.vmin = self.__v_struct()
        self.vmax = self.__v_struct()

        self.fg = self.__fg_struct()

        self.meas = self.__meas()

        self.v["p"] = self.__check_for_valid_input(p)
        self.vmin["p"] = self.__check_for_valid_input(pmin)
        self.vmax["p"] = self.__check_for_valid_input(pmax)

        self.v["t"] = self.__check_for_valid_input(t) # optional!

        self.fg["f"] = self.__check_for_valid_input(f)
        self.fg["g"] = self.__check_for_valid_input(g) # optional!

        self.meas["tg"] = self.__check_for_valid_input(tg)
        self.meas["meas"] = self.__check_for_valid_input(meas)


class ODEProblem(BasicProblem):


    def __check_for_valid_input(self, inp):

        return inp
        # return super(ODEProblem, self).__check_for_valid_input(inp)


    def __init__(self, \
                 p = None, pmax = None, pmin = None, \
                 x = None, xmin = None, xmax = None, \
                 t = None, \
                 f = None, g = None, \
                 tg = None, meas = None):

        super(ODEProblem, self).__init__(p, pmax, pmin, f, g)

        self.v["x"] = self.__check_for_valid_input(x)
        self.vmin["x"] = self.__check_for_valid_input(xmin)
        self.vmax["x"] = self.__check_for_valid_input(xmax)
