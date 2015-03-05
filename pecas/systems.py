import casadi as ca
import casadi.tools as cat

class BasicSystem(object):

    def __init__(self, \
                 t = ca.SX.sym("t", 1), u = ca.SX.sym("u", 0), \
                 x = ca.SX.sym("x", 0), z = ca.SX.sym("z", 0), \
                 p = None, \
                 y = None, \
                 f = ca.SX.sym("f", 0), g = ca.SX.sym("g", 0)):

        self.v = cat.struct_MX([
                (
                    cat.entry("t", expr = t),
                    cat.entry("u", expr = u),
                    cat.entry("x", expr = x),
                    cat.entry("z", expr = z),
                    cat.entry("p", expr = p)
                )
            ])

        self.fcn = cat.struct_MX([
                (
                    cat.entry("y", expr = y),
                    cat.entry("f", expr = f),
                    cat.entry("g", expr = g)
                )
            ])


class ExplODE(BasicSystem):

    def __init__(self, \
                 t = ca.SX.sym("t", 1), u = ca.SX.sym("u", 0), \
                 x = None, z = ca.SX.sym("z", 0), \
                 p = None, \
                 y = None, \
                 f = None, g = ca.SX.sym("g", 0)):

        super(ExplODE, self).__init__(t = t, u = u, x = x, p = p, y = y, f = f)
