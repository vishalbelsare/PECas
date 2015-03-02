import casadi as ca
import casadi.tools as cat
import pylab as pl
from abc import ABCMeta, abstractmethod

class BPEval:

    def __init__(self, bp, N):

        self.Y = []
        self.G = []
        self.N = N

        self.V = cat.struct_symMX([
                (
                    cat.entry("T", repeat = [self.N], shape = bp.v["t"].shape),
                    cat.entry("U", repeat = [self.N], shape = bp.v["u"].shape),
                    cat.entry("P", shape = bp.v["p"].shape),
                    cat.entry("W", repeat = [self.N], shape = bp.fcn["g"].shape)
                )
            ])

        yfcn = ca.SXFunction([bp.v["t"], bp.v["u"], bp.v["p"]], [bp.fcn["y"]])
        yfcn.init()

        gfcn = ca.SXFunction([bp.v["t"], bp.v["u"], bp.v["p"]], [bp.fcn["g"]])
        gfcn.init()

        for k in range(self.N):

            self.Y.append(yfcn.call([self.V["T"][k], self.V["U"][k], \
                self.V["P"]])[0])
            self.G.append(gfcn.call([self.V["T"][k], self.V["U"][k], \
                self.V["P"]])[0])

        self.Y = ca.vertcat(self.Y)
        self.G = ca.vertcat(self.G)


class CollocationBase(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, op, N):

        self.Y = []
        self.G = []
        self.N = N

        self.tau_root = ca.collocationPoints(3, "radau")

        # Degree of interpolating polynomial

        self.d = len(self.tau_root) - 1

        self.V = cat.struct_symMX([
                (
                    cat.entry("T", repeat = [self.N], shape = op.v["t"].shape),
                    cat.entry("U", repeat = [self.N, self.d], shape = op.v["u"].shape),
                    cat.entry("X", repeat = [self.N+1, self.d+1], shape = op.v["x"].shape),
                    cat.entry("P", shape = op.v["p"].shape),
                    cat.entry("W", repeat = [self.N], shape = op.v["x"].shape)
                )
            ])


        # Coefficients of the collocation equation

        self.C = pl.zeros((self.d + 1, self.d + 1))

        # Coefficients of the continuity equation

        self.D = pl.zeros(self.d + 1)

        # Dimensionless time inside one control interval

        self.tau = ca.SX.sym("tau")

        # Construct the matrix T that contains all collocation time points

        self.T = pl.zeros((self.N, self.d + 1))

        for k in range(self.N):

            for j in range(self.d + 1):

                self.T[k,j] = self.V["T", k] + \
                    (self.V["T", k+1] - self.V["T", k]) * self.tau_root[j]

        # For all collocation points

        for j in range(self.d + 1):

            # Construct Lagrange polynomials to get the polynomial basis
            # at the collocation point
            
            L = 1
            
            for r in range(self.d + 1):
            
                if r != j:
            
                    L *= (self.tau - self.tau_root[r]) / \
                        (self.tau_root[j] - self.tau_root[r])
            
            lfcn = ca.SXFunction([self.tau],[L])
            lfcn.init()
          
            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation
            
            lfcn.setInput(1.0)
            lfcn.evaluate()
            self.D[j] = lfcn.getOutput()

            # Evaluate the time derivative of the polynomial at all 
            # collocation points to get the coefficients of the
            # continuity equation
            
            tfcn = lfcn.tangent()
            tfcn.init()

            for r in range(self.d + 1):

                tfcn.setInput(self.tau_root[r])
                tfcn.evaluate()
                self.C[j,r] = tfcn.getOutput()


class ODECollocation(CollocationBase):

    def __init__(self, op, N):

        super(ODECollocation, self).__init__(op, N)

