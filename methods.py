import casadi as ca
import casadi.tools as cat
import pylab as pl
from abc import ABCMeta, abstractmethod

class BSEvaluation:

    def __init__(self, bp = None, timegrid = None):

        self.Y = []
        self.G = []
        self.timegrid = timegrid
        self.N = timegrid.shape[0]

        self.V = cat.struct_symMX([
                (
                    cat.entry("U", repeat = [self.N], shape = bp.v["u"].shape),
                    cat.entry("P", shape = bp.v["p"].shape),
                )
            ])

        yfcn = ca.SXFunction([bp.v["t"], bp.v["u"], bp.v["p"]], [bp.fcn["y"]])
        yfcn.init()

        for k in range(self.N):

            self.Y.append(yfcn.call([self.timegrid[k], self.V["U"][k], \
                self.V["P"]])[0])

        self.Y = ca.vertcat(self.Y)
        self.G = bp.fcn["g"]


class CollocationBaseClass(object):

    __metaclass__ = ABCMeta


    def repeat_input(self, val, dim):

        if not isinstance(val, list):

            return list(pl.repeat(val, dim))

        else:

            return val


    @abstractmethod
    def __init__(self, op = None, timegrid = None, \
        xmin = -pl.inf, xmax = pl.inf, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf, xinit = 0.0, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0):

        self.Y = []
        self.G = []

        self.nx = op.v["x"].shape[0]
        self.nu = op.v["u"].shape[0]
        self.np = op.v["p"].shape[0]

        self.timegrid = timegrid
        self.N = timegrid.shape[0] - 1
        self.tau_root = ca.collocationPoints(3, "radau")

        # Degree of interpolating polynomial

        self.d = len(self.tau_root) - 1

        self.V = cat.struct_symMX([
                (
                    cat.entry("U", repeat = [self.N, self.d], shape = self.nu),
                    cat.entry("X", repeat = [self.N+1, self.d+1], shape = self.nx),
                    cat.entry("P", shape = self.np),
                    cat.entry("W", repeat = [self.N], shape = self.nx)
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

                self.T[k,j] = self.timegrid[k] + \
                    (self.timegrid[k+1] - self.timegrid[k]) * self.tau_root[j]

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


        # Define bounds and initial values

        self.Vmin = self.V()
        self.Vmax = self.V()
        self.Vinit = self.V()

        # Set states and its bounds

        self.Vinit["X",:,:] = ca.tools.repeated( \
            ca.tools.repeated(self.repeat_input(xinit, self.nx)))

        self.Vmin["X",:,:] = ca.tools.repeated( \
            ca.tools.repeated(self.repeat_input(xmin, self.nx)))

        self.Vmax["X",:,:] = ca.tools.repeated( \
            ca.tools.repeated(self.repeat_input(xmax, self.nx)))

        # Set controls and its bounds

        if not self.nu == 0:

            if self.nu == 1:

                uinit = uinit[pl.newaxis,:]
                umin = umin[pl.newaxis,:]
                umax= umax[pl.newaxis,:]

            for k in range(self.N):

                self.Vinit["U", k, :] = ca.tools.repeated(uinit[:,k])
                self.Vmin["U", k, :] = ca.tools.repeated(umin[:,k])
                self.Vmax["U", k, :] = ca.tools.repeated(umax[:,k])

        # State at initial time

        self.Vmin["X",0,0] = self.repeat_input(x0min, self.nx)
        self.Vmax["X",0,0] = self.repeat_input(x0max, self.nx)

        # State at end time

        self.Vmin["X",-1,0] = self.repeat_input(xNmin, self.nx)
        self.Vmax["X",-1,0] = self.repeat_input(xNmax, self.nx)

        # Disturbances

        self.Vinit["W",:] = ca.tools.repeated(0.0)
        self.Vmin["W",:] = ca.tools.repeated(-pl.inf)
        self.Vmax["W",:] = ca.tools.repeated(pl.inf)

        # Parameters

        self.Vinit["P",:] = self.repeat_input(pinit, self.np)
        self.Vmin["P",:] = self.repeat_input(pmin, self.np)
        self.Vmax["P",:] = self.repeat_input(pmax, self.np)


class ODECollocation(CollocationBaseClass):

    def __init__(self, op = None, timegrid = None, \
        xmin = -pl.inf, xmax = pl.inf, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf, xinit = 0.0, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0):

        super(ODECollocation, self).__init__(op, timegrid, \
            xmin, xmax, x0min, x0max, xNmin, xNmax, xinit, \
            umin, umax, uinit, pmin, pmax, pinit)

        self.f = ca.SXFunction([op.v["t"], op.v["x"], op.v["u"], op.v["p"]], \
            [op.v["f"]])
        self.f.init()

        # For all finite elements

        for k in range(self.N):
          
            # For all collocation points

            for j in range(1, self.d + 1):
                
                # Get an expression for the state derivative at
                # the collocation point

                xp_jk = 0
                
                for r in range(self.d + 1):
                    
                    xp_jk += self.C[r,j] * self.V["X",k,r]
          
                # Add collocation equations to the NLP

                [fk] = self.f.call([self.T[k][j], self.V["X",k,j], \
                    self.V["U",k,j-1], self.__V["P"]])
                self.G.append((tgrid[k+1] - tgrid[k]) * fk - xp_jk)

            # Get an expression for the state at the end of
            # the finite element
            
            xf_k = 0

            for r in range(self.d + 1):

                xf_k += self.D[r] * self.V["X",k,r]
            
            # Add the continuity equation to NLP
            
            self.G.append(self.V["X",k+1,0] - xf_k + self.V["W", k])

        # Equality constraints (lbg = ubg)

        # self.__gmin = list(pl.zeros(self.__N * self.__d + self.__N))
        # self.__gmax = list(pl.zeros(self.__N * self.__d + self.__N))

        # Concatenate constraints

        # self.Y = something ...
        self.G = ca.vertcat(G)