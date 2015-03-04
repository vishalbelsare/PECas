import casadi as ca
import casadi.tools as cat
import pylab as pl
from abc import ABCMeta, abstractmethod

class PESetupBaseClass(object):

    __metaclass__ = ABCMeta


    @abstractmethod
    def __init__(self):

        pass


    def __repeat_input(self, val, dim):

        if not isinstance(val, list):

            return list(pl.repeat(val, dim))

        else:

            return val


    def set_bounds_and_initials(self, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0, \
        xmin = -pl.inf, xmax = pl.inf, xinit = 0.0, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf):

         # Define bounds and initial values

        self.Vmin = self.V()
        self.Vmax = self.V()
        self.Vinit = self.V()

        # Set control initials and bounds

        if not self.nu == 0:

            if self.nu == 1:

                uinit = uinit[pl.newaxis,:]
                umin = umin[pl.newaxis,:]
                umax= umax[pl.newaxis,:]

            for k in range(self.nsteps):

                self.Vinit["U", k, :] = ca.tools.repeated(uinit[:,k])
                self.Vmin["U", k, :] = ca.tools.repeated(umin[:,k])
                self.Vmax["U", k, :] = ca.tools.repeated(umax[:,k])

        # Set parameter initials and bounds

        self.Vinit["P",:] = self.__repeat_input(pinit, self.np)
        self.Vmin["P",:] = self.__repeat_input(pmin, self.np)
        self.Vmax["P",:] = self.__repeat_input(pmax, self.np)

        # Set states initials and bounds, if contained

        if "X" in self.V.keys():

            self.Vinit["X",:,:] = ca.tools.repeated( \
                ca.tools.repeated(self.__repeat_input(xinit, self.nx)))

            self.Vmin["X",:,:] = ca.tools.repeated( \
                ca.tools.repeated(self.__repeat_input(xmin, self.nx)))

            self.Vmax["X",:,:] = ca.tools.repeated( \
                ca.tools.repeated(self.__repeat_input(xmax, self.nx)))

            # State at initial time

            self.Vmin["X",0,0] = self.__repeat_input(x0min, self.nx)
            self.Vmax["X",0,0] = self.__repeat_input(x0max, self.nx)

            # State at end time

            self.Vmin["X",-1,0] = self.__repeat_input(xNmin, self.nx)
            self.Vmax["X",-1,0] = self.__repeat_input(xNmax, self.nx)

            # Disturbances

            self.Vinit["W",:] = ca.tools.repeated(0.0)
            self.Vmin["W",:] = ca.tools.repeated(-pl.inf)
            self.Vmax["W",:] = ca.tools.repeated(pl.inf)


class BSEvaluation(PESetupBaseClass):

    def set_bounds_and_initials(self, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0, \
        xmin = -pl.inf, xmax = pl.inf, xinit = 0.0, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf):

        super(BSEvaluation, self).set_bounds_and_initials( \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)


    def __init__(self, system = None, timegrid = None, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0):

        # Dimensions

        self.nu = system.v["u"].shape[0]
        self.np = system.v["p"].shape[0]

        self.timegrid = timegrid
        self.nsteps = timegrid.shape[0]

        # Define the struct holding the variables

        self.V = cat.struct_symMX([
                (
                    cat.entry("U", repeat = [self.nsteps, 1], \
                        shape = system.v["u"].shape),
                    cat.entry("P", shape = system.v["p"].shape),
                )
            ])

        # Set bounds and initial values

        self.set_bounds_and_initials( \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit)

        # Set up phiN

        phiN = []

        yfcn = ca.SXFunction([system.v["t"], system.v["u"], system.v["p"]], \
            [system.fcn["y"]])
        yfcn.setOption("name", "yfcn")
        yfcn.init()

        for k in range(self.nsteps):

            self.phiN.append(yfcn.call([self.timegrid[k], self.V["U", k, 0], \
                self.V["P"]])[0])

        self.phiN = ca.vertcat(self.phiN)

        # Set up s

        self.s = []

        # Set up g

        self.g = system.fcn["g"]


class CollocationBaseClass(PESetupBaseClass):

    __metaclass__ = ABCMeta

    def set_bounds_and_initials(self, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0, \
        xmin = -pl.inf, xmax = pl.inf, xinit = 0.0, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf):

        super(CollocationBaseClass, self).set_bounds_and_initials( \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)


    @abstractmethod
    def __init__(self, system = None, timegrid = None, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0, \
        xmin = -pl.inf, xmax = pl.inf, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf, xinit = 0.0):

        # Dimensions

        self.nx = system.v["x"].shape[0]
        self.nu = system.v["u"].shape[0]
        self.np = system.v["p"].shape[0]

        self.timegrid = timegrid
        self.nsteps = timegrid.shape[0] - 1

        self.tauroot = ca.collocationPoints(3, "radau")

        # Degree of interpolating polynomial

        self.ntauroot = len(self.tauroot) - 1

        # Define the struct holding the variables

        self.V = cat.struct_symMX([
                (
                    cat.entry("U", repeat = [self.nsteps, self.ntauroot], shape = self.nu),
                    cat.entry("X", repeat = [self.nsteps+1, self.ntauroot+1], \
                        shape = self.nx),
                    cat.entry("P", shape = self.np),
                    cat.entry("W", repeat = [self.nsteps], shape = self.nx)
                )
            ])

        # Define bounds and initial values

        self.set_bounds_and_initials( \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)

        # Set up phiN

        self.phiN = []

        yfcn = ca.SXFunction([system.v["t"], system.v["x"], \
            system.v["p"]], [system.fcn["y"]])
        yfcn.setOption("name", "yfcn")
        yfcn.init()

        for k in range(self.nsteps + 1):

            # DEPENDECY ON U NOT POSSIBLE AT THIS POINT! len(U) = N, not N + 1!
            # self.phiN.append(yfcn.call([self.timegrid[k], self.V["U", k, 0], \
            self.phiN.append(yfcn.call([self.timegrid[k], self.V["X", k, 0], \
                self.V["P"]])[0])

        self.phiN = ca.vertcat(self.phiN)

        # Set tp the collocation coefficients

        # Coefficients of the collocation equation

        self.C = pl.zeros((self.ntauroot + 1, self.ntauroot + 1))

        # Coefficients of the continuity equation

        self.D = pl.zeros(self.ntauroot + 1)

        # Dimensionless time inside one control interval

        tau = ca.SX.sym("tau")

        # Construct the matrix T that contains all collocation time points

        self.T = pl.zeros((self.nsteps, self.ntauroot + 1))

        for k in range(self.nsteps):

            for j in range(self.ntauroot + 1):

                self.T[k,j] = self.timegrid[k] + \
                    (self.timegrid[k+1] - self.timegrid[k]) * self.tauroot[j]

        # For all collocation points

        for j in range(self.ntauroot + 1):

            # Construct Lagrange polynomials to get the polynomial basis
            # at the collocation point
            
            L = 1
            
            for r in range(self.ntauroot + 1):
            
                if r != j:
            
                    L *= (tau - self.tauroot[r]) / \
                        (self.tauroot[j] - self.tauroot[r])
            
            lfcn = ca.SXFunction([tau],[L])
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

            for r in range(self.ntauroot + 1):

                tfcn.setInput(self.tauroot[r])
                tfcn.evaluate()
                self.C[j,r] = tfcn.getOutput()

        # Set up s

        self.s = ca.vertcat(self.V["W", :])

        # Set up g

        self.g = []


class ODECollocation(CollocationBaseClass):

    def __init__(self, system = None, timegrid = None, \
        umin = -pl.inf * pl.ones(1), umax = pl.inf * pl.ones(1), \
        uinit = pl.zeros(1), \
        pmin = -pl.inf, pmax = pl.inf, pinit = 0.0, \
        xmin = -pl.inf, xmax = pl.inf, \
        x0min = -pl.inf, x0max = pl.inf, \
        xNmin = -pl.inf, xNmax = pl.inf, xinit = 0.0):

        super(ODECollocation, self).__init__(system = system, \
            timegrid = timegrid, \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)

        ffcn = ca.SXFunction([system.v["t"], system.v["x"], system.v["u"], \
            system.v["p"]], [system.fcn["f"]])
        ffcn.setOption("name", "ffcn")
        ffcn.init()

        # For all finite elements

        for k in range(self.nsteps):
          
            # For all collocation points

            for j in range(1, self.ntauroot + 1):
                
                # Get an expression for the state derivative at
                # the collocation point

                xp_jk = 0
                
                for r in range(self.ntauroot + 1):
                    
                    xp_jk += self.C[r,j] * self.V["X",k,r]
          
                # Add collocation equations to the NLP

                [fk] = ffcn.call([self.T[k][j], self.V["X",k,j], \
                    self.V["U",k,j-1], self.V["P"]])
                self.g.append((self.timegrid[k+1] - \
                    self.timegrid[k]) * fk - xp_jk)

            # Get an expression for the state at the end of
            # the finite element
            
            xf_k = 0

            for r in range(self.ntauroot + 1):

                xf_k += self.D[r] * self.V["X",k,r]
            
            # Add the continuity equation to NLP
            
            self.g.append(self.V["X",k+1,0] - xf_k + self.V["W", k])

        # Concatenate constraints

        self.g = ca.vertcat(self.g)
