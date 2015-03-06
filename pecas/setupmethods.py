#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat
import pylab as pl
from abc import ABCMeta, abstractmethod

import systems

class SetupMethodsBaseClass(object):

    '''The abstract class :class:`SetupMethodsBaseClass` contains the basic
    functionalities of all other classes.'''

    __metaclass__ = ABCMeta


    @abstractmethod
    def __init__(self):

        '''Placeholder-function for the according __init__()-methods of the
        classes that inherit from :class:`SetupMethodsBaseClass`.'''

        pass


    def __repeat_input(self, val, dim):

        '''
        :param val: Value that will possibly be repeated.
        :type val: float, list
        :param dim: The value for how often ``val`` will possibly be repeated.
        :type dim: int

        If the input value ``val`` is not of type list, return a list with
        ``dim`` repetitions of ``val``. Else, return ``val`` without
        modifications.
        '''

        if not isinstance(val, list):

            return list(pl.repeat(val, dim))

        else:

            return val


    def check_and_set_bounds_and_initials(self, \
        umin = None, umax = None, uinit = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):

        '''
        :param tbd: tbd
        :type tbd: tbd

        Define structures for minimum, maximum and initial values for the
        several variables that build up the optimization problem,
        and prepare the values provided with the arguments properly.
        Afterwards, the values are stored inside the class variables ``Vmin``,
        ``Vmax`` and ``Vinit``, respectively.
        '''

        # Define structures from bounds and initial values from the original
        # variable struct of the problem

        self.Vmin = self.V()
        self.Vmax = self.V()
        self.Vinit = self.V()

        # Set initials and bounds for the controls
        # (only if the number of controls is not 0)

        if not self.nu == 0:

            if uinit is None:
                uinit = pl.zeros((self.nu, self.nsteps))
            if umin is None:
                umin = -pl.inf * pl.ones((self.nu, self.nsteps))   
            if umax is None:
                umax = pl.inf * pl.ones((self.nu, self.nsteps))

            uinit = pl.atleast_2d(uinit)
            umin = pl.atleast_2d(umin)
            umax = pl.atleast_2d(umax)

            if uinit.shape == (self.nsteps, self.nu):
                uinit = uinit.T
            
            if umin.shape == (self.nsteps, self.nu):
                umin = umin.T
            
            if umax.shape == (self.nsteps, self.nu):
                umax = umax.T

            if not all(arg.shape == (self.nu, self.nsteps) for \
                arg in [uinit, umin, umax]):

                raise ValueError( \
                    "Wrong dimension for argument uinit, umin or umax.")

            # Repeatd the values for each collocation point

            for k in range(self.nsteps):

                self.Vinit["U", k, :] = ca.tools.repeated(uinit[:,k])
                self.Vmin["U", k, :] = ca.tools.repeated(umin[:,k])
                self.Vmax["U", k, :] = ca.tools.repeated(umax[:,k])

        # Set initials and bounds for the parameters

        if pinit is None:
            pinit = pl.zeros(self.np)
        if pmin is None:
            pmin = -pl.inf * pl.ones(self.np)   
        if pmax is None:
            pmax = pl.inf * pl.ones(self.np)

        pinit = pl.squeeze(pinit)
        pmin = pl.squeeze(pmin)
        pmax = pl.squeeze(pmax)

        if not all(arg.shape == (self.np,) for \
            arg in [pinit, pmin, pmax]):

            raise ValueError( \
                "Wrong dimension for argument pinit, pmin or pmax.")

        self.Vinit["P",:] = pinit
        self.Vmin["P",:] = pmin
        self.Vmax["P",:] = pmax

        # If it's a dynamic problem, set initials and bounds for the states

        if "X" in self.V.keys():

            if xinit is None:
                xinit = pl.zeros((self.nx, self.nsteps + 1))
            if xmin is None:
                xmin = -pl.inf * pl.ones((self.nx, self.nsteps + 1))   
            if xmax is None:
                xmax = pl.inf * pl.ones((self.nx, self.nsteps + 1))

            xinit = pl.atleast_2d(xinit)
            xmin = pl.atleast_2d(xmin)
            xmax = pl.atleast_2d(xmax)

            if xinit.shape == (self.nsteps + 1, self.nx):
                xinit = xinit.T
            
            if xmin.shape == (self.nsteps + 1, self.nx):
                xmin = xmin.T
            
            if xmax.shape == (self.nsteps + 1, self.nx):
                xmax = xmax.T

            if not all(arg.shape == (self.nx, self.nsteps + 1) for \
                arg in [xinit, xmin, xmax]):

                raise ValueError( \
                    "Wrong dimension for argument xinit, xmin or xmax.")

            for k in range(self.nsteps + 1):

                self.Vinit["X",k,:] = ca.tools.repeated(xinit[:,k])
                self.Vmin["X",k,:] = ca.tools.repeated(xmin[:,k])
                self.Vmax["X",k,:] = ca.tools.repeated(xmax[:,k])

            # Set the state bounds at the initial time, if explicitly given

            if x0min is not None:

                x0min = pl.atleast_2d(x0min)

                if not x0min.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument x0min.")

                self.Vmin["X",0,0] = x0min

            if x0max is not None:

                x0max = pl.atleast_2d(x0max)

                if not x0max.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument x0max.")

                self.Vmax["X",0,0] = x0max

            # Set state bounds at the final time, if explicitly given

            if xNmin is not None:

                xNmin = pl.atleast_2d(xNmin)

                if not xNmin.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument xNmin.")

                self.Vmin["X",-1,0] = xNmin

            if xNmax is not None:

                xNmax = pl.atleast_2d(xNmax)

                if not xNmax.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument xNmax.")

                self.Vmax["X",-1,0] = xNmax

            # Set the bounds on the disturbances

            self.Vinit["W",:] = ca.tools.repeated(0.0)
            self.Vmin["W",:] = ca.tools.repeated(-pl.inf)
            self.Vmax["W",:] = ca.tools.repeated(pl.inf)


class BSsetup(SetupMethodsBaseClass):

    def check_and_set_bounds_and_initials(self, \
        umin = None, umax = None, uinit = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):

        super(BSsetup, self).check_and_set_bounds_and_initials( \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)


    def __init__(self, system = None, timegrid = None, \
        umin = pl.zeros(0), umax = pl.zeros(0), uinit = pl.zeros(0), \
        pmin = pl.zeros(0), pmax = pl.zeros(0), pinit = pl.zeros(0)):

        if not type(system) is systems.BasicSystem:

            raise TypeError("Setup-method " + self.__class__.__name__ + \
                " not allowed for system of type " + str(type(system)) + ".")

        # Dimensions

        self.nu = system.v["u"].shape[0]
        self.np = system.v["p"].shape[0]

        if pl.atleast_2d(timegrid).shape[0] == 1:

            self.timegrid = pl.asarray(timegrid)

        elif pl.atleast_2d(timegrid).shape[1] == 1:

                self.timegrid = pl.squeeze(pl.atleast_2d(timegrid).T)

        else:

            raise ValueError("Invalid dimension for argument timegrid.")

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

        self.check_and_set_bounds_and_initials( \
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


class CollocationBaseClass(SetupMethodsBaseClass):

    __metaclass__ = ABCMeta

    def check_and_set_bounds_and_initials(self, \
        umin = None, umax = None, uinit = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):

        super(CollocationBaseClass, self).check_and_set_bounds_and_initials( \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)


    @abstractmethod
    def __init__(self, system = None, timegrid = None, \
        umin = None, umax = None, uinit = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None, \
        systemclass = None):

        if not type(system) is systemclass:

            raise TypeError("Setup-method " + self.__class__.__name__ + \
                " not allowed for system of type " + str(type(system)) + ".")

        # Dimensions

        self.nx = system.v["x"].shape[0]
        self.nu = system.v["u"].shape[0]
        self.np = system.v["p"].shape[0]

        if pl.atleast_2d(timegrid).shape[0] == 1:

            self.timegrid = pl.asarray(timegrid)

        elif pl.atleast_2d(timegrid).shape[1] == 1:

                self.timegrid = pl.squeeze(pl.atleast_2d(timegrid).T)

        else:

            raise ValueError("Invalid dimension for argument timegrid.")

        self.nsteps = self.timegrid.shape[0] - 1

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

        self.check_and_set_bounds_and_initials( \
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


class ODEsetup(CollocationBaseClass):

    def __init__(self, system = None, timegrid = None, \
        umin = None, umax = None, uinit = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):

        super(ODEsetup, self).__init__(system = system, \
            timegrid = timegrid, \
            umin = umin, umax = umax, uinit = uinit, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax, \
            systemclass = systems.ExplODE)

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
