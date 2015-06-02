#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat
import pylab as pl
from abc import ABCMeta, abstractmethod

import pdb
import time

import systems
import intro

import time

class SetupsBaseClass(object):

    '''The abstract class :class:`SetupsBaseClass` contains the basic
    functionalities of all other classes.'''

    __metaclass__ = ABCMeta


    @abstractmethod
    def __init__(self):

        '''Placeholder-function for the according __init__()-methods of the
        classes that inherit from :class:`SetupsBaseClass`.'''

        intro.pecas_intro()
        print('\n' + 24 * '-' + \
            ' PECas system initialization ' + 25 * '-')
        print('\nStart system initialization ...')


    def check_and_set_bounds_and_initials(self, \
        u = None, \
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
        Afterwards, the values are stored inside the class variables
        ``Varsmin``, ``Varsmax`` and ``Varsinit``, respectively.
        '''

        # Define structures from bounds and initial values from the original
        # variable struct of the problem

        self.Varsmin = self.Vars()
        self.Varsmax = self.Vars()
        self.Varsinit = self.Vars()

        # Set controls values
        # (only if the number of controls is not 0, else set them 0)

        if not self.nu == 0:

            if u is None:
                u = pl.zeros((self.nu, self.nsteps))

            u = pl.atleast_2d(u)

            if u.shape == (self.nsteps, self.nu):
                u = u.T

            if not u.shape == (self.nu, self.nsteps):

                raise ValueError( \
                    "Wrong dimension for control values u.")

            self.u = u

        else:

            self.u = pl.zeros((1, self.nsteps))

        # Set initials and bounds for the parameters

        if pinit is None:
            pinit = pl.zeros(self.np)
        if pmin is None:
            pmin = -pl.inf * pl.ones(self.np)   
        if pmax is None:
            pmax = pl.inf * pl.ones(self.np)

        pinit = pl.atleast_1d(pl.squeeze(pinit))
        pmin = pl.atleast_1d(pl.squeeze(pmin))
        pmax = pl.atleast_1d(pl.squeeze(pmax))

        if not all(arg.shape == (self.np,) for \
            arg in [pinit, pmin, pmax]):

            raise ValueError( \
                "Wrong dimension for argument pinit, pmin or pmax.")

        self.Varsinit["P",:] = pinit
        self.Varsmin["P",:] = pmin
        self.Varsmax["P",:] = pmax

        # If it's a dynamic problem, set initials and bounds for the states

        if "X" in self.Vars.keys():

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

            for k in range(self.nsteps):

                self.Varsinit["X",k,:] = ca.tools.repeated(xinit[:,k])
                self.Varsmin["X",k,:] = ca.tools.repeated(xmin[:,k])
                self.Varsmax["X",k,:] = ca.tools.repeated(xmax[:,k])

            self.Varsinit["XF"] = xinit[:,-1]
            self.Varsmin["XF"] = xmin[:,-1]
            self.Varsmax["XF"] = xmax[:,-1]

            # Set the state bounds at the initial time, if explicitly given

            if x0min is not None:

                x0min = pl.atleast_2d(x0min)

                if x0min.shape == (self.nx, 1):

                    x0min = x0min.T

                if not x0min.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument x0min.")

                self.Varsmin["X",0,0] = x0min

            if x0max is not None:

                x0max = pl.atleast_2d(x0max)

                if x0max.shape == (self.nx, 1):

                    x0max = x0max.T

                if not x0max.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument x0max.")

                self.Varsmax["X",0,0] = x0max

            # Set state bounds at the final time, if explicitly given

            if xNmin is not None:

                xNmin = pl.atleast_2d(xNmin)

                if xNmin.shape == (self.nx, 1):

                    xNmin = xNmin.T

                if not xNmin.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument xNmin.")

                self.Varsmin["XF"] = xNmin

            if xNmax is not None:

                xNmax = pl.atleast_2d(xNmax)

                if xNmax.shape == (self.nx, 1):

                    xNmax = xNmax.T

                if not xNmax.shape == (1, self.nx):

                    raise ValueError("Wrong dimension for argument xNmax.")

                self.Varsmax["XF"] = xNmax

            # Set the bounds on the equation errors

            self.Varsinit["W",:] = ca.tools.repeated(0.0)
            self.Varsmin["W",:] = ca.tools.repeated(-pl.inf)
            self.Varsmax["W",:] = ca.tools.repeated(pl.inf)

        # Set the bounds on the measurement errors

        self.Varsinit["V",:] = ca.tools.repeated(0.0)
        self.Varsmin["V",:] = ca.tools.repeated(-pl.inf)
        self.Varsmax["V",:] = ca.tools.repeated(pl.inf)


class BSsetup(SetupsBaseClass):

    def check_and_set_bounds_and_initials(self, \
        u = None,
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):

        self.tstart_setup = time.time()

        super(BSsetup, self).check_and_set_bounds_and_initials( \
            u = u,
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)


    def __init__(self, system = None, timegrid = None, \
        u = None, \
        pmin = None, pmax = None, pinit = None):

        SetupsBaseClass.__init__(self)

        if not type(system) is systems.BasicSystem:

            raise TypeError("Setup-method " + self.__class__.__name__ + \
                " not allowed for system of type " + str(type(system)) + ".")

        self.system = system

        # Dimensions

        self.nu = system.vars["u"].shape[0]
        self.np = system.vars["p"].shape[0]
        self.nv = system.fcn["y"].shape[0]
        self.ny = system.fcn["y"].shape[0]

        if pl.atleast_2d(timegrid).shape[0] == 1:

            self.timegrid = pl.asarray(timegrid)

        elif pl.atleast_2d(timegrid).shape[1] == 1:

                self.timegrid = pl.squeeze(pl.atleast_2d(timegrid).T)

        else:

            raise ValueError("Invalid dimension for argument timegrid.")

        self.nsteps = timegrid.shape[0]

        # Define the struct holding the variables

        self.Vars = cat.struct_symMX([
                (
                    cat.entry("P", shape = self.np),
                    cat.entry("V", repeat = [self.nsteps], \
                        shape = self.nv),
                )
            ])

        # Set bounds and initial values

        self.check_and_set_bounds_and_initials( \
            u = u,
            pmin = pmin, pmax = pmax, pinit = pinit)

        # Set up phiN

        self.phiN = []

        yfcn = ca.MXFunction([system.vars["t"], system.vars["u"], \
            system.vars["p"]], [system.fcn["y"]])
        yfcn.setOption("name", "yfcn")
        yfcn.init()

        for k in range(self.nsteps):

            self.phiN.append(yfcn.call([self.timegrid[k], \
                self.u[:, k], self.Vars["P"]])[0])

        self.phiN = ca.vertcat(self.phiN)

        self.phiNfcn = ca.MXFunction([self.Vars], [self.phiN])
        self.phiNfcn.setOption("name", "phiNfcn")
        self.phiNfcn.init()

        # Set up g

        # TODO! Can/should/must gfcn depend on u and/or t?

        gfcn = ca.MXFunction([system.vars["p"]], [system.fcn["g"]])
        gfcn.setOption("name", "gfcn")
        gfcn.init()

        self.g = gfcn.call([self.Vars["P"]])[0]

        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of BasicSystem system sucessful.')


class CollocationBaseClass(SetupsBaseClass):

    __metaclass__ = ABCMeta

    def check_and_set_bounds_and_initials(self, \
        u = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):

        super(CollocationBaseClass, self).check_and_set_bounds_and_initials( \
            u = u, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)


    @abstractmethod
    def __init__(self, system = None, timegrid = None, \
        u = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None, \
        systemclass = None):

        SetupsBaseClass.__init__(self)

        if not type(system) is systemclass:

            raise TypeError("Setup-method " + self.__class__.__name__ + \
                " not allowed for system of type " + str(type(system)) + ".")

        self.system = system

        # Dimensions

        self.nx = system.vars["x"].shape[0]
        self.nu = system.vars["u"].shape[0]
        self.np = system.vars["p"].shape[0]
        self.nv = system.fcn["y"].shape[0]
        self.nw = system.vars["w"].shape[0]
        self.ny = system.fcn["y"].shape[0]

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

        self.Vars = cat.struct_symMX([
                (
                    cat.entry("P", shape = self.np), \
                    cat.entry("X", repeat = [self.nsteps, self.ntauroot+1], \
                        shape = self.nx), \
                    cat.entry("XF", shape = self.nx), \
                    cat.entry("V", repeat = [self.nsteps+1], \
                        shape = self.nv),
                    cat.entry("W", repeat = [self.nsteps, self.ntauroot], \
                        shape = self.nw)
                )
            ])

        # Define bounds and initial values

        self.check_and_set_bounds_and_initials( \
            u = u, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax)

        # Set up phiN

        self.phiN = []

        yfcn = ca.MXFunction([system.vars["t"], system.vars["x"], \
            system.vars["p"], system.vars["u"],system.vars["w"]], [system.fcn["y"]])
        yfcn.setOption("name", "yfcn")
        yfcn.init()

        for k in range(self.nsteps):

            # DEPENDECY ON U NOT POSSIBLE AT THIS POINT! len(U) = N, not N + 1!
            # self.phiN.append(yfcn.call([self.timegrid[k], self.Vars["U", k, 0], \
            self.phiN.append(yfcn.call([self.timegrid[k], self.Vars["X", k, 0], \
                self.Vars["P"], self.u[:, k], self.Vars["W", k, 0]])[0])

        self.phiN.append(yfcn.call([self.timegrid[k], self.Vars["XF"], \
            self.Vars["P"], self.u[:, -1],self.Vars["W", -1, 0]])[0])

        self.phiN = ca.vertcat(self.phiN)

        self.phiNfcn = ca.MXFunction([self.Vars], [self.phiN])
        self.phiNfcn.setOption("name", "phiNfcn")
        self.phiNfcn.init()


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
            # collocation equation
            
            tfcn = lfcn.tangent()
            tfcn.init()

            for r in range(self.ntauroot + 1):

                tfcn.setInput(self.tauroot[r])
                tfcn.evaluate()
                self.C[j,r] = tfcn.getOutput()

        # Start setting up g

        self.g = []


class ODEsetup(CollocationBaseClass):

    def __init__(self, system = None, timegrid = None, \
        u = None, \
        pmin = None, pmax = None, pinit = None, \
        xmin = None, xmax = None, xinit = None, \
        x0min = None, x0max = None, \
        xNmin = None, xNmax = None):
            
        self.tstart_setup = time.time()

        super(ODEsetup, self).__init__(system = system, \
            timegrid = timegrid, \
            u = u, \
            pmin = pmin, pmax = pmax, pinit = pinit, \
            xmin = xmin, xmax = xmax, xinit = xinit, \
            x0min = x0min, x0max = x0max, \
            xNmin = xNmin, xNmax = xNmax, \
            systemclass = systems.ExplODE)

        ffcn = ca.MXFunction([system.vars["t"], system.vars["x"], \
            system.vars["u"], system.vars["p"], system.vars["w"]], \
            [system.fcn["f"]])
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
                    
                    xp_jk += self.C[r,j] * self.Vars["X",k,r]
          
                # Add collocation equations to the NLP

                [fk] = ffcn.call([self.T[k][j], self.Vars["X",k,j], \
                    self.u[:, k], self.Vars["P"], \
                    self.Vars["W", k, j-1]])

                self.g.append((self.timegrid[k+1] - \
                    self.timegrid[k]) * fk - xp_jk)

            # Get an expression for the state at the end of
            # the finite element
            
            xf_k = 0

            for r in range(self.ntauroot + 1):

                xf_k += self.D[r] * self.Vars["X",k,r]
            
            # Add the continuity equation to NLP
            
            if k == (self.nsteps - 1):

                self.g.append(self.Vars["XF"] - xf_k)

            else:

                self.g.append(self.Vars["X",k+1,0] - xf_k)

        # Concatenate constraints

        self.g = ca.vertcat(self.g)
        
        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of ExplODE system sucessful.')