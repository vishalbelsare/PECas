#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat
import numpy as np
from abc import ABCMeta, abstractmethod

import ipdb
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
        uN = None, \
        pinit = None, \
        xinit = None):

        '''
        :param tbd: tbd
        :type tbd: tbd

        Define structures for minimum, maximum and initial values for the
        several variables that build up the optimization problem,
        and prepare the values provided with the arguments properly.
        Afterwards, the values are stored inside the class variables
        ``Varsmin``, ``Varsmax`` and ``Varsinit``, respectively.
        '''

        # Define structures for initial values from the original
        # variable struct of the problem

        self.Varsinit = self.Vars()

        # Set controls values
        # (only if the number of controls is not 0, else set them 0)

        if not self.nu == 0:

            if uN is None:
                uN = np.zeros((self.nu, self.nsteps))

            uN = np.atleast_2d(uN)

            if uN.shape == (self.nsteps, self.nu):
                uN = uN.T

            if not uN.shape == (self.nu, self.nsteps):

                raise ValueError( \
                    "Wrong dimension for control values uN.")

            self.uN = uN

        else:

            self.uN = ca.DMatrix(0, self.nsteps)

        # Set initials for the parameters

        if pinit is None:
            pinit = np.zeros(self.np)

        pinit = np.atleast_1d(np.squeeze(pinit))

        if not pinit.shape == (self.np,):

            raise ValueError( \
                "Wrong dimension for argument pinit.")

        self.Varsinit["P",:] = pinit


        # If it's a dynamic problem, set initials and bounds for the states

        if "X" in self.Vars.keys():

            if xinit is None:
                xinit = np.zeros((self.nx, self.nsteps + 1))

            xinit = np.atleast_2d(xinit)

            if xinit.shape == (self.nsteps + 1, self.nx):
                xinit = xinit.T

            if not xinit.shape == (self.nx, self.nsteps + 1):

                raise ValueError( \
                    "Wrong dimension for argument xinit.")

            for k in range(self.nsteps):

                self.Varsinit["X",k,:] = ca.tools.repeated(xinit[:,k])

            self.Varsinit["XF"] = xinit[:,-1]


            # Set the bounds on the equation errors

            self.Varsinit["WE",:] = ca.tools.repeated(0.0)
            
            # Set the bounds on the input errors
            
            self.Varsinit["WU",:] = ca.tools.repeated(0.0)
            
        # Set the bounds on the measurement errors

        self.Varsinit["V",:] = ca.tools.repeated(0.0)


class BSsetup(SetupsBaseClass):

    def check_and_set_bounds_and_initials(self, \
        uN = None,
        pinit = None, \
        xinit = None):

        self.tstart_setup = time.time()

        super(BSsetup, self).check_and_set_bounds_and_initials( \
            uN = uN,
            pinit = pinit, \
            xinit = xinit)


    def __init__(self, system = None, \
        tu = None, uN = None, \
        pinit = None):

        SetupsBaseClass.__init__(self)

        if not type(system) is systems.BasicSystem:

            raise TypeError("Setup-method " + self.__class__.__name__ + \
                " not allowed for system of type " + str(type(system)) + ".")

        self.system = system

        # Dimensions

        self.nu = system.u.shape[0]
        self.np = system.p.shape[0]
        self.nphi = system.phi.shape[0]

        if np.atleast_2d(tu).shape[0] == 1:

            self.tu = np.asarray(tu)

        elif np.atleast_2d(tu).shape[1] == 1:

                self.tu = np.squeeze(np.atleast_2d(tu).T)

        else:

            raise ValueError("Invalid dimension for argument tu.")

        self.nsteps = tu.shape[0]

        # Define the struct holding the variables

        self.Vars = cat.struct_symMX([
                (
                    cat.entry("P", shape = self.np),
                    cat.entry("V", repeat = [self.nsteps], \
                        shape = self.nphi),
                )
            ])

        # Set bounds and initial values

        self.check_and_set_bounds_and_initials( \
            uN = uN,
            pinit = pinit)

        # Set up phiN

        self.phiN = []

        phifcn = ca.MXFunction("phifcn", \
            [system.t, system.u, system.p], [system.phi])

        for k in range(self.nsteps):

            self.phiN.append(phifcn([self.tu[k], \
                self.uN[:, k], self.Vars["P"]])[0])

        self.phiN = ca.vertcat(self.phiN)

        self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

        # Set up g

        # TODO! Can/should/must gfcn depend on uN and/or t?

        gfcn = ca.MXFunction("gfcn", [system.p], [system.g])

        self.g = gfcn.call([self.Vars["P"]])[0]

        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of BasicSystem system sucessful.')


class ODEsetup(SetupsBaseClass):

    def check_and_set_bounds_and_initials(self, \
        uN = None, \
        pinit = None, \
        xinit = None):

        super(ODEsetup, self).check_and_set_bounds_and_initials( \
            uN = uN, \
            pinit = pinit, \
            xinit = xinit)


    def __init__(self, system = None, \
        tu = None, uN = None, \
        ty = None, yN = None,
        pinit = None, \
        xinit = None, \
        scheme = "radau", \
        order = 3):

        self.tstart_setup = time.time()

        SetupsBaseClass.__init__(self)

        if not type(system) is systems.ExplODE:

            raise TypeError("Setup-method " + self.__class__.__name__ + \
                " not allowed for system of type " + str(type(system)) + ".")

        self.system = system

        # Dimensions

        self.nx = system.x.shape[0]
        self.nu = system.u.shape[0]
        self.np = system.p.shape[0]
        self.nwe = system.we.shape[0]
        self.nwu = system.wu.shape[0]        
        self.nphi = system.phi.shape[0]

        if np.atleast_2d(tu).shape[0] == 1:

            self.tu = np.asarray(tu)

        elif np.atleast_2d(tu).shape[1] == 1:

                self.tu = np.squeeze(np.atleast_2d(tu).T)

        else:

            raise ValueError("Invalid dimension for argument tu.")


        # pdb.set_trace()

        if ty == None:

            self.ty = self.tu

        elif np.atleast_2d(ty).shape[0] == 1:

            self.ty = np.asarray(ty)

        elif np.atleast_2d(ty).shape[1] == 1:

            self.ty = np.squeeze(np.atleast_2d(ty).T)

        else:

            raise ValueError("Invalid dimension for argument ty.")


        self.nsteps = self.tu.shape[0] - 1

        self.scheme = scheme
        self.order = order
        self.tauroot = ca.collocationPoints(order, scheme)

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
                        shape = self.nphi),
                    cat.entry("WE", repeat = [self.nsteps, self.ntauroot], \
                        shape = self.nwe),
                    cat.entry("WU", repeat = [self.nsteps, self.ntauroot], \
                        shape = self.nwu)
                )
            ])

        # Define bounds and initial values

        self.check_and_set_bounds_and_initials( \
            uN = uN, \
            pinit = pinit, \
            xinit = xinit)


        # Set tp the collocation coefficients

        # Coefficients of the collocation equation

        self.C = np.zeros((self.ntauroot + 1, self.ntauroot + 1))

        # Coefficients of the continuity equation

        self.D = np.zeros(self.ntauroot + 1)

        # Dimensionless time inside one control interval

        tau = ca.SX.sym("tau")

        # Construct the matrix T that contains all collocation time points

        self.T = np.zeros((self.nsteps, self.ntauroot + 1))

        for k in range(self.nsteps):

            for j in range(self.ntauroot + 1):

                self.T[k,j] = self.tu[k] + \
                    (self.tu[k+1] - self.tu[k]) * self.tauroot[j]

        # For all collocation points

        self.lfcns = []

        for j in range(self.ntauroot + 1):

            # Construct Lagrange polynomials to get the polynomial basis
            # at the collocation point
            
            L = 1
            
            for r in range(self.ntauroot + 1):
            
                if r != j:
            
                    L *= (tau - self.tauroot[r]) / \
                        (self.tauroot[j] - self.tauroot[r])
            
            lfcn = ca.SXFunction("lfcn", [tau],[L])
          
            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation
            
            [self.D[j]] = lfcn([1])

            # Evaluate the time derivative of the polynomial at all 
            # collocation points to get the coefficients of the
            # collocation equation
            
            tfcn = lfcn.tangent()

            for r in range(self.ntauroot + 1):

                self.C[j,r] = tfcn([self.tauroot[r]])[0]

            self.lfcns.append(lfcn)


        # Initialize phiN

        self.phiN = []

        # Initialize measurement function

        phifcn = ca.MXFunction("phifcn", \
            [system.t, system.u, system.x, system.wu, system.p], \
            [system.phi])

        # Initialzie setup of g

        self.g = []

        # Initialize ODE right-hand-side

        ffcn = ca.MXFunction("ffcn", \
            [system.t, system.u, system.x, system.we, system.wu, system.p], \
            [system.f])

        # For all finite elements

        for k in range(self.nsteps):

            hk = self.tu[k + 1] - self.tu[k]
            t_meas = self.ty[np.where(np.logical_and( \
                self.ty >= self.tu[k], self.ty < self.tu[k + 1]))]

            for t_meas_j in t_meas:

                if t_meas_j == self.tu[k]:

                    self.phiN.append(phifcn([ \
                        self.tu[k], self.uN[:, k], self.Vars["X", k, 0], \
                        self.Vars["WU", k, 0], self.Vars["P"]])[0])

                else:


                    tau = (t_meas_j - self.tu[k]) / hk

                    x_temp = 0

                    for r in range(self.ntauroot + 1):

                        x_temp += self.lfcns[r]([tau])[0] * self.Vars["X",k,r]

                    self.phiN.append(phifcn([\
                        t_meas_j, self.uN[:, k], x_temp, \
                        self.Vars["WU", k, 0],self.Vars["P"]])[0])

            # For all collocation points

            for j in range(1, self.ntauroot + 1):
                
                # Get an expression for the state derivative at
                # the collocation point

                xp_jk = 0
                
                for r in range(self.ntauroot + 1):
                    
                    xp_jk += self.C[r,j] * self.Vars["X",k,r]
          
                # Add collocation equations to the NLP

                [fk] = ffcn([ \
                    self.T[k][j], self.uN[:, k], self.Vars["X",k,j], \
                    self.Vars["WE", k, j-1], self.Vars["WU", k, j-1], \
                    self.Vars["P"]])

                self.g.append(hk * fk - xp_jk)

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
    

        # for k in range(self.nsteps):

            # DEPENDECY ON U NOT POSSIBLE AT THIS POINT! len(U) = N, not N + 1!
            # self.phiN.append(yfcn.call([self.tu[k], self.Vars["U", k, 0], \
            # self.phiN.append(yfcn.call([self.tu[k], self.Vars["X", k, 0], \
            #     self.Vars["P"], self.uN[:, k], self.Vars["WE", k, 0],\
            #     self.Vars["WU", k, 0]])[0])

        if self.tu[-1] in self.ty:


            self.phiN.append(phifcn([self.tu[-1], self.uN[:, -1], \
                self.Vars["XF"], self.Vars["WU", -1, 0], self.Vars["P"]])[0])

        self.phiN = ca.vertcat(self.phiN)

        self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of ExplODE system sucessful.')
