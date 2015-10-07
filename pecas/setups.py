#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat
import numpy as np
from abc import ABCMeta, abstractmethod

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

    # @profile
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

        # Set controls values
        # (only if the number of controls is not 0, else set them nothing)

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

        self.Pinit = pinit


        # If it's a dynamic problem, set initials and bounds for the states

        if type(self.system) is not systems.BasicSystem:

            if xinit is None:
                xinit = np.zeros((self.nx, self.nsteps + 1))

            xinit = np.atleast_2d(xinit)

            if xinit.shape == (self.nsteps + 1, self.nx):
                xinit = xinit.T

            if not xinit.shape == (self.nx, self.nsteps + 1):

                raise ValueError( \
                    "Wrong dimension for argument xinit.")


            self.Xinit = ca.repmat(xinit[:,:-1], self.ntauroot+1, 1)
            self.XFinit = xinit[:,-1]

        else:

            self.Xinit = ca.DMatrix(0, 0)
            self.XFinit = ca.DMatrix(0, 0)

        self.Vinit = np.zeros(self.V.shape)
        self.EPS_Einit = np.zeros(self.EPS_E.shape)
        self.EPS_Uinit = np.zeros(self.EPS_U.shape)


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

        self.P = ca.MX.sym("P", self.np)
        self.X = ca.DMatrix(0, self.nsteps)
        self.XF = ca.DMatrix(0, self.nsteps)
        
        self.V = ca.MX.sym("V", self.nphi, self.nsteps)

        self.EPS_E = ca.DMatrix(0, self.nsteps)
        self.EPS_U = ca.DMatrix(0, self.nsteps)

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
                self.uN[:, k], self.P])[0])

        self.phiN = ca.vertcat(self.phiN)

        # self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

        # Set up g

        # TODO! Can/should/must gfcn depend on uN and/or t?

        gfcn = ca.MXFunction("gfcn", [system.p], [system.g])

        self.g = gfcn.call([self.P])[0]

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


    # @profile
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
        self.neps_e = system.eps_e.shape[0]
        self.neps_u = system.eps_u.shape[0]        
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

        # Define the optimization variables

        self.P = ca.MX.sym("P", self.np)
        self.X = ca.MX.sym("X", (self.nx * (self.ntauroot+1)), self.nsteps)
        self.XF = ca.MX.sym("XF", self.nx)

        self.V = ca.MX.sym("V", self.nphi, self.nsteps+1)

        if self.neps_e != 0:

            self.EPS_E = ca.MX.sym("EPS_E", \
                (self.neps_e * self.ntauroot), self.nsteps)

        else:

            self.EPS_E = ca.DMatrix(0, self.nsteps)

        if self.neps_u != 0:
                
            self.EPS_U = ca.MX.sym("EPS_U", \
                (self.neps_u * self.ntauroot), self.nsteps)

        else:

            self.EPS_U = ca.DMatrix(0, self.nsteps)

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

        self.T = self.T.T

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
            [system.t, system.u, system.x, system.eps_u, system.p], \
            [system.phi])

        # Initialzie setup of g

        self.g = []

        # Initialize ODE right-hand-side

        ffcn = ca.MXFunction("ffcn", \
            [system.t, system.u, system.x, system.eps_e, system.eps_u, \
            system.p], [system.f])

        # Collect information for measurement function

        # Structs to hold variables for later mapped evaluation

        Tphi = []
        Uphi = []
        Xphi = []
        EPS_Uphi = []

        for k in range(self.nsteps):

            hk = self.tu[k + 1] - self.tu[k]
            t_meas = self.ty[np.where(np.logical_and( \
                self.ty >= self.tu[k], self.ty < self.tu[k + 1]))]

            for t_meas_j in t_meas:

                Uphi.append(self.uN[:, k])
                EPS_Uphi.append(self.EPS_U[:self.neps_u, k])

                if t_meas_j == self.tu[k]:

                    Tphi.append(self.tu[k])
                    Xphi.append(self.X[:self.nx, k])

                else:

                    tau = (t_meas_j - self.tu[k]) / hk

                    x_temp = 0

                    for r in range(self.ntauroot + 1):

                        x_temp += self.lfcns[r]([tau])[0] * \
                        self.X[r*self.nx : (r+1) * self.nx, k]

                    Tphi.append(t_meas_j)
                    Xphi.append(x_temp)

        if self.tu[-1] in self.ty:

            Tphi.append(self.tu[-1])
            Uphi.append(self.uN[:,-1])
            Xphi.append(self.XF)
            EPS_Uphi.append(self.EPS_U[:self.neps_u,-1])


        # Mapped calculation of the collocation equations

        # Collocation nodes

        hc = ca.MX.sym("hc", 1)
        tc = ca.MX.sym("tc", self.ntauroot)
        xc = ca.MX.sym("xc", self.nx * (self.ntauroot+1))
        eps_ec = ca.MX.sym("eps_ec", self.neps_e * self.ntauroot)
        eps_uc = ca.MX.sym("eps_uc", self.neps_u * self.ntauroot)

        coleqn = ca.vertcat([ \

            hc * ffcn([tc[j-1], \
                system.u, \
                xc[j*self.nx : (j+1)*self.nx], \
                eps_ec[(j-1)*self.neps_e : j*self.neps_e], \
                eps_uc[(j-1)*self.neps_u : j*self.neps_u], \
                system.p])[0] - \

            sum([self.C[r,j] * xc[r*self.nx : (r+1)*self.nx] \

                for r in range(self.ntauroot + 1)]) \
                    
                    for j in range(1, self.ntauroot + 1)])

        coleqnfcn = ca.MXFunction("coleqnfcn", \
            [hc, tc, system.u, xc, eps_ec, eps_uc, system.p], [coleqn])
        coleqnfcn = coleqnfcn.expand()

        [gcol] = coleqnfcn.map([ \
            np.atleast_2d((self.tu[1:] - self.tu[:-1])), self.T[1:,:], \
            self.uN, self.X, self.EPS_E, self.EPS_U, self.P])


        # Continuity nodes

        xnext = ca.MX.sym("xnext", self.nx)

        conteqn = xnext - sum([self.D[r] * xc[r*self.nx : (r+1)*self.nx] \
            for r in range(self.ntauroot + 1)])

        conteqnfcn = ca.MXFunction("conteqnfcn", [xnext, xc], [conteqn])
        conteqnfcn = conteqnfcn.expand()

        [gcont] = conteqnfcn.map([ \
            ca.horzcat([self.X[:self.nx, 1:], self.XF]), self.X])


        # Stack equality constraints together

        self.g = ca.veccat([gcol, gcont])


        # Evaluation of the measurement function

        [self.phiN] = phifcn.map( \
            [ca.horzcat(k) for k in Tphi, Uphi, Xphi, EPS_Uphi] + \
            [self.P])

        # self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of ExplODE system sucessful.')
