#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as cat
import numpy as np
from abc import ABCMeta, abstractmethod

import time
import ipdb

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

        # self.Varsinit = self.Vars()

        # Set controls values
        # (only if the number of controls is not 0, else set them 0)

        if not self.nu == 0:

            if uN is None:
                uN = np.zeros((self.nu, self.nsteps))
                # uN = np.zeros((self.nsteps, self.nu))

            uN = np.atleast_2d(uN)

            if uN.shape == (self.nsteps, self.nu):
            # if uN.shape == (self.nu, self.nsteps):
                uN = uN.T

            if not uN.shape == (self.nu, self.nsteps):
            # if not uN.shape == (self.nsteps, self.nu):

                raise ValueError( \
                    "Wrong dimension for control values uN.")

            self.uN = uN

        else:

            self.uN = ca.DMatrix(0, self.nsteps)
            # self.uN = ca.DMatrix(self.nsteps, 0)

        # Set initials for the parameters

        if pinit is None:
            pinit = np.zeros(self.np)

        pinit = np.atleast_1d(np.squeeze(pinit))

        if not pinit.shape == (self.np,):

            raise ValueError( \
                "Wrong dimension for argument pinit.")

        # self.Varsinit["P",:] = pinit
        self.Pinit = pinit


        # If it's a dynamic problem, set initials and bounds for the states

        # if "X" in self.Vars.keys():
        if type(self.system) is not systems.BasicSystem:

            if xinit is None:
                xinit = np.zeros((self.nx, self.nsteps + 1))
                # xinit = np.zeros((self.nsteps + 1, self.nx))

            xinit = np.atleast_2d(xinit)

            if xinit.shape == (self.nsteps + 1, self.nx):
            # if xinit.shape == (self.nx, self.nsteps + 1):
                xinit = xinit.T

            if not xinit.shape == (self.nx, self.nsteps + 1):
            # if not xinit.shape == (self.nsteps + 1, self.nx):

                raise ValueError( \
                    "Wrong dimension for argument xinit.")

            # for k in range(self.nsteps):

            #     self.Varsinit["X",k,:] = ca.tools.repeated(xinit[:,k])

            # self.Varsinit["XF"] = xinit[:,-1]

        self.Xinit = ca.repmat(xinit[:,:-1], self.ntauroot+1, 1)
        self.XFinit = xinit[:,-1]

        self.Vinit = np.zeros(self.V.shape)
        self.WEinit = np.zeros(self.WE.shape)
        self.WUinit = np.zeros(self.WU.shape)


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

        # self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

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

        # self.Vars = cat.struct_symMX([
        #         (
        #             cat.entry("P", shape = self.np), \
        #             cat.entry("X", repeat = [self.nsteps, self.ntauroot+1], \
        #                 shape = self.nx), \
        #             cat.entry("XF", shape = self.nx), \
        #             cat.entry("V", repeat = [self.nsteps+1], \
        #                 shape = self.nphi),
        #             cat.entry("WE", repeat = [self.nsteps, self.ntauroot], \
        #                 shape = self.nwe),
        #             cat.entry("WU", repeat = [self.nsteps, self.ntauroot], \
        #                 shape = self.nwu)
        #         )
        #     ])

        self.P = ca.MX.sym("P", self.np)
        self.X = ca.MX.sym("X", (self.nx * (self.ntauroot+1)), self.nsteps)
        self.XF = ca.MX.sym("XF", self.nx)

        self.V = ca.MX.sym("V", self.nphi, self.nsteps+1)

        if self.nwe != 0:

            self.WE = ca.MX.sym("WE", (self.nwe * self.ntauroot), self.nsteps)

        else:

            self.WE = ca.DMatrix(0, self.nsteps)

        if self.nwu != 0:
                
            self.WU = ca.MX.sym("WU", (self.nwu * self.ntauroot), self.nsteps)

        else:

            self.WU = ca.DMatrix(0, self.nsteps)

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
            [system.t, system.u, system.x, system.wu, system.p], \
            # [system.t, system.u, system.x, system.p], \
            [system.phi])
        # phifcn.expand()

        # Initialzie setup of g

        self.g = []

        # Initialize ODE right-hand-side

        ffcn = ca.MXFunction("ffcn", \
            [system.t, system.u, system.x, system.we, system.wu, system.p], \
            # [system.t, system.u, system.x, system.wu, system.p], \
            [system.f])
        # ffcn.expand()

        # Collect information for measurement function

        # Structs to hold variables for later mapped evaluation

        Tphi = []
        Uphi = []
        Xphi = []
        WUphi = []

        for k in range(self.nsteps):

            hk = self.tu[k + 1] - self.tu[k]
            t_meas = self.ty[np.where(np.logical_and( \
                self.ty >= self.tu[k], self.ty < self.tu[k + 1]))]

            for t_meas_j in t_meas:

                Uphi.append(self.uN[:, k])
                WUphi.append(self.WU[:self.nwu, k])

                if t_meas_j == self.tu[k]:

                    Tphi.append(self.tu[k])
                    Xphi.append(self.X[:self.nx, k])

                else:

                    tau = (t_meas_j - self.tu[k]) / hk

                    x_temp = 0

                    for r in range(self.ntauroot + 1):

                        x_temp += self.lfcns[r]([tau])[0] * \
                        self.X[r*self.nx : (r+1) * self.nx, k]

                    Tphi.appaned(self.t_meas_j)
                    Xphi.append(x_temp)

        if self.tu[-1] in self.ty:

            Tphi.append(self.tu[-1])
            Uphi.append(self.uN[:,-1])
            Xphi.append(self.XF)
            WUphi.append(self.WU[:,-1])

        # Mapped calculation of the collocation equations

        # Collocation nodes

        hc = ca.MX.sym("hc", 1)
        tc = ca.MX.sym("tc", self.ntauroot)
        xc = ca.MX.sym("xc", self.nx * (self.ntauroot+1))
        wec = ca.MX.sym("wenc", self.nwe * self.ntauroot)
        wuc = ca.MX.sym("wunc", self.nwu * self.ntauroot)

        # collnode1 = ca.horzcat([sum([self.C[r,j] * \
        #     xnode[r*self.nx : (r+1)*self.nx] \
        #     for r in range(self.ntauroot + 1)]) \
        #     for j in range(1, self.ntauroot + 1)])

        coleqn = ca.vertcat([ \

            hc * ffcn([tc[j-1], \
                system.u, \
                xc[j*self.nx : (j+1)*self.nx], \
                wec[(j-1)*self.nwe : j*self.nwe], \
                wuc[(j-1)*self.nwu : j*self.nwu], \
                system.p])[0] - \

            sum([self.C[r,j] * xc[r*self.nx : (r+1)*self.nx] \

                for r in range(self.ntauroot + 1)]) \
                    
                    for j in range(1, self.ntauroot + 1)])

        # collnodefcn = ca.MXFunction("collnodefcn", [xnode], [collnode])
        # collnodefcn.expand()

        coleqnfcn = ca.MXFunction("coleqnfcn", \
            [hc, tc, system.u, xc, wec, wuc, system.p], \
            # [hc, tc, system.u, xc, wuc, system.p], \
            [coleqn])
        coleqnfcn.expand()

        # ipdb.set_trace()

        [g1] = coleqnfcn.map([ \
            np.atleast_2d((self.tu[1:] - self.tu[:-1])), self.T[1:,:], \
            self.uN, self.X, self.WE, self.WU, self.P])
            # self.uN, self.X, self.WU, self.P])

        # out2 = coleqnfcn.map([np.atleast_2d(self.T[:,0]), self.T[:,1:].T, self.uN.T, self.X.T, self.WE.T, self.WU.T, self.P])
        
        # bl = [ca.horzcat(e) for e in self.Vars["X"]]


        # [op] = collnodefcn.map([ca.horzcat(bl)])

        # XP_JKx = ca.horzcat([op])


        # Continuity nodes

        xnext = ca.MX.sym("xnext", self.nx)

        conteqn = xnext - sum([self.D[r] * xc[r*self.nx : (r+1)*self.nx] \
            for r in range(self.ntauroot + 1)])

        conteqnfcn = ca.MXFunction("conteqnfcn", [xnext, xc], [conteqn])
        conteqnfcn.expand()

        # ipdb.set_trace()

        [g2] = conteqnfcn.map([ \
            ca.horzcat([self.X[:self.nx, 1:], self.XF]), self.X])

        # contnodefcn.expand()

        # [op] = contnodefcn.map([ca.horzcat(bl)])

        # XF_Kx = ca.horzcat([op])


        # Collocation variables

        # Tx = [self.T[k,j] \
        #     for k in range(self.nsteps) \
        #     for j in range(1, self.ntauroot + 1)]

        # Ux = [ca.repmat(self.uN[:, k], 1, self.ntauroot) \
        #     for k in range(self.nsteps)]

        # XCx = [self.Vars["X",k,j]  \
        #     for k in range(self.nsteps) \
        #     for j in range(1, self.ntauroot + 1)]

        # XDx = [self.Vars["X",k+1,0] for k in range(self.nsteps-1)]

        # XDx = XDx + [self.Vars["XF"]]

        # WEx = [self.Vars["WE", k, j-1] \
        #     for k in range(self.nsteps) \
        #     for j in range(1, self.ntauroot + 1)]

        # # WEx = sum(self.Vars["WE"], [])

        # WUx = [self.Vars["WU", k, j-1] \
        #     for k in range(self.nsteps) \
        #      for j in range(1, self.ntauroot + 1)]

        # # WUx = sum(self.Vars["WU"], [])

        # HKx = [ca.repmat((self.tu[k + 1] - self.tu[k]), 1, self.ntauroot) \
        #     for k in range(self.nsteps)]
    
        # XDx = ca.horzcat(XDx)
        # HKx = ca.horzcat(HKx)

        # Evaluate

        [self.phiN] = phifcn.map( \
            [ca.horzcat(k) for k in Tphi, Uphi, Xphi, WUphi] + \
            # [ca.horzcat(k) for k in Tphi, Uphi, Xphi] + \
            [self.P])

        # self.phiN = self.phiN[:]

        # [FK] = ffcn.map([ca.horzcat(k) for k in Tx, Ux, XCx, WEx, WUx] + \
        #     [ca.repmat(self.Vars["P"], 1, len(Tx))])

        # ipdb.set_trace()

        # a = ca.MX.sym("a", self.nx)
        # b = ca.MX.sym("b", self.nx)
        # c = ca.MX.sym("c", self.nx)
        # d = a * b - c
        # gfunc = ca.MXFunction("gfunc", [a, b, c], [d])
        # gfunc.expand()

        # [gall] = gfunc.map([ca.repmat(HKx, self.nx, 1), FK ,XP_JKx])


        # self.g.append((ca.repmat(HKx, self.nx, 1) * FK - XP_JKx)[:])

        # self.g.append((XDx - XF_Kx)[:])

        # self.g = ca.vertcat(self.g)

        # ipdb.set_trace()

        # self.g = ca.vertcat([gall[:], (XDx - XF_Kx)[:]])
        self.g = ca.veccat([g1, g2])

        # self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of ExplODE system sucessful.')
