#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

import casadi as ca
import numpy as np
from abc import ABCMeta, abstractmethod

import time

import systems
import intro
import ipdb

import time

class SetupsBaseClass(object):

    '''The abstract class :class:`SetupsBaseClass` contains the basic
    functionalities of all other classes.'''

    __metaclass__ = ABCMeta


    def check_and_set_time_points_input(self, tp):

        if np.atleast_2d(tp).shape[0] == 1:

            tp = np.squeeze(np.asarray(tp))

        elif np.atleast_2d(tp).shape[1] == 1:

            tp = np.squeeze(np.atleast_2d(tp).T)

        else:

            raise ValueError("Invalid dimension for tp.")

        return tp       


    def check_and_set_control_time_points_input(self, tu):

        try:

            self.tu = self.check_and_set_time_points_input(tu)

        except ValueError:

            raise ValueError("Invalid dimension for tu.")


    def check_and_set_measurement_time_points_input(self, ty):

        if ty is not None:

            try:

                self.ty = self.check_and_set_time_points_input(ty)

            except ValueError:

                raise ValueError("Invalid dimension for ty.")

        else:

            self.ty = self.tu


    def check_and_set_controls_data(self, udata):

        if not self.nu == 0:

            if udata is None:
                udata = np.zeros((self.nu, self.ncontrols))

            udata = np.atleast_2d(udata)

            if udata.shape == (self.ncontrols, self.nu):
                udata = udata.T

            if not udata.shape == (self.nu, self.ncontrols):

                raise ValueError( \
                    "Control values provided by user have wrong dimension.")

            self.udata = udata

        else:

            self.udata = ca.DMatrix(0, self.nsteps)


    def check_and_set_parameter_data(self, pdata):

        if pdata is None:
            pdata = np.zeros(self.np)

        pdata = np.atleast_1d(np.squeeze(pdata))

        if not pdata.shape == (self.np,):

            raise ValueError( \
                "Parameter values provided by user have wrong dimension.")

        self.pdata = pdata


    def check_and_set_states_data(self, xdata):

        if not self.nx == 0:

            if xdata is None:
                xdata = np.zeros((self.nx, self.nsteps + 1))

            xdata = np.atleast_2d(xdata)

            if xdata.shape == (self.nsteps + 1, self.nx):
                xdata = xdata.T

            if not xdata.shape == (self.nx, self.nsteps + 1):

                raise ValueError( \
                    "State values provided by user have wrong dimension.")

            self.xdata = xdata
            # self.Xinit = ca.repmat(xinit[:,:-1], self.ntauroot+1, 1)
            # self.XFinit = xinit[:,-1]
    
        else:

            self.xdata = ca.DMatrix(0,0)
            # self.Xinit = ca.DMatrix(0, 0)
            # self.XFinit = ca.DMatrix(0, 0)



    def set_error_initials_to_zero(self):

        self.Vinit = np.zeros(self.V.shape)
        self.EPS_Einit = np.zeros(self.EPS_E.shape)
        self.EPS_Uinit = np.zeros(self.EPS_U.shape)


    # @profile
    def check_and_set_all_inputs_and_initials(self, \
        controls, initials):

        self.check_and_set_controls_data(controls["uN"])
        self.check_and_set_parameter_data(initials["pinit"])
        self.check_and_set_states_data(initials["xinit"])

        self.set_error_initials_to_zero()


    def set_problem_dimensions_from_system_information(self):

        self.nu = self.system.u.shape[0]
        self.np = self.system.p.shape[0]
        self.nphi = self.system.phi.shape[0]

        try:

            self.nx = self.system.x.shape[0]
            self.neps_e = self.system.eps_e.shape[0]
            self.neps_u = self.system.eps_u.shape[0]

        except AttributeError:

            self.nx = 0
            self.neps_e = 0
            self.neps_u = 0


    def set_optimization_variables(self):

        self.P = ca.MX.sym("P", self.np)

        self.V = ca.MX.sym("V", self.nphi, self.nsteps+1)

        if self.nx != 0:

            self.X = ca.MX.sym("X", (self.nx * (self.ntauroot+1)), self.nsteps)
            self.XF = ca.MX.sym("XF", self.nx)

        else:

            self.X = ca.DMatrix(0, self.nsteps)
            self.XF = ca.DMatrix(0, self.nsteps)
        
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


    @abstractmethod
    def __init__(self, system, controls, measurements):

        intro.pecas_intro()
        print('\n' + 24 * '-' + \
            ' PECas system initialization ' + 25 * '-')
        print('\nStart system initialization ...')

        self.system = system

        self.set_problem_dimensions_from_system_information()

        self.check_and_set_control_time_points_input(controls["tu"])
        self.check_and_set_measurement_time_points_input(measurements["ty"])


class NDSetup(SetupsBaseClass):

    def __init__(self, system, controls, measurements, weightings, initials):

        self.tstart_setup = time.time()

        SetupsBaseClass.__init__(self, system = system, \
            controls = controls, measurements = measurements)

        # self.set_problem_dimensions_from_system_information()

        # self.check_and_set_control_time_points_input(tu)
        self.nsteps = controls["tu"].shape[0] - 1
        self.ncontrols = self.nsteps + 1

        self.set_optimization_variables()
        self.check_and_set_all_inputs_and_initials( \
            controls = controls, initials = initials)

        # set_up_measurement_function()

        # Set up phiN

        self.phiN = []

        phifcn = ca.MXFunction("phifcn", \
            [self.system.t, self.system.u, self.system.p], [self.system.phi])

        for k in range(self.nsteps+1):

            self.phiN.append(phifcn([self.tu[k], \
                self.uN[:, k], self.P])[0])

        self.phiN = ca.vertcat(self.phiN)

        # self.phiNfcn = ca.MXFunction("phiNfcn", [self.Vars], [self.phiN])

        # Set up g

        # TODO! Can/should/must gfcn depend on uN and/or t?

        gfcn = ca.MXFunction("gfcn", [self.system.p], [self.system.g])

        self.g = gfcn.call([self.P])[0]

        self.tend_setup = time.time()
        self.duration_setup = self.tend_setup - self.tstart_setup

        print('Initialization of NonDyn system sucessful.')


class ODESetup(SetupsBaseClass):

    def __init__(self, system, controls, measurements, weightings, \
        initials, collocation_settings):

        self.tstart_setup = time.time()

        SetupsBaseClass.__init__(self, system = system, \
            controls = controls, measurements = measurements)

        # self.assure_correct_system_type_for_setup_method( \
        #     system, systems.ExplODE)
        # self.set_problem_dimensions_from_system_information()

        # self.check_and_set_control_time_points_input(tu)
        # self.check_and_set_measurement_time_points_input(ty)

        self.nsteps = controls["tu"].shape[0] - 1
        self.ncontrols = self.nsteps

        self.collocation_settings = collocation_settings
        self.tauroot = ca.collocationPoints( \
            self.collocation_esttings["order"], \
            self.collocation_settings["scheme"])

        # Degree of interpolating polynomial

        self.ntauroot = len(self.tauroot) - 1

        self.set_optimization_variables()

        self.check_and_set_all_inputs_and_initials( \
            controls = controls, initials = initials)

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
            [self.system.t, self.system.u, self.system.x, self.system.eps_u, self.system.p], \
            [self.system.phi])

        # Initialzie setup of g

        self.g = []

        # Initialize ODE right-hand-side

        ffcn = ca.MXFunction("ffcn", \
            [self.system.t, self.system.u, self.system.x, self.system.eps_e, self.system.eps_u, \
            self.system.p], [self.system.f])

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
                self.system.u, \
                xc[j*self.nx : (j+1)*self.nx], \
                eps_ec[(j-1)*self.neps_e : j*self.neps_e], \
                eps_uc[(j-1)*self.neps_u : j*self.neps_u], \
                self.system.p])[0] - \

            sum([self.C[r,j] * xc[r*self.nx : (r+1)*self.nx] \

                for r in range(self.ntauroot + 1)]) \
                    
                    for j in range(1, self.ntauroot + 1)])

        coleqnfcn = ca.MXFunction("coleqnfcn", \
            [hc, tc, self.system.u, xc, eps_ec, eps_uc, self.system.p], [coleqn])
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
