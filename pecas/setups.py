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

#     def set_error_initials_to_zero(self):

#         self.Vinit = np.zeros(self.V.shape)
#         self.EPS_Einit = np.zeros(self.EPS_E.shape)
#         self.EPS_Uinit = np.zeros(self.EPS_U.shape)


#     # @profile
#     def check_and_set_all_inputs_and_initials(self, \
#         controls, initials):

#         self.check_and_set_controls_data(controls["uN"])
#         self.check_and_set_parameter_data(initials["pinit"])
#         self.check_and_set_states_data(initials["xinit"])

#         self.set_error_initials_to_zero()


#     @abstractmethod
#     def __init__(self, system, controls, measurements):

#         intro.pecas_intro()
#         print('\n' + 24 * '-' + \
#             ' PECas system initialization ' + 25 * '-')
#         print('\nStart system initialization ...')

#         self.system = system

#         self.set_problem_dimensions_from_system_information()

#         self.check_and_set_control_time_points_input(controls["tu"])
#         self.check_and_set_measurement_time_points_input(measurements["ty"])


from setupsbaseclass import SetupsBaseClass

class NDSetup(SetupsBaseClass):

    def __init__(self, system, controls, measurements, weightings, initials):

        self.tstart_setup = time.time()

        SetupsBaseClass.__init__(self, \
            system = system, \
            controls = controls, \
            measurements = measurements, \
            discretization_method = discretization_method, \
            collocation_polynomial_order = collocation_polynomial_order, \
            collocation_scheme = collocation_scheme)

        # self.set_problem_dimensions_from_system_information()

        # self.check_and_set_control_time_points_input(tu)
        # self.nsteps = controls["tu"].shape[0] - 1
        self.ncontrols = self.nintervals + 1

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
        initials, discretization_settings):

        self.tstart_setup = time.time()

        SetupsBaseClass.__init__(self, system = system, \
            controls = controls, measurements = measurements)

        # # self.assure_correct_system_type_for_setup_method( \
        # #     system, systems.ExplODE)
        # # self.set_problem_dimensions_from_system_information()

        # # self.check_and_set_control_time_points_input(tu)
        # # self.check_and_set_measurement_time_points_input(ty)

        # # self.nsteps = controls["tu"].shape[0] - 1
        # self.ncontrols = self.nintervals

        # # self.collocation_settings = collocation_settings
        # # self.tauroot = ca.collocationPoints( \
        # #     self.collocation_settings["order"], \
        # #     self.collocation_settings["scheme"])

        # # # Degree of interpolating polynomial

        # # self.ntauroot = len(self.tauroot) - 1

        # self.check_and_set_all_inputs_and_initials( \
        #     controls = controls, initials = initials)

        # # Set tp the collocation coefficients

        # # Coefficients of the collocation equation

        # self.C = np.zeros((self.ntauroot + 1, self.ntauroot + 1))

        # # Coefficients of the continuity equation

        # self.D = np.zeros(self.ntauroot + 1)

        # # Dimensionless time inside one control interval

        # tau = ca.SX.sym("tau")

        # # Construct the matrix T that contains all collocation time points

        # self.T = np.zeros((self.nsteps, self.ntauroot + 1))

        # for k in range(self.nsteps):

        #     for j in range(self.ntauroot + 1):

        #         self.T[k,j] = self.tu[k] + \
        #             (self.tu[k+1] - self.tu[k]) * self.tauroot[j]

        # self.T = self.T.T

        # # For all collocation points

        # self.lfcns = []

        # for j in range(self.ntauroot + 1):

        #     # Construct Lagrange polynomials to get the polynomial basis
        #     # at the collocation point
            
        #     L = 1
            
        #     for r in range(self.ntauroot + 1):
            
        #         if r != j:
            
        #             L *= (tau - self.tauroot[r]) / \
        #                 (self.tauroot[j] - self.tauroot[r])
            
        #     lfcn = ca.SXFunction("lfcn", [tau],[L])
          
        #     # Evaluate the polynomial at the final time to get the
        #     # coefficients of the continuity equation
            
        #     [self.D[j]] = lfcn([1])

        #     # Evaluate the time derivative of the polynomial at all 
        #     # collocation points to get the coefficients of the
        #     # collocation equation
            
        #     tfcn = lfcn.tangent()

        #     for r in range(self.ntauroot + 1):

        #         self.C[j,r] = tfcn([self.tauroot[r]])[0]

        #     self.lfcns.append(lfcn)


        # Initialize phiN

        self.phiN = []

        # Initialize measurement function

        phifcn = ca.MXFunction("phifcn", \
            [self.system.t, self.system.u, self.system.x, self.system.eps_u, self.system.p], \
            [self.system.phi])

        # Initialzie setup of g

        # self.g = []

        # Initialize ODE right-hand-side

        # ffcn = ca.MXFunction("ffcn", \
        #     [self.system.t, self.system.u, self.system.x, self.system.eps_e, self.system.eps_u, \
        #     self.system.p], [self.system.f])

        # # Collect information for measurement function

        # # Structs to hold variables for later mapped evaluation

        # Tphi = []
        # Uphi = []
        # Xphi = []
        # EPS_Uphi = []

        # for k in range(self.nsteps):

        #     hk = self.tu[k + 1] - self.tu[k]
        #     t_meas = self.ty[np.where(np.logical_and( \
        #         self.ty >= self.tu[k], self.ty < self.tu[k + 1]))]

        #     for t_meas_j in t_meas:

        #         Uphi.append(self.uN[:, k])
        #         EPS_Uphi.append(self.EPS_U[:self.neps_u, k])

        #         if t_meas_j == self.tu[k]:

        #             Tphi.append(self.tu[k])
        #             Xphi.append(self.X[:self.nx, k])

        #         else:

        #             tau = (t_meas_j - self.tu[k]) / hk

        #             x_temp = 0

        #             for r in range(self.ntauroot + 1):

        #                 x_temp += self.lfcns[r]([tau])[0] * \
        #                 self.X[r*self.nx : (r+1) * self.nx, k]

        #             Tphi.append(t_meas_j)
        #             Xphi.append(x_temp)

        # if self.tu[-1] in self.ty:

        #     Tphi.append(self.tu[-1])
        #     Uphi.append(self.uN[:,-1])
        #     Xphi.append(self.XF)
        #     EPS_Uphi.append(self.EPS_U[:self.neps_u,-1])


        # Mapped calculation of the collocation equations

        # Collocation nodes

        # hc = ca.MX.sym("hc", 1)
        # tc = ca.MX.sym("tc", self.ntauroot)
        # xc = ca.MX.sym("xc", self.nx * (self.ntauroot+1))
        # eps_ec = ca.MX.sym("eps_ec", self.neps_e * self.ntauroot)
        # eps_uc = ca.MX.sym("eps_uc", self.neps_u * self.ntauroot)

        # coleqn = ca.vertcat([ \

        #     hc * ffcn([tc[j-1], \
        #         self.system.u, \
        #         xc[j*self.nx : (j+1)*self.nx], \
        #         eps_ec[(j-1)*self.neps_e : j*self.neps_e], \
        #         eps_uc[(j-1)*self.neps_u : j*self.neps_u], \
        #         self.system.p])[0] - \

        #     sum([self.C[r,j] * xc[r*self.nx : (r+1)*self.nx] \

        #         for r in range(self.ntauroot + 1)]) \
                    
        #             for j in range(1, self.ntauroot + 1)])

        # coleqnfcn = ca.MXFunction("coleqnfcn", \
        #     [hc, tc, self.system.u, xc, eps_ec, eps_uc, self.system.p], [coleqn])
        # coleqnfcn = coleqnfcn.expand()

        # [gcol] = coleqnfcn.map([ \
        #     np.atleast_2d((self.tu[1:] - self.tu[:-1])), self.T[1:,:], \
        #     self.uN, self.X, self.EPS_E, self.EPS_U, self.P])


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
