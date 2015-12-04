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

import numpy as np
from abc import ABCMeta, abstractmethod

import time

from ..interfaces import casadi_interface as ci
from .. import intro
import ipdb

import time

class SetupsBaseClass(object):

    __metaclass__ = ABCMeta

    class DiscretizationSettings(object):

        def __init__(self, \
            discretization_method = None, \
            number_of_collocation_points = 3, \
            collocation_scheme = "radau"):

            self.discretization_method = discretization_method
            self.number_of_collocation_points = number_of_collocation_points
            self.collocation_scheme = collocation_scheme


        def collocation_points(self):

            if self.discretization_method == "collocation":

                if self.number_of_collocation_points and \
                    self.collocation_scheme:

                    return ci.collocation_points( \
                        self.number_of_collocation_points, \
                        self.collocation_scheme)

            else:

                return []


        def collocation_polynomial_degree(self):

            return max(0, len(self.collocation_points()) - 1)


    def set_system(self, system):

        self.system = system


    def set_problem_dimensions_from_system_information(self):

        self.nu = self.system.u.shape[0]
        self.np = self.system.p.shape[0]
        self.nphi = self.system.phi.shape[0]

        try:

            self.nx = self.system.x.shape[0]

        except AttributeError:

            self.nx = 0

        try:

            self.neps_e = self.system.eps_e.shape[0]
            
        except AttributeError:

            self.neps_e = 0

        try:

            self.neps_u = self.system.eps_u.shape[0]

        except AttributeError:

            self.neps_u = 0


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


    def set_number_of_control_intervals(self):

        self.nintervals = self.nu - 1


    def check_and_set_time_points(self, controls, measurements):

        tu = controls["tu"]
        ty = measurements["ty"]

        self.check_and_set_control_time_points_input(tu)
        self.check_and_set_measurement_time_points_input(ty)
        self.set_number_of_control_intervals()


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

            self.udata = ci.dmatrix(0, self.nintervals)


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
                xdata = np.zeros((self.nx, self.nintervals + 1))

            xdata = np.atleast_2d(xdata)

            if xdata.shape == (self.nintervals + 1, self.nx):
                xdata = xdata.T

            if not xdata.shape == (self.nx, self.nintervals + 1):

                raise ValueError( \
                    "State values provided by user have wrong dimension.")

            self.xdata = xdata
            # self.Xinit = ca.repmat(xinit[:,:-1], self.ntauroot+1, 1)
            # self.XFinit = xinit[:,-1]
    
        else:

            self.xdata = ci.dmatrix(0,0)
            # self.Xinit = ci.dmatrix(0, 0)
            # self.XFinit = ci.dmatrix(0, 0)


    def check_and_set_measurement_data(self, ydata):

        if ydata is None:
            ydata = np.zeros((self.nphi, self.ty.size))

        ydata = np.atleast_2d(ydata)

        if ydata.shape == (self.ty.size, self.nphi):
            ydata = ydata.T

        if not ydata.shape == (self.nphi, self.ty.size):

            raise ValueError( \
                "Measurement data provided by user has wrong dimension.")

        self.ydata = ydata


    def check_and_set_measurement_weightings(self, wv):

        if wv is None:
            wv = np.ones(self.ydata.shape)

        wv = np.atleast_2d(wv)

        if wv.shape == self.ydata.T.shape:
            wv = wv.T

        if not wv.shape == self.ydata.shape:

            raise ValueError( \
                "Measurement weightings provided by user have wrong dimension.")

        self.wv = wv


    def check_and_set_equation_error_weightings(self, weps_e):

        if not self.neps_e == 0:

            if weps_e is None:
                weps_e = np.ones(self.neps_e)

            weps_e = np.atleast_1d(np.squeeze(weps_e))

            if not weps_e.shape == (self.neps_e,):

                raise ValueError( \
                    "Equation error weightings provided by user have wrong dimension.")

            self.weps_e = weps_e

        else:

            self.weps_e = ci.dmatrix(0, 0)


    def check_and_set_input_error_weightings(self, weps_u):

        if not self.neps_u == 0:

            if weps_u is None:
                weps_u = np.ones(self.neps_u)

            weps_u = np.atleast_1d(np.squeeze(weps_u))

            if not weps_u.shape == (self.neps_u,):

                raise ValueError( \
                    "Input error weightings provided by user have wrong dimension.")

            self.weps_u = weps_u

        else:

            self.weps_u = ci.dmatrix(0, 0)

    # def set_error_initials_to_zero(self):

    #     self.Vinit = np.zeros(self.V.shape)
    #     self.EPS_Einit = np.zeros(self.EPS_E.shape)
    #     self.EPS_Uinit = np.zeros(self.EPS_U.shape)


    # # @profile
    # def check_and_set_all_inputs_and_initials(self, \
    #     controls, initials):

    #     self.check_and_set_controls_data(controls["uN"])
    #     self.check_and_set_parameter_data(initials["pinit"])
    #     self.check_and_set_states_data(initials["xinit"])

    #     self.set_error_initials_to_zero()


    def set_optimization_variables(self):

        ntauroot = self.discretization_settings.collocation_polynomial_degree()

        self.optimvars = {key: ci.dmatrix(0, self.nintervals) \
            for key in ["P", "V", "X", "EPS_E", "EPS_U", "U"]}

        self.optimvars["P"] = ci.mx_sym("P", self.np)

        self.optimvars["V"] = ci.mx_sym("V", self.nphi, self.nintervals+1)

        if self.nx != 0:

            # Attention! Way of ordering has changed! Consider when
            # reapplying collocation and multiple shooting!
            self.optimvars["X"] = ci.mx_sym("X", \
                self.nx, (ntauroot + 1) * self.nintervals)
        
        if self.neps_e != 0:

            self.optimvars["EPS_E"] = ci.mx_sym("EPS_E", \
                self.neps_e, ntauroot * self.nintervals)

        if self.neps_u != 0:
                
            self.optimvars["EPS_U"] = ci.mx_sym("EPS_U", \
                self.neps_u, ntauroot * self.nintervals)

        if self.nu != 0:

            self.optimvars["U"] = ci.mx_sym("U", \
                self.nu, self.nintervals)


    @abstractmethod
    def __init__(self, system, controls, measurements, discretization_method, \
        number_of_collocation_points, collocation_scheme):

        intro.pecas_intro()
        print('\n' + 24 * '-' + \
            ' PECas system initialization ' + 25 * '-')
        print('\nStart system initialization ...')

        self.set_system

        self.set_problem_dimensions_from_system_information()

        self.check_and_set_time_points(controls = controls, \
            measurements = measurements)
        
        self.discretization_settings = \
            self.DiscretizationSettings( \
                discretization_method = discretization_method, \
                number_of_collocation_points = number_of_collocation_points, \
                collocation_scheme = collocation_scheme)

        self.set_optimization_variables()
