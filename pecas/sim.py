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

from interfaces import casadi_interface as ci

import inputchecks

class Simulation(object):

    @property
    def simulation_results(self):

        try:

            return self.__simulation_results

        except AttributeError:

            raise AttributeError('''
A system simulation has to be executed before the simulation results
can be accessed, please run run_system_simulation() first.
''')


    def __generate_simulation_ode(self, pdata):

        p = inputchecks.check_parameter_data(pdata, self.__system.np)

        ode_fcn = ci.mx_function("ode_fcn", \
            [self.__system.t, self.__system.u, self.__system.x, \
            self.__system.eps_e, self.__system.eps_u, self.__system.p], \
            [self.__system.f])

        # Needs to be changes for allowance of explicit time dependecy!

        self.__ode_parameters_applied = ode_fcn([ \
            np.zeros(1), self.__system.u, self.__system.x, \
            np.zeros(self.__system.neps_e), \
            np.zeros(self.__system.neps_u), p])[0]


    def __generate_scaled_dae(self):

        t_scale = ci.mx_sym("t_scale", 1)

        dae_scaled = \
            ci.mx_function("dae_scaled", \
                ci.daeIn(x = self.__system.x, \
                    p = ci.vertcat([t_scale, self.__system.u])), \
                ci.daeOut(ode = t_scale * self.__ode_parameters_applied))

        self.__dae_scaled = dae_scaled.expand()


    def __init__(self, system, pdata):

        self.__system = inputchecks.set_system(system)

        self.__generate_simulation_ode(pdata)
        self.__generate_scaled_dae()


    def __initialize_simulation(self, x0, time_points, udata, \
        integrator_options_user):

        self.__x0 = inputchecks.check_states_data(x0, self.__system.nx, 0)

        time_points = inputchecks.check_time_points_input(time_points)
        number_of_integration_steps = time_points.size - 1
        time_steps = time_points[1:] - time_points[:-1]

        udata = inputchecks.check_controls_data(udata, self.__system.nu, \
            number_of_integration_steps)

        self.__simulation_input = ci.vertcat([np.atleast_2d(time_steps), udata])

        integrator_options = integrator_options_user.copy()
        integrator_options.update({"t0": 0, "tf": 1})
        integrator = ci.Integrator("integrator", "cvodes", \
            self.__dae_scaled, integrator_options)

        self.__simulation = integrator.mapaccum("simulation", \
            number_of_integration_steps)


    def run_system_simulation(self, x0, time_points, udata = None, \
        integrator_options = {}):

        print('\n' + 27 * '-' + \
            ' PECas system simulation ' + 26 * '-')
        print('\nPerforming system simulation, this might take some time ...') 

        self.__initialize_simulation(x0 = x0, time_points = time_points, \
            udata = udata, integrator_options_user = integrator_options)

        self.__simulation_results = ci.horzcat([ \

            self.__x0,
            self.__simulation(x0 = self.__x0, p = self.__simulation_input)["xf"]

            ])

        print("System simulation finished.")
