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

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci

import inputchecks

class LSq(object):

    def __discretize_system(self, system, time_points, discretization_method, **kwargs):

        if system.nx == 0 and system.nz == 0:

            self.discretization = NoDiscretization(system, time_points)

        elif system.nx != 0 and system.nz == 0:

            if discretization_method == "collocation":

                self.discretization = ODECollocation( \
                    system, time_points, **kwargs)

            elif discretization_method == "multiple_shooting":

                self.discretization = ODEMultipleShooting( \
                    system, time_points, **kwargs)

            else:

                raise NotImplementedError('''
Unknow discretization method: {0}.
Possible values are "collocation" and "multiple_shooting".
'''.format(str(discretization_method)))

        elif system.nx != 0 and system.nz != 0:

            raise NotImplementedError('''
Support of implicit DAEs is not implemented yet,
but will be in future versions.
''')            


    def __apply_controls_to_equality_constraints(self, udata):

        udata = inputchecks.check_controls_data(self, udata)

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self.discretization.optimization_variables["U"], 
                self.discretization.optimization_variables["X"], 
                self.discretization.optimization_variables["EPS_U"], 
                self.discretization.optimization_variables["EPS_E"], 
                self.discretization.optimization_variables["P"], 

            ])

        optimization_variables_controls_applied = ci.veccat([ \

                udata, 
                self.discretization.optimization_variables["X"], 
                self.discretization.optimization_variables["EPS_U"], 
                self.discretization.optimization_variables["EPS_E"], 
                self.discretization.optimization_variables["P"], 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self.discretization.equality_constraints])

        [self.equality_constraints_controls_applied] = \
            equality_constraints_fcn([optimization_variables_controls_applied])


    def __apply_controls_to_measurements(self, udata):

        udata = inputchecks.check_controls_data(self, udata)

        optimization_variables_for_measurements = ci.veccat([ \

                self.discretization.optimization_variables["U"], 
                self.discretization.optimization_variables["X"], 
                self.discretization.optimization_variables["EPS_U"], 
                self.discretization.optimization_variables["P"], 

            ])

        optimization_variables_controls_applied = ci.veccat([ \

                udata, 
                self.discretization.optimization_variables["X"], 
                self.discretization.optimization_variables["EPS_U"], 
                self.discretization.optimization_variables["P"], 

            ])

        measurements_fcn = ci.mx_function( \
            "measurements_fcn", \
            [optimization_variables_for_measurements], \
            [self.discretization.measurements])

        [self.measurements_controls_applied] = \
            measurements_fcn([optimization_variables_controls_applied])


    def __apply_controls_to_discretization(self, udata):

        self.__apply_controls_to_equality_constraints(udata)
        self.__apply_controls_to_measurements(udata)


    def __set_optimization_variables(self):

        self.optimization_variables = ci.veccat([ \

                self.discretization.optimization_variables["P"],
                self.discretization.optimization_variables["X"],
                self.discretization.optimization_variables["V"],
                self.discretization.optimization_variables["EPS_E"],
                self.discretization.optimization_variables["EPS_U"],

            ])


    def __set_optimization_variables_initials(self, pinit, xinit):

        xinit = inputchecks.check_states_data(self, xinit)
        repretitions_xinit = \
            self.discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self.discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self.discretization.system.nx, \
                Xinit.size() / self.discretization.system.nx)),
            xinit[:, -1],

            ])

        pinit = inputchecks.check_parameter_data(self, pinit)
        Pinit = pinit

        Vinit = np.zeros(self.discretization.optimization_variables["V"].shape)
        EPS_Einit = np.zeros( \
            self.discretization.optimization_variables["EPS_E"].shape)
        EPS_Uinit = np.zeros( \
            self.discretization.optimization_variables["EPS_U"].shape)

        self.optimization_variables_initials = ci.veccat([ \

                Pinit,
                Xinit,
                Vinit,
                EPS_Einit,
                EPS_Uinit,

            ])


    def __set_measurement_data(self, ydata):

        measurement_data = inputchecks.check_measurement_data(self, ydata)
        self.measurement_data_vectorized = ci.vec(measurement_data)


    def __set_weightings(self, wv, weps_e, weps_u):

        measurement_weightings = \
            inputchecks.check_measurement_weightings(self, wv)

        equation_error_weightings = \
            inputchecks.check_equation_error_weightings(self, weps_e)

        input_error_weightings = \
            inputchecks.check_input_error_weightings(self, weps_u)

        self.weightings_vectorized = ci.veccat([ \

            measurement_weightings,
            equation_error_weightings,
            input_error_weightings, 

            ])


    def __set_measurement_deviations(self):

        self.measurement_deviations = ci.vertcat([ \

                ci.vec(self.measurements_controls_applied) - \
                self.measurement_data_vectorized + \
                ci.vec(self.discretization.optimization_variables["V"])

            ])


    def __set_R(self):

        self.R = ci.sqrt(self.weightings_vectorized) * \
            ci.veccat([ \

                self.discretization.optimization_variables["V"],
                self.discretization.optimization_variables["EPS_E"],
                self.discretization.optimization_variables["EPS_U"],

            ])


    def __set_g(self):

        self.g = ci.vertcat([ \

                self.measurement_deviations,
                self.equality_constraints_controls_applied,

            ])


    def __init__(self, system, time_points, \
        udata = None, ydata = None, \
        pinit = None, xinit = None, \
        wv = None, weps_e = None, weps_u = None, \
        discretization_method = "collocation", **kwargs):

        self.__discretize_system( \
            system, time_points, discretization_method, **kwargs)

        self.__apply_controls_to_discretization(udata)

        self.__set_optimization_variables()

        self.__set_optimization_variables_initials(pinit, xinit)

        self.__set_measurement_data(ydata)

        self.__set_weightings(wv, weps_e, weps_u)

        self.__set_measurement_deviations()

        self.__set_R()

        self.__set_g()

        nlp = ci.mx_function("nlp", ci.nlpIn(x = self.optimization_variables), \
            ci.nlpOut(f = 0.5 * ci.mul([self.R.T, self.R]), g = self.g))

        solver = ci.NlpSolver("solver", "ipopt", nlp, options = {})

        self.sol = solver(x0 = self.optimization_variables_initials, lbg = 0, ubg = 0)