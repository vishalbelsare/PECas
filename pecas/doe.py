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

'''The module :file:`pecas.pe` can used for parameter estimation applications.
For now, only least squares parameter estimation problems are covered.'''

import numpy as np
import time

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci
from covariance_matrix import setup_covariance_matrix, setup_a_criterion
from intro import pecas_intro
from sim import Simulation

import inputchecks

class DoE(object):

    '''The class :class:`pecas.pe.LSq` is used to set up least squares parameter
    estimation problems for systems defined with the PECas
    :class:`pecas.system.System`
    class, using a given set of user provided control 
    data, measurement data and different kinds of weightings.'''

    @property
    def estimation_results(self):

        try:

            return self.__estimation_results

        except AttributeError:

            raise AttributeError('''
A parameter estimation has to be executed before the estimation results
can be accessed, please run run_parameter_estimation() first.
''')


    @property
    def estimated_parameters(self):

        try:

            return self.__estimation_results["x"][ \
                :self.__discretization.system.np]

        except AttributeError:

            raise AttributeError('''
A parameter estimation has to be executed before the estimated parameters
can be accessed, please run run_parameter_estimation() first.
''')


    @property
    def covariance_matrix(self):

        try:

            return self.__covariance_matrix

        except AttributeError:

            raise AttributeError('''
Covariance matrix for the estimated parameters not yet computed.
Run compute_covariance_matrix() to do so.
''')


    @property
    def standard_deviations(self):

        try:

            return ci.sqrt([abs(var) for var \
                in ci.diag(self.covariance_matrix)])

        except AttributeError:

            raise AttributeError('''
Standard deviations for the estimated parameters not yet computed.
Run compute_covariance_matrix() to do so.
''')


    def __discretize_system(self, system, time_points, discretization_method, \
        **kwargs):

        if system.nx == 0 and system.nz == 0:

            self.__discretization = NoDiscretization(system, time_points)

        elif system.nx != 0 and system.nz == 0:

            if discretization_method == "collocation":

                self.__discretization = ODECollocation( \
                    system, time_points, **kwargs)

            elif discretization_method == "multiple_shooting":

                self.__discretization = ODEMultipleShooting( \
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


    def __apply_parameters_to_equality_constraints(self, pdata):

        udata = inputchecks.check_parameter_data(pdata, \
            self.__discretization.system.np)

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["EPS_E"], 
                self.__discretization.optimization_variables["P"], 

            ])

        optimization_variables_parameters_applied = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["EPS_E"], 
                pdata, 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self.__discretization.equality_constraints])

        [self.__equality_constraints_parameters_applied] = \
            equality_constraints_fcn([optimization_variables_parameters_applied])


    def __apply_parameters_to_measurements(self, pdata):

        udata = inputchecks.check_parameter_data(pdata, \
            self.__discretization.system.np)

        optimization_variables_for_measurements = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["P"], 

            ])

        optimization_variables_parameters_applied = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                pdata, 

            ])

        measurements_fcn = ci.mx_function( \
            "measurements_fcn", \
            [optimization_variables_for_measurements], \
            [self.__discretization.measurements])

        [self.__measurements_parameters_applied] = \
            measurements_fcn([optimization_variables_parameters_applied])


    def __apply_parameters_to_discretization(self, pdata):

        self.__apply_parameters_to_equality_constraints(pdata)
        self.__apply_parameters_to_measurements(pdata)


    def __set_optimization_variables(self):

        self.__optimization_variables = ci.veccat([ \

                self.__discretization.optimization_variables["U"],
                self.__discretization.optimization_variables["X"],

            ])


    def __set_optimization_variables_initials(self, pdata, x0, uinit):

        self.simulation = Simulation(self.__discretization.system, pdata)
        self.simulation.run_system_simulation(x0, \
            self.__discretization.time_points, uinit)
        xinit = self.simulation.simulation_results

        repretitions_xinit = \
            self.__discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self.__discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self.__discretization.system.nx, \
                Xinit.size() / self.__discretization.system.nx)),
            xinit[:, -1],

            ])

        uinit = inputchecks.check_controls_data(uinit, \
            self.__discretization.system.nu, \
            self.__discretization.number_of_intervals)
        Uinit = uinit

        self.__optimization_variables_initials = ci.veccat([ \

                Uinit,
                Xinit,

            ])


    def __set_optimization_variables_lower_bounds(self, umin, xmin):

        umin_user_provided = umin

        umin = inputchecks.check_controls_data(umin, \
            self.__discretization.system.nu, 1)

        if umin_user_provided is None:

            umin = -np.inf * np.ones(umin.shape)

        Umin = ci.repmat(umin, 1, \
            self.__discretization.optimization_variables["U"].shape[1])


        xmin_user_provided = xmin

        xmin = inputchecks.check_states_data(xmin, \
            self.__discretization.system.nx, 0)

        if xmin_user_provided is None:

            xmin = -np.inf * np.ones(xmin.shape)

        Xmin = ci.repmat(xmin, 1, \
            self.__discretization.optimization_variables["X"].shape[1])


        self.__optimization_variables_lower_bounds = ci.veccat([ \

                Umin,
                Xmin,

            ])


    def __set_optimization_variables_upper_bounds(self, umax, xmax):

        umax_user_provided = umax

        umax = inputchecks.check_controls_data(umax, \
            self.__discretization.system.nu, 1)

        if umax_user_provided is None:

            umax = np.inf * np.ones(umax.shape)

        Umax = ci.repmat(umax, 1, \
            self.__discretization.optimization_variables["U"].shape[1])


        xmax_user_provided = xmax

        xmax = inputchecks.check_states_data(xmax, \
            self.__discretization.system.nx, 0)

        if xmax_user_provided is None:

            xmax = np.inf * np.ones(xmax.shape)

        Xmax = ci.repmat(xmax, 1, \
            self.__discretization.optimization_variables["X"].shape[1])


        self.__optimization_variables_upper_bounds = ci.veccat([ \

                Umax,
                Xmax,

            ])


    def __set_measurement_data(self):

        measurement_data = inputchecks.check_measurement_data( \
            self.simulation.simulation_results, \
            self.__discretization.system.nphi, \
            self.__discretization.number_of_intervals + 1)
        self.__measurement_data_vectorized = ci.vec(measurement_data)


    def __set_weightings(self, wv, weps_e, weps_u):

        measurement_weightings = \
            inputchecks.check_measurement_weightings(wv, \
            self.__discretization.system.nphi, \
            self.__discretization.number_of_intervals + 1)

        equation_error_weightings = \
            inputchecks.check_equation_error_weightings(weps_e, \
            self.__discretization.system.neps_e)

        input_error_weightings = \
            inputchecks.check_input_error_weightings(weps_u, \
            self.__discretization.system.neps_u)

        self.__weightings_vectorized = ci.veccat([ \

            measurement_weightings,
            equation_error_weightings,
            input_error_weightings, 

            ])


    def __set_measurement_deviations(self):

        self.__measurement_deviations = ci.vertcat([ \

                ci.vec(self.__measurements_parameters_applied) - \
                self.__measurement_data_vectorized + \
                ci.vec(self.__discretization.optimization_variables["V"])

            ])


    # def __setup_residuals(self):

    #     self.__residuals = ci.sqrt(self.__weightings_vectorized) * \
    #         ci.veccat([ \

    #             self.__discretization.optimization_variables["V"],
    #             self.__discretization.optimization_variables["EPS_E"],
    #             self.__discretization.optimization_variables["EPS_U"],

    #         ])


    def __setup_constraints(self):

        self.__constraints = ci.vertcat([ \

                self.__measurement_deviations,
                self.__equality_constraints_parameters_applied,

            ])


    def __set_cov_matrix_derivative_directions(self):

        # These correspond to the optimization valiables of the parameter
        # estimation problem; the evaluation of the covarianve matrix, though,
        # does not depend on the actual values of V, EPS_E and EPS_U, and with
        # this, the DoE problem does not

        self.__cov_matrix_derivative_directions = ci.veccat([ \

                self.__discretization.optimization_variables["U"],
                self.__discretization.optimization_variables["X"],
                self.__discretization.optimization_variables["V"],
                self.__discretization.optimization_variables["EPS_E"],
                self.__discretization.optimization_variables["EPS_U"],

            ])


    def __setup_objective(self):

        self.__covariance_matrix_symbolic = setup_covariance_matrix( \
                self.__cov_matrix_derivative_directions, \
                self.__weightings_vectorized, \
                self.__constraints, self.__discretization.system.np)

        self.__objective = setup_a_criterion(self.__covariance_matrix_symbolic)


    def __setup_nlp(self):

        self.__nlp = ci.mx_function("nlp", \
            ci.nlpIn(x = self.__optimization_variables), \
            ci.nlpOut(f = self.__objective, \
                g = self.__equality_constraints_parameters_applied))


    def __init__(self, system, time_points, \
        uinit = None, umin = None, umax = None, \
        pdata = None, x0 = None, \
        xmin = None, xmax = None, \
        wv = None, weps_e = None, weps_u = None, \
        discretization_method = "multiple_shooting", **kwargs):

        pecas_intro()

        self.__discretize_system( \
            system, time_points, discretization_method, **kwargs)

        self.__apply_parameters_to_discretization(pdata)

        self.__set_optimization_variables()

        self.__set_optimization_variables_initials(pdata, x0, uinit)

        self.__set_optimization_variables_lower_bounds(umin, xmin)

        self.__set_optimization_variables_upper_bounds(umax, xmax)

        self.__set_measurement_data()

        self.__set_weightings(wv, weps_e, weps_u)

        self.__set_measurement_deviations()

        # self.__setup_residuals()

        self.__set_cov_matrix_derivative_directions()

        self.__setup_constraints()

        self.__setup_objective()

        self.__setup_nlp()


    def run_experimental_design(self, solver_options = {"linear_solver": "ma57"}):

        nlpsolver = ci.NlpSolver("solver", "ipopt", self.__nlp, \
            options = solver_options)

        self.__design_results = \
            nlpsolver(x0 = self.__optimization_variables_initials, \
                lbg = 0, ubg = 0, \
                lbx = self.__optimization_variables_lower_bounds, \
                ubx = self.__optimization_variables_upper_bounds)
