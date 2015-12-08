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

from ..interfaces import casadi_interface as ci
from discretization import Discretization

from .. import inputchecks


class ODECollocation(Discretization):

    def set_collocation_settings(self, number_of_points, scheme):

        self.number_of_points = number_of_points
        self.scheme = scheme

        self.collocation_points = ci.collocation_points( \
            self.number_of_points, self.scheme)
        self.collocation_polynomial_degree = len(self.collocation_points) - 1


    def set_optimization_variables(self):

        self.optimvars = {key: ci.dmatrix(0, self.nintervals) \
            for key in ["P", "V", "X", "EPS_E", "EPS_U", "U"]}

        self.optimvars["P"] = ci.mx_sym("P", self.system.np)

        self.optimvars["V"] = ci.mx_sym("V", self.system.nphi, \
            self.nintervals + 1)

        if self.system.nu != 0:

            self.optimvars["U"] = ci.mx_sym("U", self.system.nu, \
                self.nintervals)

        if self.system.nx != 0:

            # Attention! Way of ordering has changed! Consider when
            # reapplying collocation and multiple shooting!
            self.optimvars["X"] = ci.mx_sym("X", self.system.nx, \
                (self.collocation_polynomial_degree + 1) * self.nintervals + 1)
        
        if self.system.neps_e != 0:

            self.optimvars["EPS_E"] = ci.mx_sym("EPS_E", self.system.neps_e, \
                self.collocation_polynomial_degree * self.nintervals)

        if self.system.neps_u != 0:
                
            self.optimvars["EPS_U"] = ci.mx_sym("EPS_U", self.system.neps_u, \
                self.collocation_polynomial_degree * self.nintervals)


    def compute_collocation_time_points(self):

        self.T = np.zeros((self.collocation_polynomial_degree + 1, \
            self.nintervals))

        for k in range(self.nintervals):

            for j in range(self.collocation_polynomial_degree + 1):

                self.T[j,k] = self.tu[k] + \
                    (self.tu[k+1] - self.tu[k]) * \
                    self.collocation_points[j]


    def compute_collocation_coefficients(self):
    
        # Coefficients of the collocation equation

        self.C = np.zeros((self.collocation_polynomial_degree + 1, \
            self.collocation_polynomial_degree + 1))

        # Coefficients of the continuity equation

        self.D = np.zeros(self.collocation_polynomial_degree + 1)

        # Dimensionless time inside one control interval

        tau = ci.sx_sym("tau")

        # For all collocation points

        for j in range(self.collocation_polynomial_degree + 1):

            # Construct Lagrange polynomials to get the polynomial basis
            # at the collocation point
            
            L = 1
            
            for r in range(self.collocation_polynomial_degree + 1):
            
                if r != j:
            
                    L *= (tau - self.collocation_points[r]) / \
                        (self.collocation_points[j] - \
                            self.collocation_points[r])
    

            lfcn = ci.sx_function("lfcn", [tau],[L])
          
            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation
            
            [self.D[j]] = lfcn([1])

            # Evaluate the time derivative of the polynomial at all 
            # collocation points to get the coefficients of the
            # collocation equation
            
            tfcn = lfcn.tangent()

            for r in range(self.collocation_polynomial_degree + 1):

                self.C[j,r] = tfcn([self.collocation_points[r]])[0]


    def initialize_ode_right_hand_side(self):

        self.ffcn = ci.mx_function("ffcn", \
            [self.system.t, self.system.u, self.system.p, self.system.x, \
            self.system.eps_e, self.system.eps_u], [self.system.f])


    def compute_collocation_nodes(self):

        h = ci.mx_sym("h", 1)

        t = ci.mx_sym("t", self.collocation_polynomial_degree)
        u = self.system.u
        p = self.system.p

        x = ci.mx_sym("x", self.system.nx * \
            (self.collocation_polynomial_degree + 1))
        eps_e = ci.mx_sym("eps_e", \
            self.system.neps_e * self.collocation_polynomial_degree)
        eps_u = ci.mx_sym("eps_u", \
            self.system.neps_u * self.collocation_polynomial_degree)

        collocation_node = ci.vertcat([ \

            h * self.ffcn([ \

                t[j-1], u, p, \

                x[j*self.system.nx : (j+1)*self.system.nx], \
                eps_e[(j-1)*self.system.neps_e : j*self.system.neps_e], \
                eps_u[(j-1)*self.system.neps_u : j*self.system.neps_u]])[0] - \

                sum([self.C[r,j] * x[r*self.system.nx : (r+1)*self.system.nx] \

                    for r in range(self.collocation_polynomial_degree + 1)]) \
                    
                        for j in range(1, self.collocation_polynomial_degree + 1)])


        collocation_node_fcn = ci.mx_function("coleqnfcn", \
            [h, t, u, x, eps_e, eps_u, p], [collocation_node])
        collocation_node_fcn = collocation_node_fcn.expand()

        X = self.optimvars["X"][:, :-1][:].reshape( \
            (self.system.nx * (self.collocation_polynomial_degree + 1), \
            self.nintervals))

        EPS_E = self.optimvars["EPS_E"][:].reshape( \
            (self.system.neps_e * self.collocation_polynomial_degree, \
            self.nintervals))

        EPS_U = self.optimvars["EPS_U"][:].reshape( \
            (self.system.neps_u * self.collocation_polynomial_degree, \
            self.nintervals))

        [self.collocation_nodes] = collocation_node_fcn.map([ \
            np.atleast_2d((self.tu[1:] - self.tu[:-1])), self.T[1:,:], \
            self.optimvars["U"], X, EPS_E, EPS_U, self.optimvars["P"]])


    def compute_continuity_nodes(self):

        x = ci.mx_sym("x", self.system.nx * \
            (self.collocation_polynomial_degree + 1))
        x_next = ci.mx_sym("x_next", self.system.nx)

        continuity_node = x_next - sum([self.D[r] * \
            x[r*self.system.nx : (r+1)*self.system.nx] \
            for r in range(self.collocation_polynomial_degree + 1)])

        continuity_node_fcn = ci.mx_function("continuity_node_fcn", \
            [x_next, x], [continuity_node])
        continuity_node_fcn = continuity_node_fcn.expand()

        # --> TODO!

        import ipdb
        ipdb.set_trace()

        X_NEXT = self.optimvars["X"][:, \
            (self.collocation_polynomial_degree + 1) :: \
            (self.collocation_polynomial_degree + 1)]

        X = self.optimvars["X"][:, :-1][:].reshape( \
            (self.system.nx * (self.collocation_polynomial_degree + 1), \
            self.nintervals))

        [self.continuity_nodes] = continuity_node_fcn.map([X_NEXT, X])


        # <-- TODO!

    def __init__(self, system, number_of_points = 3, scheme = "radau"):

        super(ODECollocation, self).__init__(system)

        self.set_collocation_settings(number_of_points, scheme)


    def discretize(self, tu):

        self.tu = inputchecks.check_and_set_time_points_input(tu)

        self.set_optimization_variables()

        self.compute_collocation_time_points()
        self.compute_collocation_coefficients()

        self.initialize_ode_right_hand_side()
        self.compute_collocation_nodes()
        self.compute_continuity_nodes()
