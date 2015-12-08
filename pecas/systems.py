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

from interfaces import casadi_interface as ci

from intro import pecas_intro
from discretization.odecollocation import ODECollocation

class System:

    def check_all_system_parts_are_casadi_symbolics(self):

        for arg in self.__init__.__code__.co_varnames[1:]:

                if not isinstance(getattr(self, arg), type(ci.mx_sym("a"))):

                    raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')


    def check_no_explicit_time_dependecy(self):

        if ci.depends_on(self.f, self.t):

            raise NotImplementedError('''
Explicit time dependecies of the ODE right hand side are not yet supported in
PECas, but will be in future versions.''')


    def system_validation(self):

        self.check_all_system_parts_are_casadi_symbolics()
        self.check_no_explicit_time_dependecy()

        if self.nx == 0 and self.nz == 0:

            self.print_nondyn_system_information()
            self.discretization = NoDiscretization()

        elif self.nx != 0 and self.nz == 0:

            self.print_ode_system_information()
            self.discretization = ODECollocation(self)

        elif self.nx != 0 and self.nz != 0:

            raise NotImplementedError('''
Support of implicit DAEs is not implemented yet,
but will be in future versions.
''')

        else:

            raise NotImplementedError('''
The system definition provided by the user is invalid.
See the documentation for a list of valid definitions.
''')


    def print_nondyn_system_information(self):

        print('''
The system is a non-dynamic systems with the general input-output
structure and equality constraints:

y = phi(t, u, p),
g(t, u, p) = 0.

Particularly, the system has:
{0} inputs u
{1} parameters p
{2} outputs phi'''.format(self.nu,self.nz, self.nphi))
        
        print("\nwhere phi is defined by ")
        for i, yi in enumerate(self.phi):         
            print("y[{0}] = {1}".format(i, yi))
                 
        print("\nand where g is defined by ")
        for i, gi in enumerate(self.g):              
            print("g[{0}] = {1}".format(i, gi))


    def print_ode_system_information(self):

        print('''
The system is a dynamic system defined by a set of explicit ODEs xdot
which establish the system state x and by an output function phi which
sets the system measurements:

xdot = f(t, u, x, p, eps_e, eps_u),
y = phi(t, x, p).

Particularly, the system has:
{0} inputs u
{1} parameters p
{2} states x
{3} outputs phi'''.format(self.nu,self.np, self.nx, self.nphi))

        
        print("\nwhere xdot is defined by ")
        for i, xi in enumerate(self.f):         
            print("xdot[{0}] = {1}".format(i, xi))
                 
        print("\nand where phi is defined by ")
        for i, yi in enumerate(self.phi):              
            print("y[{0}] = {1}".format(i, yi))


    @property
    def nu(self):

        return self.u.size()


    @property
    def np(self):

        return self.p.size()


    @property
    def nx(self):

        return self.x.size()


    @property
    def nz(self):

        return self.z.size()

    @property
    def neps_e(self):

        return self.eps_e.size()


    @property
    def neps_u(self):

        return self.eps_u.size()

    @property
    def nphi(self):

        return self.phi.size()


    def __init__(self, \
             t = ci.mx_sym("t", 1),
             u = ci.mx_sym("u", 0), \
             p = None, \
             x = ci.mx_sym("u", 0), \
             z = ci.mx_sym("u", 0),
             eps_e = ci.mx_sym("eps_e", 0), \
             eps_u = ci.mx_sym("eps_u", 0), \
             phi = None, \
             f = ci.mx_sym("u", 0), \
             g = ci.mx_sym("g", 0)):


        r'''
        :param t: time :math:`t \in \mathbb{R}` (not yet supported!)
        :type t: casadi.casadi.MX

        :param u: controls :math:`u \in \mathbb{R}^{n_{u}}` (optional)
        :type u: casadi.casadi.MX

        :param p: unknown parameters :math:`p \in \mathbb{R}^{n_{p}}`
        :type p: casadi.casadi.MX

        :param x: differential states :math:`x \in \mathbb{R}^{n_{x}}` (optional)
        :type x: casadi.casadi.MX

        :param z: algebraic states :math:`x \in \mathbb{R}^{n_{z}}` (optional)
        :type z: casadi.casadi.MX

        :param eps_e: equation errors :math:`\epsilon_{e} \in \mathbb{R}^{n_{\epsilon_{e}}}` (optional)
        :type eps_e: casadi.casadi.MX

        :param eps_u: input errors :math:`\epsilon_{u} \in \mathbb{R}^{n_{\epsilon_{u}}}` (optional)
        :type eps_u: casadi.casadi.MX

        :param phi: output function :math:`\phi(t, u, x, p) = y \in \mathbb{R}^{n_{y}}`
        :type phi: casadi.casadi.MX

        :param f: explicit system of ODEs :math:`f(t, u, x, z, p, \epsilon_{e}, \epsilon_{u}) = \dot{x} \in \mathbb{R}^{n_{x}}` (optional)
        :type f: casadi.casadi.MX

        :param g: equality constraints :math:`g(t, u, x, z, p) = 0 \in \mathbb{R}^{n_{g}}` (optional)
                  (optional)
        :type g: casadi.casadi.MX

        :raises: TypeError


        The class :class:`System` is used to define non-dynamic, explicit ODE-
        or fully implicit DAE-systems systems within PECas. Depending on the
        inputs the user provides, :class:`System` is interpreted as follows:


        **Non-dynamic system** (x = None, z = None):

        .. math::

            y = \phi(t, u, p)

            0 = g(t, u, p).


        **ODE system** (x != None, z = None):

        .. math::

            y & = & \phi(t, u, x, p) \\

            \dot{x}  & = & f(t, u, x, p, \epsilon_{e}, \epsilon_{u}).


        **DAE system** (x != None, z != None) (not yet supported!):

        .. math::

            y & = & \phi(t, u, x, p) \\

            \dot{x}  & = & f(t, u, x, z, p, \epsilon_{e}, \epsilon_{u}).

            0 = g(t, u, x, z, p)

        '''


        pecas_intro()
        
        print('\n' + 26 * '-' + \
            ' PECas system definition ' + 27 * '-')
        print('\nStarting system definition ...')

        self.t = t
        self.u = u
        self.p = p

        self.x = x
        self.z = z

        self.eps_e = eps_e
        self.eps_u = eps_u

        self.phi = phi
        self.f = f
        self.g = g

        self.system_validation()
