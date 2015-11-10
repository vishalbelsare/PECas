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

'''
PECas provides different classes for defining systems for parameter estimation
problems that can be solved within PECas. According to a
system\'s properties, a suitable class needs to be used:

* :class:`BasicSystem`: non-dynamic, contains an output function and possibly
  equality constraints, possibly dependent on time and/or controls.

* :class:`ExplODE`: dynamic system of explicit ODEs, contains an output
  function but no algebraic equations, possibly dependent on time and/or
  controls, equation errors, input errors.

* :class:`ImplDAE` (not yet implemented!): dynamic system of implicit DAEs,
  possibly dependent on time and/or controls.

All systems also need to depend on unknown parameters that will be estimated.
For more information on the several classes, see their descriptions below.
'''

import casadi as ca

import intro

from abc import ABCMeta, abstractmethod

class PECasSystem:

    __metaclass__ = ABCMeta

    def show_system_information(self, showEquations = False):

        r'''
        :param showEquations: show model equations and measurement functions
        :type showEquations: bool

        This function shows the system type and the dimension of the system
        components. If `showEquations` is set to `True`, also the model
        equations and measurement functions are shown.
        '''
        
        intro.pecas_intro()

        print('\n' + 26 * '-' + \
            ' PECas system information ' + 26 * '-')

        if isinstance(self, BasicSystem):
            
            print('''\The system is a non-dynamic systems with the general 
input-output structure and contrain equations: ''')
            
            print("y = phi(t, u, p), g(t, u, p) = 0 ")
            
            print('''\nWith {0} inputs u, {1} parameters p and {2} outputs phi
            '''.format(self.u.size(),self.p.size(), \
                self.phi.size()))


            if showEquations:
                
                print("\nAnd where phi is defined by: ")
                for i, yi in enumerate(self.phi):         
                    print("y[{0}] = {1}".format(\
                         i, yi))
                         
                print("\nAnd where g is defined by: ")
                for i, gi in enumerate(self.g):              
                    print("g[{0}] = {1}".format(\
                         i, gi))

        elif isinstance(self, ExplODE):

            print('''\nThe system is a dynamic system defined by a set of
explicit ODEs xdot which establish the system state x:
    xdot = f(t, u, x, p, eps_e, eps_u)
and by an output function phi which sets the system measurements:
    y = phi(t, x, p).
''')
            
            
            print('''Particularly, the system has:
    {0} inputs u
    {1} parameters p
    {2} states x
    {3} outputs phi'''.format(self.u.size(),self.p.size(),\
                                self.x.size(), \
                                self.phi.size()))

            if showEquations:
                
                print("\nWhere xdot is defined by: ")
                for i, xi in enumerate(self.f):         
                    print("xdot[{0}] = {1}".format(\
                         i, xi))
                         
                print("\nAnd where phi is defined by: ")
                for i, yi in enumerate(self.phi):              
                    print("y[{0}] = {1}".format(\
                         i, yi))

        else:
            
            raise NotImplementedError('''
This feature of PECas is currently disabled, but will be 
available when the DAE systems are implemented.
''')


class BasicSystem(PECasSystem):

    '''
    :param t: time :math:`t \in \mathbb{R}` (optional)
    :type t: casadi.casadi.MX

    :param u: controls :math:`u \in \mathbb{R}^{n_{u}}` (optional)
    :type u: casadi.casadi.MX

    :param p: unknown parameters :math:`p \in \mathbb{R}^{n_{p}}`
    :type p: casadi.casadi.MX

    :param phi: output function :math:`\phi(t, u, p) = y \in \mathbb{R}^{n_{y}}`
    :type phi: casadi.casadi.MX

    :param g: equality constraints :math:`g(t, u, p) = 0 \in \mathbb{R}^{n_{g}}`
              (optional)
    :type g: casadi.casadi.MX

    :raises: TypeError


    The class :class:`BasicSystem` is used to define non-dynamic
    systems for parameter estimation of the following structure:

    .. math::

        y = \phi(t, u, p)

        0 = g(t, u, p).
        
    '''

    def __init__(self, \
                 t = ca.MX.sym("t", 1), \
                 u = ca.MX.sym("u", 0), \
                 p = None, \
                 phi = None, \
                 g = ca.MX.sym("g", 0)):

        intro.pecas_intro()
        print('\n' + 26 * '-' + \
            ' PECas system definition ' + 27 * '-')
        print('\nStarting definition of BasicSystem system ...')

        if not all(isinstance(arg, ca.casadi.MX) for \
            arg in [t, u, p, phi, g]):

            raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')

        self.t = t
        self.u = u
        self.p = p

        self.phi = phi
        self.g = g

        print('\nDefinition of BasicSystem system sucessful.')


class ExplODE(PECasSystem):

    r'''
    :param t: time :math:`t \in \mathbb{R}` (optional)
    :type t: casadi.casadi.MX

    :param u: controls :math:`u \in \mathbb{R}^{n_{u}}` (optional)
    :type u: casadi.casadi.MX

    :param x: states :math:`x \in \mathbb{R}^{n_{x}}`
    :type x: casadi.casadi.MX

    :param p: unknown parameters :math:`p \in \mathbb{R}^{n_{p}}`
    :type p: casadi.casadi.MX

    :param eps_e: equation errors :math:`\epsilon_{e} \in \mathbb{R}^{n_{\epsilon_{e}}}` (optional)
    :type eps_e: casadi.casadi.MX

    :param eps_u: input errors :math:`\epsilon_{u} \in \mathbb{R}^{n_{\epsilon_{u}}}` (optional)
    :type eps_u: casadi.casadi.MX

    :param phi: output function :math:`\phi(t, u, x, p) = y \in \mathbb{R}^{n_{y}}`
    :type phi: casadi.casadi.MX

    :param f: explicit system of ODEs :math:`f(t, u, x, p, \epsilon_{e}, \epsilon_{u}) = \dot{x} \in \mathbb{R}^{n_{x}}`
    :type f: casadi.casadi.MX

    :raises: TypeError


    The class :class:`ExplODE` is used to define dynamic systems of explicit
    ODEs for parameter estimation of the following structure:

    .. math::

        y & = & \phi(t, u, x, p) \\

        \dot{x}  & = & f(t, u, x, p, \epsilon_{e}, \epsilon_{u}).

    '''

    def __init__(self, \
                 t = ca.MX.sym("t", 1),
                 u = ca.MX.sym("u", 0), \
                 x = None, \
                 p = None, \
                 eps_e = ca.MX.sym("eps_e", 0), \
                 eps_u = ca.MX.sym("eps_u", 0), \
                 phi = None, \
                 f = None):

        intro.pecas_intro()
        print('\n' + 26 * '-' + \
            ' PECas system definition ' + 27 * '-')
        print('\nStarting definition of ExplODE system ...')

        if not all(isinstance(arg, ca.casadi.MX) for \
            arg in [t, u, x, p, eps_e, eps_u, phi, f]):

            raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')

        if ca.dependsOn(f, t):

            raise NotImplementedError('''
Explicit time dependecies of the ODE right hand side are not yet supported in
PECas, but probably will be in future versions.''')

        self.t = t
        self.u = u
        self.x = x
        self.eps_e = eps_e
        self.eps_u = eps_u
        self.p = p

        self.phi = phi
        self.f = f

        print('Definition of ExplODE system sucessful.')


class ImplDAE(PECasSystem):

    '''
    :raises: NotImplementedError


    The class :class:`ImplDAE` will be used to define dynamic systems of 
    implicit DAEs for parameter estimation, but is not supported yet.

    '''

    def __init__(self, \
             t = ca.MX.sym("t", 1),
             u = ca.MX.sym("u", 0), \
             x = None, \
             p = None, \
             eps_e = ca.MX.sym("eps_e", 0), \
             eps_u = ca.MX.sym("eps_u", 0), \
             phi = None, \
             f = None, \
             g = ca.MX.sym("g", 0)):

        raise NotImplementedError( \
            "Support of implicit DAEs is not implemented yet.")
