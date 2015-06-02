#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This modules contains the classes that are used for defining a system for
which a parameter estimation problem can be solved using PECas. According to a
system\'s properties, different classes need to be used:

* :class:``BasicSystem``: non-dynamic, contains an output function and possibly
  equality constraints, possibly dependent on time and/or controls.

* :class:``ExplODE``: dynamic system of explicit ODEs, contains an output
  function but no algebraic equations, possibly dependent on time and/or
  controls.

* :class:``ImplDAE``: dynamic system of implicit DAEs (not yet implemented),
  possibly dependent on time and/or controls.

All systems need also to depend on unknown parameters that will be estimated.
For more information on the several class, see their documentations.
'''

import casadi as ca
import casadi.tools as cat

import intro

class BasicSystem(object):

    '''
    :param t: CasADi symbolic variable for the time :math:``t \in \mathbb{R}``.
    :type t: casadi.casadi.SX
    :param u: CasADi symbolic variable for the controls :math:``u \in
    \mathbb{R}^{n_{u}}``.
    :type u: casadi.casadi.SX
    :param p: CasADi symbolic variable for the unknow parameters :math:``p \in
    \mathbb{R}^{n_{p}}``.
    :type p: casadi.casadi.SX
    :param y: CasADi symbolic variable describing the output function
              :math:``y(t, u, p) \in \mathbb{R}^{n_{y}}``, i. e. the output of
              the system :math:``\phi = y(\dot)`` that can be measured, and
              for which in later process measurement data can be provided.
    :type y: casadi.casadi.SX
    :param g: CasADi symbolic variable describing the equality constraints
              :math:``g(t, u, p) \in \mathbb{R}^{n_{g}}``,
              while .:math:``0 = g(\dot)``.
    :type g: casadi.casadi.SX

    The class :class:``BasicSystem`` is used to define non-dynamic
    systems for parameter estimation of the following structure:

    .. math::

        \phi = y(t, u, p)
        0 = g(t, u, p).
        
    '''

    def __init__(self, \
                 t = ca.MX.sym("t", 1), \
                 u = ca.MX.sym("u", 0), \
                 p = None, \
                 y = None, \
                 g = ca.MX.sym("g", 0)):

        intro.pecas_intro()
        print('\n' + 26 * '-' + \
            ' PECas system definition ' + 27 * '-')
        print('\nStarting definition of BasicSystem system ...')

        if not all(isinstance(arg, ca.casadi.MX) for \
            arg in [t, u, p, y, g]):

            raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')

        self.vars = cat.struct_MX([
                (
                    cat.entry("t", expr = t),
                    cat.entry("u", expr = u),
                    cat.entry("p", expr = p),
                )
            ])

        self.fcn = cat.struct_MX([
                (
                    cat.entry("y", expr = y),
                    cat.entry("g", expr = g)
                )
            ])

        print('\nDefinition of BasicSystem system sucessful.')


class ExplODE(object):

    '''
    :param t: CasADi symbolic variable for the time :math:``t \in \mathbb{R}``.
    :type t: casadi.casadi.SX
    :param u: CasADi symbolic variable for the controls :math:``u \in
    \mathbb{R}^{n_{u}}``.
    :type u: casadi.casadi.SX
    :param x: CasADi symbolic variable for the states :math:``x \in
    \mathbb{R}^{n_{x}}``.
    :type x: casadi.casadi.SX
    :param p: CasADi symbolic variable for the unknow parameters :math:``p \in
    \mathbb{R}^{n_{p}}``.
    :type p: casadi.casadi.SX
    :param w: CasADi symbolic variable for the equation errors :math:``w \in
    \mathbb{R}^{n_{w}}``.
    :type w: casadi.casadi.SX
    :param y: CasADi symbolic variable describing the output function
              :math:``y(t, p) \in \mathbb{R}^{n_{y}}``, i. e. the output of
              the system :math:``\phi = y(\dot)`` that can be measured, and
              for which in later process measurement data can be provided
              (note that in this case, :math:``y`` does *not* depend on
              :math:``u``).
    :type y: casadi.casadi.SX
    :param g: CasADi symbolic variable describing the explicit system of ODEs
              :math:``f(t, u, x, p) \in \mathbb{R}^{n_{x}}``,
              so that .:math:``\dot{x} = f(\dot)``.
    :type g: casadi.casadi.SX

    The class :class:``ExplODE`` is used to define dynamic systems of explicit
    ODEs for parameter estimation of the following structure:

    .. math::

        \phi = y(t, u, x, p)
        \dot{x} = f(t, u, x, p, w)
        
    '''

    def __init__(self, \
                 t = ca.MX.sym("t", 1),
                 u = ca.MX.sym("u", 0), \
                 x = None, \
                 p = None, \
                 w = ca.MX.sym("w", 0), \
                 y = None, \
                 f = None):

        intro.pecas_intro()
        print('\n' + 26 * '-' + \
            ' PECas system definition ' + 27 * '-')
        print('\nStarting definition of ExplODE system ...')

        if not all(isinstance(arg, ca.casadi.MX) for \
            arg in [t, u, x, p, y, f]):

            raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')

        self.vars = cat.struct_MX([
                (
                    cat.entry("t", expr = t),
                    cat.entry("u", expr = u),
                    cat.entry("x", expr = x),
                    cat.entry("w", expr = w),
                    cat.entry("p", expr = p)
                )
            ])

        self.fcn = cat.struct_MX([
                (
                    cat.entry("y", expr = y),
                    cat.entry("f", expr = f)
                )
            ])

        print('Definition of ExplODE system sucessful.')


class ImplDAE(object):

    '''
    The class :class:``ImplDAE`` will be used to define dynamic systems of 
    implicit DAEs for parameter estimation, but is not supported yet.

    '''

    def __init__(self, \
             t = ca.MX.sym("t", 1),
             u = ca.MX.sym("u", 0), \
             x = None, \
             p = None, \
             w = ca.MX.sym("w", 0), \
             y = None, \
             f = None, \
             g = ca.MX.sym("g", 0)):

        raise NotImplementedError( \
            "Support of implicit DAEs is not implemented yet.")
