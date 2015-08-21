How to use PECas
================

The next sections will show you the central concepts and functions of PECas. Application examples can be found in later sections.

General concept
---------------

Since PECas uses CasADi for performing parameter estimations, the user first has to define the considered system using CasADi symbolic variables (of type MX). Afterwards, the symbolic variables (which define states, controls, parameters, etc. of the system) can be brought into connection by creating a PECas system object.

This system object can then be used within a PECas parameter estimation problem to estimate the unknown parameters of the previously defined system for user-provided measurement data, while different weightings can applied.

In a further step, the covariance-matrix for the estimated parameters can be computed to support results interpretation.

Following, the general concept and the several classes and functions are described in more detail. Introductory examples for PECas can be found :ref:`in the next section <examples>`. 

Defining a PECas system object
------------------------------

.. automodule:: pecas.systems
    :members:

Defining and solving a parameter estimation problem
---------------------------------------------------

For now, PECas provides one class for defining and solving parameter estimation problems.

.. automodule:: pecas.pecas
    :members:

.. Utilities for results interpretation
.. ------------------------------------

