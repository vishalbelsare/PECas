PECas 0.4
=========

Parameter estimation using CasADi
---------------------------------

|travis| |coverall| |rtd|

.. |travis| image:: https://travis-ci.org/adbuerger/PECas.svg?branch=master
    :target: https://travis-ci.org/adbuerger/PECas
    :alt: Travis CI build status master branch

.. |coverall| image:: https://coveralls.io/repos/adbuerger/PECas/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/adbuerger/PECas?branch=master
    :alt: Coverage Status

.. |rtd| image:: https://readthedocs.org/projects/pecas/badge/?version=latest
    :target: http://pecas.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

PECas holds a user-friendly environment for solving parameter estimation
problems and for interpretation of the results recieved. It does so by
providing Python classes that can be initialized with the problem
specifications, while the computations can then easily be performed by
using the available class functions.

As it's name suggests, PECas makes use of the optimization framework
`CasADi <http://casadi.org>`_ to solve parameter estimation
problems. For PECas to work, you need CasADi version >= 2.4.0-rc2 to be
installed on your system.

**Please note:** PECas is still in it's testing state, and does not yet
contain all the features it will provide in future versions. Therefore,
you should check for updates on a regular basis.

For an installation guide, a tutorial on how to use PECas and
a detailed documentation, please
visit `the manual pages <http://pecas.readthedocs.org/>`_ .
