Tutorial on using PECas
=======================

Within the next sections, you will get to know how to initialize a parameter
estimation problem with PECas, and how to perfom the provided operations.

A simple first example
----------------------

In a simple first example, the general usage of PECas is shown. This example can also be found in ``ex1.py`` in the ``examples`` directory of the PECas repository.

First, open a Python console, and import all modules that are necessary for using PECas.

.. code:: python

    import pecas
    import numpy as np
    import casadi as ca   

Then, define the :math:`d` optimization variables, i. e. the parameters that will be estimated. In this example, there will be only one parameter. The variable :math:`x` has to be a column vector of type ``casadi.casadi.MX``.

.. code:: python

    d = 1
    x = ca.MX.sym("x", d)

Afterwards, create the model using the optimization variables in :math:`x`. By doing this, the model :math:`M` automatically becomes of the necessary type ``casadi.casadi.MX``. The variable :math:`M` has also to be a column vector and of size :math:`N`.

.. code:: python

    M = np.array([1./3., 2./3., 3./3., 4./3.]) * x[0]

Define the standard deviations of the measurements, and store them within a column vector :math:`\sigma` of type ``np.ndarray``. Make sure to pass the standard deviations, not the variances of the measurements. The vector has to be of the same size :math:`N` as the model vector :math:`M`.

.. code:: python

    sigma = 0.1 * np.ones(M.shape[0])

Store the measurement data in a variable :math:`Y` of type ``np.ndarray`` that also has to be a column vector and of the same size :math:`N` as the model vector :math:`M` and the vector for the standard deviations :math:`\sigma`.

.. code:: python

    Y = np.array([2.5, 4.1, 6.3, 8.2])
