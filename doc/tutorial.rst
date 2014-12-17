Tutorial on using PECas
=======================

Within the next sections, you will get to know how to initialize a parameter
estimation problem with PECas, and how to perfom the provided operations.

A simple first example
----------------------

In a simple first example, the general usage of PECas is shown. This example can also be found in ``ex1.py`` in the ``examples`` directory of the `PECas repository <https://github.com/adbuerger/PECas>`_.

The task for this example is to estimate a single parameter :math:`x \in \mathbb{R}` using :math:`N = 4` measurements with identical standard deviations :math:`\sigma_{i} = 0.1,\,i = 1, \dotsc, N`. The model :math:`M` and the corresponding measurements are described as follows:

.. math::

    M = \begin{pmatrix}
            {\frac{1}{3} x} \\
            {\frac{2}{3} x} \\
            {\frac{3}{3} x} \\
            {\frac{4}{3} x} \\
        \end{pmatrix},~
    Y = \begin{pmatrix}
            {2,5} \\
            {4,1} \\
            {6,3} \\
            {8,2} \\
        \end{pmatrix}.

First, open a Python console, and import all modules that are necessary for using PECas.

.. code:: python

    >>> import pecas
    >>> import numpy as np
    >>> import casadi as ca   

Then, define the ``d`` optimization variables, i. e. the parameters that will be estimated. Here, there will be only one parameter. The variable ``x`` has to be a column vector of type ``casadi.casadi.MX``.

.. code:: python

    >>> d = 1
    >>> x = ca.MX.sym("x", d)

Afterwards, create the model using the optimization variables in ``x``. By doing this, the model ``M`` automatically becomes of the necessary type ``casadi.casadi.MX``. The variable ``M`` has also to be a column vector and of size ``N``.

.. code:: python

    >>> M = np.array([1./3., 2./3., 3./3., 4./3.]) * x[0]

Define the standard deviations of the measurements, and store them within a column vector ``sigma`` of type ``np.ndarray``. Make sure to pass the standard deviations, not the variances of the measurements. The vector has to be of the same size ``N`` as the model vector ``M``.

.. code:: python

    >>> sigma = 0.1 * np.ones(M.shape[0])

Store the measurement data in a variable ``Y`` of type ``np.ndarray`` that also has to be a column vector and of the same size ``N`` as the model vector ``M`` and the vector for the standard deviations ``sigma``.

.. code:: python

    >>> Y = np.array([2.5, 4.1, 6.3, 8.2])

Now, create an instance of the class PECasProb to define the parameter estimation problem within PECas. While ``x``, ``M`` and ``sigma``, which are the first three arguments, are mandatory, the other variables are either optional or mutually substitutable, and therefor need to be adressed.

.. code:: python

    >>> pep = pecas.PECasProb(x, M, sigma, Y = Y)


With the problem set up, you can now perform a least squares parameter estimation for the problem. You should then see the outputs of IPOPT, the solver that is used for solving the optimization problems.

.. code:: python

    >>> pep.run_parameter_estimation()

    This is Ipopt version 3.11.8, running with linear solver mumps.
    NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

    Number of nonzeros in equality constraint Jacobian...:        0
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:        1

    Total number of variables............................:        1
                         variables with only lower bounds:        0
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0

    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  1.2999000e+03 0.00e+00 1.00e+02  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  1.9800000e+00 0.00e+00 1.41e-14  -1.0 6.24e+00    -  1.00e+00 1.00e+00f  1

    Number of Iterations....: 1

                                       (scaled)                 (unscaled)
    Objective...............:   4.7596153846153877e-01    1.9800000000000009e+00
    Dual infeasibility......:   1.4091292235626991e-14    5.8619775700208278e-14
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   1.4091292235626991e-14    5.8619775700208278e-14


    Number of objective function evaluations             = 2
    Number of objective gradient evaluations             = 2
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 1
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.001
    Total CPU secs in NLP function evaluations           =      0.000

    EXIT: Optimal Solution Found.
    time spent in eval_f: 1e-05 s. (2 calls, 0.005 ms. average)
    time spent in eval_grad_f: 3.6e-05 s. (3 calls, 0.012 ms. average)
    time spent in eval_g: 0 s.
    time spent in eval_jac_g: 0 s.
    time spent in eval_h: 1.8e-05 s. (2 calls, 0.009 ms. average)
    time spent in main loop: 0.001633 s.
    time spent in callback function: 0 s.
    time spent in callback preparation: 4e-06 s.

IPOPT stopped with the message "``EXIT: Optimal Solution Found``", so the solver converged. You can then compute the covariance matrix and display the results of the parameter estimation. You should then see the results of the parameter estimation, i. e. ``beta``, the residual ``Rhat``, the estimated parameters and their standard deviations and the covariance matrix (which in this case has only one entry).

.. code:: python

    >>> pep.compute_covariance_matrix()
    >>> pep.print_results()

    ## Begin of parameter estimation results ## 

    Factor beta and residual Rhat:

    beta = 0.66
    Rhat = 1.98


    Estimated parameters xi:

    x0   = 6.24       +/- 0.140712  


    Covariance matrix Cov(x):

    [['1.98000e-02']]


    ##  End of parameter estimation results  ## 
