Tutorial on using PECas
=======================

Within the next sections, you will get to know how to initialize a parameter
estimation problem with PECas, and how to perform the provided operations. For a detailed description of the used functions and commands (and many more), see :doc:`tool`.

General usage of PECas (``examples/ex1.py``)
--------------------------------------------

In this first example, the general usage of PECas is shown. This example can be found in ``ex1.py`` in the ``examples`` directory of the `PECas repository <https://github.com/adbuerger/PECas>`_.

The task for this example is to estimate a single parameter :math:`x \in \mathbb{R}` using :math:`N = 4` measurements with identical standard deviations :math:`\sigma_{i} = 0.1,\,i = 1, \dotsc, N`. The model :math:`M` and the corresponding measurements are described as follows:

.. math::

    M = \begin{pmatrix}
            {\frac{x}{3}} \\
            {\frac{2x}{3}} \\
            {x} \\
            {\frac{4x}{3}}
        \end{pmatrix},~
    Y = \begin{pmatrix}
            {2,5} \\
            {4,1} \\
            {6,3} \\
            {8,2}
        \end{pmatrix}.

First, open a Python console (e. g. `iPython <http://ipython.org/>`_) or development environment (e. g. `Spyder <https://code.google.com/p/spyderlib/>`_), and import all modules that are necessary for using PECas.

.. code:: python

    >>> import pecas
    >>> import numpy as np
    >>> import casadi as ca   

Then, define the ``d`` optimization variables, i. e. the parameters that will be estimated. Here, there will be only one parameter. The variable ``x`` has to be a column vector of type ``casadi.casadi_core.SX``.

.. code:: python

    >>> d = 1
    >>> x = ca.SX.sym("x", d)

Afterwards, create the model using the optimization variables in ``x``. By doing this, the model ``M`` automatically becomes of the necessary type ``casadi.casadi_core.SX``. The variable ``M`` has also to be a column vector and of size ``N``.

.. code:: python

    >>> M = np.array([1., 2., 3., 4.]) / 3. * x[0]

Define the standard deviations of the measurements, and store them within a column vector ``sigma`` of type ``numpy.ndarray``. Make sure to pass the standard deviations, not the variances of the measurements. The vector has to be of the same size ``N`` as the model vector ``M``.

.. code:: python

    >>> sigma = 0.1 * np.ones(M.shape[0])

Store the measurement data in a variable ``Y`` of type ``numpy.ndarray`` that also has to be a column vector and of the same size ``N`` as the model vector ``M`` and the vector for the standard deviations ``sigma``.

.. code:: python

    >>> Y = np.array([2.5, 4.1, 6.3, 8.2])

Now, create an instance ``pep`` of the class PECasLSq to define the parameter estimation problem within PECas. While ``x``, ``M`` and ``sigma``, which are the first three arguments, are mandatory, the other variables are either optional or mutually substitutable, and therefore need to be addressed.

.. code:: python

    >>> pep = pecas.PECasLSq(x, M, sigma, Y = Y)


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



Using equality constraints and initial guesses (``examples/ex2.py``)
--------------------------------------------------------------------

In this example, the usage of equality constraints and an initial guess in PECas is shown. This example can be found in ``ex2.py`` in the ``examples`` directory of the `PECas repository <https://github.com/adbuerger/PECas>`_.

The task for this example is to estimate :math:`d = 2` parameters :math:`x \in \mathbb{R}^{d}` using :math:`N = 4` measurements with identical standard deviations :math:`\sigma_{i} = 0.5,\,i = 1, \dotsc, N`. The model :math:`M` and the corresponding measurements are described as follows:

.. math::

    M = \begin{pmatrix}
            {x_{1} + x_{2}^{2}} \\
            {x_{1} + 2 x_{2}^{2}} \\
            {x_{1} + 3 x_{2}^{2}} \\
            {x_{1} + 4 x_{2}^{2}}
        \end{pmatrix},~
    Y = \begin{pmatrix}
            {2,23947} \\
            {2,84568} \\
            {4,55041} \\
            {5,08583}
        \end{pmatrix}.

In addition to this, there exist :math:`m = 1` equality constraints :math:`G \in (0)^{m}` and an initial guess for the solution :math:`x_{init} \in \mathbb{R}^{d}`, which are described as follows:

.. math::

    G = \begin{pmatrix}
            {2 - \|x\|_{2}^{2}}
        \end{pmatrix},~
    x_{init} = \begin{pmatrix}
            {1} \\
            {1}
        \end{pmatrix}.

Again, import the necessary modules, and define the optimization variables ``x``.

.. code:: python

    >>> import pecas
    >>> import numpy as np
    >>> import casadi as ca 
    >>> d = 2
    >>> x = ca.SX.sym("x", d)

Then, define the model ``M`` from the descriptions above, e. g. by using the CasADi commands for matrix multiplication ``casadi.mul()`` and vertical concatenation ``casadi.vertcat()``.

.. code:: python

    >>> M = ca.mul(np.matrix([np.ones(4), range(1,5)]).T, ca.vertcat((x[0], x[1]**2)))

Define the column vector ``G`` of type ``casadi.casadi_core.SX`` for the equality constraints using the optimization variables ``x``.

.. code:: python

    >>> G = 2 - ca.mul(x.T,x)

The initial guess ``xinit`` has to be defined as a column vector of type ``numpy.ndarray``.

.. code:: python

    >>> xinit = np.array([1, 1])

Finally, again create the vectors for the measurements ``Y`` and the standard deviations ``sigma``.

.. code:: python

    >>> Y = np.array([2.23947, 2.84568, 4.55041, 5.08583])
    >>> sigma = 0.5 * np.ones(M.shape[0])

Now, an instance ``pep`` of the class PECasLSq can be created by also addressing the equality constraints and the initial guess, and perform the least squares estimation.


.. code:: python

    >>> pep = pecas.PECasLSq(x, M, sigma, Y = Y, G = G, xinit = xinit)
    >>> pep.run_parameter_estimation()

    This is Ipopt version 3.11.8, running with linear solver mumps.
    NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

    Number of nonzeros in equality constraint Jacobian...:        2
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:        3

    Total number of variables............................:        2
                         variables with only lower bounds:        0
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        1
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0

    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  7.8295700e-01 0.00e+00 6.26e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  5.4949144e-01 2.73e-03 5.79e-01  -1.0 3.69e-02    -  1.00e+00 1.00e+00f  1
       2  5.5015562e-01 1.33e-06 3.73e-04  -1.7 1.12e-03    -  1.00e+00 1.00e+00h  1
       3  5.5015597e-01 2.88e-12 6.72e-10  -5.7 1.52e-06    -  1.00e+00 1.00e+00h  1
       4  5.5015597e-01 0.00e+00 4.60e-14 -11.0 3.93e-12    -  1.00e+00 1.00e+00h  1

    Number of Iterations....: 4

                                       (scaled)                 (unscaled)
    Objective...............:   5.5015597053016108e-01    5.5015597053016108e-01
    Dual infeasibility......:   4.5963233219481481e-14    4.5963233219481481e-14
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   4.5963233219481481e-14    4.5963233219481481e-14


    Number of objective function evaluations             = 5
    Number of objective gradient evaluations             = 5
    Number of equality constraint evaluations            = 5
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 5
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 4
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.001
    Total CPU secs in NLP function evaluations           =      0.000

    EXIT: Optimal Solution Found.
    time spent in eval_f: 1.1e-05 s. (5 calls, 0.0022 ms. average)
    time spent in eval_grad_f: 3e-05 s. (6 calls, 0.005 ms. average)
    time spent in eval_g: 2.8e-05 s. (5 calls, 0.0056 ms. average)
    time spent in eval_jac_g: 2.1e-05 s. (7 calls, 0.003 ms. average)
    time spent in eval_h: 4.7e-05 s. (5 calls, 0.0094 ms. average)
    time spent in main loop: 0.001539 s.
    time spent in callback function: 0 s.
    time spent in callback preparation: 6e-06 s.


After the solver converged, the covariance matrix can be computed, and the results of the parameter estimation can be displayed.

.. code:: python

    >>> pep.compute_covariance_matrix()
    >>> pep.print_results()

    ## Begin of parameter estimation results ## 

    Factor beta and residual Rhat:

    beta = 0.183385
    Rhat = 0.550156


    Estimated parameters xi:

    x0   = 0.961943   +/- 0.0346066 
    x1   = 1.03666    +/- 0.0321124 


    Covariance matrix Cov(x):

    [['1.19762e-03' '-1.11130e-03']
     ['-1.11130e-03' '1.03120e-03']]


    ##  End of parameter estimation results  ## 


Generating pseudo measurement data and using ``get``-methods (``examples/ex3.py``)
----------------------------------------------------------------------------------

In this example, the generation of pseudo measurement data in PECas and usage of the ``get``-methods is shown. This example can be found in ``ex3.py`` in the ``examples`` directory of the `PECas repository <https://github.com/adbuerger/PECas>`_.

The task for this example is to generate pseudo measurement data for the model from `the previous example <file:///mnt/data/Dokumente/imtek/highwind/PECas-doc/html/tutorial.html#using-equality-constraints-and-initial-guesses-examples-ex2-py>`_ and return the data for further usage. The data is created from the given model :math:`M`, the standard deviations :math:`\sigma_{i},\,i = 1, \dotsc, N` and some true values for the parameters :math:`x_{true}` that are assumed to be known, which in this case will be defined as

.. math::

    x_{true} = \begin{pmatrix} {1} \\ {1} \end{pmatrix}.

For a more detailed description of the performed computations, see :doc:`tool`.

The intention why one might want to do this is to generate pseudo measurement data that can either be used for testing the parameter estimation properties of the given setup (provided that the true parameter values :math:`x_{true}` can be supported), testing of the software itself, or to generate data for a given model that can later be used e. g. within exercises and lessons on parameter estimation.

First, define the several components of the parameter estimation problem just as before, but without specifying the measurement values.

.. code:: python

    >>> import pecas
    >>> import numpy as np
    >>> import casadi as ca 
    >>> d = 2
    >>> x = ca.SX.sym("x", d)   
    >>> M = ca.mul(np.matrix([np.ones(4), range(1,5)]).T, ca.vertcat((x[0], x[1]**2)))
    >>> G = 2 - ca.mul(x.T,x)
    >>> sigma = 0.5 * np.ones(M.shape[0])
    >>> xinit = np.array([1, 1])

Then, define a column vector ``xtrue`` of type ``numpy.ndarray`` for the true values of the parameters, and create an instance ``pep`` of the class PECasLSq providing all information.

.. code:: python

    >>> xtrue = np.array([1, 1])
    >>> pep = pecas.PECasLSq(x, M, sigma, xtrue = xtrue, G = G, xinit = xinit)

Afterwards, the class instance ``pep`` can be used to generate pseudo measurement data that will be stored as measurement data inside the object.

.. code::

    >>> pep.generate_pseudo_measurement_data()

The generated data can be returned using the ``get``-method for the measurement data, and then e. g. be stored in another variable ``meas_data`` for further usage.

.. code::

    >>> meas_data = pep.get_Y()

For an overview of the other available ``get``-methods, see again :doc:`tool`.
