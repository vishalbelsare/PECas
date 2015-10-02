Concept test: covariance matrix computation
===========================================

Simulate system. Then: add random noise, estimate, store estimated parameter, repeat. Afterwards, compute standard deviation of estimated parameters, and compare to single covariance matrix computation done in PECas.

ODE, 2 states, 1 control, 4 params, (silverbox example)
-------------------------------------------------------

.. code-block:: python

    y_test = lsqpe_sim.Xsim + 0.1 * (pl.rand(*lsqpe_sim.Xsim.shape) - 0.5)


**100 repetitions**

.. code-block:: python

    p_mean = [5.62495101, 2.29884392, 1.00000177, 4.69072852]
    p_est_single = [ 5.62465644, 2.29883142, 0.99996443, 4.68878935]

    p_std = [3.47177266e-04, 1.72163185e-04, 7.47112900e-05, 2.67881473e-03]
    p_std_covmat = [0.000240307, 0.000106346, 4.85647e-05, 0.00208527]


**200 repetitions**

.. code-block:: python

    p_mean = [ 5.6249788   2.29885709  1.00000682  4.69085254]
    p_est_single = [5.62481045, 2.29903087, 1.00001874, 4.68705539]

    p_std = [4.02467180e-04, 1.68369492e-04, 8.57892235e-05, 3.60015410e-03]
    p_std_covmat = [0.00025883, 0.00011546, 5.72565e-05, 0.00263298]


ODE, 2 states, 1 control, 1 param (pendulum example)
----------------------------------------------------

**100 repetitions**

.. code-block:: python

    p_mean = 3.00007798152
    p_est_single = 3.01470021

    p_std = 0.00690586777508
    p_std_covmat = 0.00488056


**200 repetitions**

.. code-block:: python

    p_mean = 3.0008566537017924
    p_est_single = 2.99838121

    p_std = 0.0065451287256237736
    p_std_covmat = 0.0048511


Observations
------------

- All corresponding values for the standard deviations obtained from the repeated estimations and from the covariance matrix computations are in the same order of magnitude and the values close to each other.

- For the observed scenarios, all values for the standard deviations obtained from the covariance matrix computations were smaller then the correpsonding values from the repeated estimations.