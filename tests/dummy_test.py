#!/usr/bin/env python

# dummy_test.py: just something to get the testing working

import pecas
import numpy as np
import casadi as ca
from nose.tools import assert_equals

class TestDummy(object):
    """ basic dummy test """
    def test_it_woo(self):
        d = 1
        x = ca.SX.sym("x", d)
        
        M = np.array([1., 2., 3., 4.]) / 3. * x[0]
        
        sigma = 0.1 * np.ones(M.shape[0])
        
        Y = np.array([2.5, 4.1, 6.3, 8.2])
        
        pep = pecas.PECasLSq(x, M, sigma, Y = Y)
        
        pep.run_parameter_estimation()
        
        pep.compute_covariance_matrix()
        pep.print_results()

        assert_equals(1, 0) # heh
