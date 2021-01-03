# -*- coding: utf-8 -*-
"""
Created on 05.10.2014
@author: Oliver

Berthelot Exmaple 14.5.5 provides stresses and strains per layer in an 
asymmetric laminate subjected to membrane loading.

"""

from __future__ import division, print_function
import unittest
import numpy as np 
from numpy.testing import assert_allclose
#import math
import clt
#from clt import Laminate
from material import TransverseIsotropicPlyMaterial as Material
#import plainstrength

class BerthelotExample_14_5_5(unittest.TestCase):
    # Material: see 14.4.3.3 (Example 2) and Laminate: see 14.4.3.3 (Example 3)
    

    expect_a = 1000 * np.array([[158.22, 30.432, 0],
                                [30.432, 51.277, 0],
                                [0, 0, 33.676]]) # N/mm
        
    expect_b = 1000 * np.array([[-13.384, 5.2247, -1.6154],
                                [5.2247, 2.9342, 5.9258],
                                [-1.6154, 5.9258, 5.2247]]) # N
    
    expect_d = 1000 * np.array([[327.38, 64.271, 60.686],
                                [64.271, 107.32, 15.438],
                                [60.686, 15.438, 71.025]]) # Nmm
    
    # Note: kappa_y is -1.027 (use values in first equation of ch. 14.5.5.2)
    # eps: mm/mm, kappa: 1/mm
    expect_epskappa = np.array([5.064e-3, 6.897e-3, 7.836e-3, 0.580e-3, -1.027e-3, -1.309e-3])
    
    # Fig. 14.6 a)
    expect_epsg_3_e = 1.0e-3 * np.array([6.5, 4.3, 4.6])
    expect_epsg_0_a = 1.0e-3 * np.array([3.6, 9.5, 11.1])
    
    # Fig. 14.6 b)
    expect_epsl_3_e = 1.0e-3 * np.array([7.9, 2.9, 0.4])
    expect_epsl_3_a = 1.0e-3 * np.array([8.3, 2.96, 2.4])
    expect_epsl_2_e = 1.0e-3 * np.array([4.4, 6.9, 5.3])
    expect_epsl_2_a = 1.0e-3 * np.array([3.2, 8.7, 5.9])
    expect_epsl_1_e = 1.0e-3 * np.array([2.1, 9.8, 2.3])
    expect_epsl_1_a = 1.0e-3 * np.array([1.4, 11, 1.5])
    expect_epsl_0_e = 1.0e-3 * np.array([7, 5.4, 9.6])
    expect_epsl_0_a = 1.0e-3 * np.array([6.8, 6.3, 12.5])
    
    # Fig. 14.6 c)
    expect_sigg_3_e = np.array([250, 118, 116])
    expect_sigg_3_a = np.array([256, 130, 126])
    expect_sigg_2_e = np.array([194, 74, -12])
    expect_sigg_2_a = np.array([158, 84, 3]) # book values incorrect? np.array([158, 87, 4]) 
    expect_sigg_1_e = np.array([115, 93, -2])
    expect_sigg_1_a = np.array([96, 96, 11])
    expect_sigg_0_e = np.array([257, 102, 85])
    expect_sigg_0_a = np.array([246, 114, 91])

    # Fig. 14.6 d)
    expect_sigl_3_e = np.array([318, 50, 1.4])
    expect_sigl_3_a = np.array([333, 52, 9])
    expect_sigl_2_e = np.array([193, 76, 19])
    expect_sigl_2_a = np.array([151, 90, 21])
    expect_sigl_1_e = np.array([112, 97, 8])
    expect_sigl_1_a = np.array([86, 106, 6])
    expect_sigl_0_e = np.array([289, 71, 35])
    expect_sigl_0_a = np.array([283, 78, 45])

    def setUp(self):
        mat10 = Material(E11=38000., E22=9000., G12=3600., nu12=0.32, t=1.)
        mat15 = Material(E11=38000., E22=9000., G12=3600., nu12=0.32, t=1.5)    

            
        self.lam = clt.Laminate([(15, mat15), (-30, mat10), (-15, mat15), (30, mat10)])
    
        # mload = np.array([2000, 1000, 500, 0, 0, 0]) # used in text
        mload = np.array([1000, 500, 250, 0, 0, 0]) # used in calculations
        self.sol = self.lam.get_linear_response(mload)

    def test_symmetric(self):
        self.assertFalse(self.lam.is_symmetric())
        
    def test_balanced(self):
        self.assertRaises(NotImplementedError, self.lam.is_balanced)        
        
    def test_a(self):
        assert_allclose(self.lam.A(), self.expect_a, atol=10)
        
    def test_b(self):
        assert_allclose(self.lam.B(), self.expect_b, 1e-4)
        
    def test_d(self):
        assert_allclose(self.lam.D(), self.expect_d, 1e-4)        
        
    def test_epskappa(self):
        assert_allclose(self.sol.eps_kappa(), self.expect_epskappa, 1e-3)
        
    def test_epsg_3e(self):
        #layer = self.sol.layers[3]
        assert_allclose(self.sol.eps_g(3, 1.0), self.expect_epsg_3_e, atol=0.0001)

    def test_epsg_0a(self):
        #layer = self.sol.layers[0]
        assert_allclose(self.sol.eps_g(0, 0.0), self.expect_epsg_0_a, atol=0.0001)
        
    # eps_l
        
    def test_epsl_0a(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.eps_l(layer, 0.0), self.expect_epsl_0_a, atol=0.0001)        

    def test_epsl_1a(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.eps_l(layer, 0.0), self.expect_epsl_1_a, atol=0.0001)

    def test_epsl_2a(self):
        layer = self.sol.layers[2]
        assert_allclose(self.sol.eps_l(layer, 0.0), self.expect_epsl_2_a, atol=0.0001)        

    def test_epsl_3a(self):
        layer = self.sol.layers[3]
        assert_allclose(self.sol.eps_l(layer, 0.0), self.expect_epsl_3_a, atol=0.0001)    
        
    def test_epsl_0e(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.eps_l(layer, 1.0), self.expect_epsl_0_e, atol=0.0001)        

    def test_epsl_1e(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.eps_l(layer, 1.0), self.expect_epsl_1_e, atol=0.0001)        

    def test_epsl_2e(self):
        layer = self.sol.layers[2]
        assert_allclose(self.sol.eps_l(layer, 1.0), self.expect_epsl_2_e, atol=0.0001)        

    def test_epsl_3e(self):
        layer = self.sol.layers[3]
        assert_allclose(self.sol.eps_l(layer, 1.0), self.expect_epsl_3_e, atol=0.0001)              

    # sig_l
        
    def test_sigl_0a(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.sigma_l(layer, 0.0), self.expect_sigl_0_a, atol=1)        

    def test_sigl_1a(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.sigma_l(layer, 0.0), self.expect_sigl_1_a, atol=1)        

    def test_sigl_2a(self):
        layer = self.sol.layers[2]
        assert_allclose(self.sol.sigma_l(layer, 0.0), self.expect_sigl_2_a, atol=1)        

    def test_sigl_3a(self):
        layer = self.sol.layers[3]
        assert_allclose(self.sol.sigma_l(layer, 0.0), self.expect_sigl_3_a, atol=1)    
        
    def test_sigl_0e(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.sigma_l(layer, 1.0), self.expect_sigl_0_e, atol=1)        

    def test_sigl_1e(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.sigma_l(layer, 1.0), self.expect_sigl_1_e, atol=1)        

    def test_sigl_2e(self):
        layer = self.sol.layers[2]
        assert_allclose(self.sol.sigma_l(layer, 1.0), self.expect_sigl_2_e, atol=1)        

    def test_sigl_3e(self):
        layer = self.sol.layers[3]
        assert_allclose(self.sol.sigma_l(layer, 1.0), self.expect_sigl_3_e, atol=1)  
        
    # sig_g
        
    def test_sigg_0a(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.sigma_g(layer, 0.0), self.expect_sigg_0_a, atol=1)        

    def test_sigg_1a(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.sigma_g(layer, 0.0), self.expect_sigg_1_a, atol=1)        

    def test_sigg_2a(self):
        layer = self.sol.layers[2]
        assert_allclose(self.sol.sigma_g(layer, 0.0), self.expect_sigg_2_a, atol=1)        

    def test_sigg_3a(self):
        layer = self.sol.layers[3]
        assert_allclose(self.sol.sigma_g(layer, 0.0), self.expect_sigg_3_a, atol=1)    
        
    def test_sigg_0e(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.sigma_g(layer, 1.0), self.expect_sigg_0_e, atol=1)        

    def test_sigg_1e(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.sigma_g(layer, 1.0), self.expect_sigg_1_e, atol=1)        

    def test_sigg_2e(self):
        layer = self.sol.layers[2]
        assert_allclose(self.sol.sigma_g(layer, 1.0), self.expect_sigg_2_e, atol=1)        

    def test_sigg_3e(self):
        layer = self.sol.layers[3]
        assert_allclose(self.sol.sigma_g(layer, 1.0), self.expect_sigg_3_e, atol=1)      


######################################


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()