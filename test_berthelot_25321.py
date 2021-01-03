# -*- coding: utf-8 -*-
"""
Created on 06.10.2014
@author: Oliver
"""

from __future__ import division, print_function
import unittest
import numpy as np 
from numpy.testing import assert_allclose
import clt
from material import TransverseIsotropicPlyMaterial as Material


class BerthelotExample_25_3_2_1(unittest.TestCase):
    # thermal stress stuff
    
    expect_q0 = np.array([[45982, 3168, 0],
                          [3168, 10218, 0],
                          [0, 0, 4500]])    
    expect_sigl_0 = np.array([-19.1, 11.7, 0])
    expect_sigl_90 = np.array([-5.9, 9.5, 0])
    expect_sigx_90 = np.array([9.6, -5.9, 0])
    expect_ntunit = np.array([733.7, 806.7, 0, 0, 0, 0]) / 1000 # N/(mm*K)  
      
    def setUp(self):
        mat25 = Material(E11=45000., E22=10000., G12=4500., 
                         a11t=5e-6, a22t=20e-6, nu12=0.31, t=1.)
        self.lam = clt.Laminate([(90, mat25), (0, mat25), (90, mat25)])
        
        self.mload = np.zeros(6)
        self.dtemp = -100
        self.sol = self.lam.get_linear_response(self.mload, self.dtemp)
    
    
    def test_apparent_alpha(self):
        expect = np.array([10.05, 6.96, 0]) # microstrains
        result = self.lam.alphaT() * 1e6 # convert to microstrains
        assert_allclose(result, expect, atol = 0.01)
    
    def test_unit_thermal_strains_90g(self):
        expect = np.array([10.05e-6, 6.96e-6, 0])
        sol = self.lam.get_linear_response(self.mload, 1)
        result = sol.eps_g(self.lam.layers[0])
        assert_allclose(1e6*result, 1e6*expect, atol = 0.01)
        
    def test_unit_thermal_strains_0l(self):
        expect = np.array([10.05e-6, 6.96e-6, 0])
        sol = self.lam.get_linear_response(self.mload, 1)
        result = sol.eps_l(self.lam.layers[1])
        assert_allclose(1e6*result, 1e6*expect, atol = 0.01)
        
    def test_unit_thermal_strains_0g(self):
        expect = np.array([10.05e-6, 6.96e-6, 0])
        sol = self.lam.get_linear_response(self.mload, 1)
        result = sol.eps_g(1)
        assert_allclose(1e6*result, 1e6*expect, atol = 0.01)    
        
    def test_unit_thermal_strains_0gr(self):
        expect = np.array([10.05e-6, 6.96e-6, 0])
        sol = self.lam.get_linear_response(self.mload, 1)
        result = sol.eps_g_r(1)
        assert_allclose(1e6*result, 1e6*expect, atol = 0.01)
        
    def test_unit_thermal_strains_0gr_withmload(self):
        expect = np.array([10.05e-6, 6.96e-6, 0])
        sol = self.lam.get_linear_response([123, 4, -34, 0, 0, 0], 1)
        result = sol.eps_g_r(1)
        assert_allclose(1e6*result, 1e6*expect, atol = 0.01)        
        
    def test_unit_thermal_strains_0gm(self):
        #expect = np.array([10.05e-6, 6.96e-6, 0])
        sol = self.lam.get_linear_response(self.mload, 1)
        result = sol.eps_g_m(1)
        assert_allclose(1e6*result, 0, atol = 0.01)        
            
    def test_q0(self):
        layer = self.lam.layers[1]
        assert_allclose(layer.Qbar(), self.expect_q0, atol=1)
        
    def test_thermal_force(self):
        result = self.lam.thermal_force(1)
        assert_allclose(result, self.expect_ntunit[:3], atol=1e-4)

    def test_thermal_moment(self):
        result = self.lam.thermal_moment(1)
        assert_allclose(result, self.expect_ntunit[3:], atol=1e-4)
        
    def test_sigl_therm_0(self):
        layer = self.lam.layers[1]
        assert_allclose(self.sol.sigma_l(layer), self.expect_sigl_0, atol=0.1)
     
    def test_sigl_therm_90(self):
        layer = self.lam.layers[0]
        assert_allclose(self.sol.sigma_l(layer), self.expect_sigl_90, atol=0.1)
    
    
################################################################    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()