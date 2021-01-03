# -*- coding: utf-8 -*-
"""
Created on 05.10.2014
@author: Oliver

Aerospatiale composite stress manual, Example V9.3
The example provides laminate stiffness matrix, and internal curing stresses 
in a laminated plate.

"""

from __future__ import division, print_function
import unittest
import numpy as np
from numpy.testing import assert_allclose
from material import TransverseIsotropicPlyMaterial as Material
from clt import Laminate

MAT = Material(name='T300/BSL914',
               E11=130000., E22=4650., G12=4650., nu12=0.35, 
               a11t=-1e-6, a22t=40e-6, t=0.13) 

SSEQ = [(theta, MAT) for theta in (0, 45, -45, 90, 90, -45, 45, 0)]

# FIXME: this lacks some of the results in manual...

class MTS006_ExampleV93(unittest.TestCase):
    
    def setUp(self):
        self.lam = Laminate(SSEQ)
        load = np.zeros(6)
        self.dtemp = -160
        self.sol = self.lam.get_linear_response(load, self.dtemp)
        
    def test_a(self):
        """Verify the laminate stiffness matrix."""
        expect = np.array([[55580, 16420, 0],
                           [16420, 55580, 0],
                           [0, 0, 19560]])
        result = self.lam.A()
        assert_allclose(result, expect, atol=20)
        
    def test_ntunit(self):
        """Test the unit thermal forces."""
        expect = 10*np.array([6.232e-3, 6.232e-3, 0, 0, 0, 0])
        result = self.lam.thermal_force(1)
        assert_allclose(result, expect[:3], atol=1e-4)
        
    def test_mtunit(self):
        """Test the unit thermal forces."""
        expect = 10*np.array([6.232e-3, 6.232e-3, 0, 0, 0, 0])
        result = self.lam.thermal_moment(1)
        assert_allclose(result, expect[3:], atol=1e-4)
                
    def test_global_deformation(self):
        """Verify global strains and curvatures, eps_0 and kappa."""
        expect = np.array([-138.5, -138.5, 0, 0, 0, 0]) # microstrains
        result = self.sol.eps_kappa() * 1e6
        assert_allclose(result, expect, atol=0.2)
        
    def test_stress_0(self):
        """Verify total stresses in 0 deg layer."""
        expect = np.array([-28.8, 28.8, 0]) # MPa
        result = self.sol.sigma_l(self.lam.layers[0])
        assert_allclose(result, expect, atol=0.1)
        
    def test_stress_45(self):
        layer = self.lam.layers[1]
        expect = self.sol.sigma_l(layer)
        result = self.sol.sigma_l_r(layer)
        assert_allclose(expect, result, atol=1e-6)
        
    def test_strain_45(self):
        layer = self.lam.layers[1]
        expect = self.sol.eps_l(layer)
        result = self.sol.eps_l_r(layer)
        assert_allclose(expect, result, atol=1e-6)        
    
    def test_stress_m45(self):
        layer = self.lam.layers[5]
        expect = self.sol.sigma_l(layer)
        result = self.sol.sigma_l_r(layer)
        assert_allclose(expect, result, atol=1e-6)
        
    def test_strain_m45(self):
        layer = self.lam.layers[5]
        expect = self.sol.eps_l(layer)
        result = self.sol.eps_l_r(layer)
        assert_allclose(expect, result, atol=1e-6)     
        
    def test_stress_90(self):
        layer = self.lam.layers[3]
        expect = self.sol.sigma_l(layer)
        result = self.sol.sigma_l_r(layer)
        assert_allclose(expect, result, atol=1e-6)
        
    def test_strain_90(self):
        layer = self.lam.layers[3]
        expect = self.sol.eps_l(layer)
        result = self.sol.eps_l_r(layer)
        assert_allclose(expect, result, atol=1e-6)          
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()