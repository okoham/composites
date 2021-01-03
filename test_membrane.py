# -*- coding: utf-8 -*-
"""
Created on 24.09.2014
@author: Oliver
"""

from __future__ import division, print_function
import unittest
import numpy as np 
from numpy.testing import assert_allclose
#import math
import clt
import plainstrength
#from clt6 import UnorderedLaminate, MembraneLaminate
from material import TransverseIsotropicPlyMaterial


class Jones_4_5_5(unittest.TestCase):
    
    expect_a = np.array([[37207, 6976.7, 0],
                         [6976.7, 74405, 0],
                         [0, 0, 13137]])
    expect_alphat = np.array([6.3e-6, 20.52e-6, 0])
    expect_sigtunit = np.array([0.41049, 0.43407, 0]) # unit stress!!!
    expect_sigg_0 = np.array([0.4409, -0.1977, 0]) # for delta T = 1
    expect_sigg_90 = np.array([-0.08819, 0.03954, 0])
    #lam = None
    
    def setUp(self):
        mat = TransverseIsotropicPlyMaterial(E11=53780, E22=17930, nu12=0.25, G12=8620,
                                         a11t=6.3e-6, a22t=20.52e-6, t=0.127,
                                         F11t=1035, F11c=1035, F22t=27.6, F22c=138, F12s=41.4)
        #sseq = [0] + [90]*10 + [0]
        layup = [(0, 2), (90, 10)]
        self.lam = clt.UnorderedLaminate(layup, mat)
        dtemp = 1
        self.sol = self.lam.get_linear_response(np.zeros(3), dtemp) 
        
    def test_a(self):
        assert_allclose(self.lam.A(), self.expect_a, atol=1)

    def test_thickness(self):
        self.assertEqual(self.lam.thickness(), 1.524)

    @unittest.skip('apparent thermal expansion for laminates not implemented yet')
    def test_apparent_alpha(self):
        assert_allclose(self.lam.alphat(), self.expect_alphat)

    def test_unit_thermal_load(self):
        assert_allclose(self.lam.thermal_force(1), 
                        self.expect_sigtunit*self.lam.thickness(), atol=1e-5)

    def test_sigma_g_0(self):
        layer = self.lam.get_layer_by_angle(0)
        assert_allclose(self.sol.sigma_g(layer), self.expect_sigg_0, atol=1e-3)

    def test_sigma_g_90(self):
        layer = self.lam.get_layer_by_angle(90)
        assert_allclose(self.sol.sigma_g(layer), self.expect_sigg_90, atol=1e-3)


###############################################################################

class CltTestMTS006C(unittest.TestCase):
    
    mat = TransverseIsotropicPlyMaterial(name='T300/BSL914',
        E11=130000., E22=4650., G12=4650., nu12=0.35, nu23=0.45,
        a11t=0.15e-6, a22t=28.7e-6, t=0.13,
        F11t=1200., F11c=1000., F22t=50., F22c=120., F12s=75.)
        
    expect_qbar_0 = np.array([[130572.13, 1634.66, 0], 
                              [1634.66, 4670.46, 0], 
                              [0, 0, 4650]])
    
    expect_qbar_45 = np.array([[39278.0, 29978.0, 31475.4], 
                               [29978.0, 39278.0, 31475.4], 
                               [31475.4, 31475.4, 32993.3]])
    
    expect_qbar_m45 = np.array([[39278.0, 29978.0, -31475.4], 
                                [29978.0, 39278.0, -31475.4], 
                                [-31475.4, -31475.4, 32993.3]])
    
    expect_qbar_90 = np.array([[4670.46, 1634.66, 0], 
                               [1634.66, 130572.13, 0], 
                               [0, 0, 4650]])
    
    expect_a = np.array([[56284.0, 12972.0, 0], 
                         [12972.0, 56284.0, 0], 
                         [0, 0, 15987.3]])
    
    def setUp(self):
        # aerospatiale example, C6
        angles = (45, -45, 0, 90, 0, 90, 45, -45, 0, 90, 90, 0, -45, 45, 90, 0, 90, 0, -45, 45)
        #self.plydef = [(a, self.mat) for a in angles]
        stseq = clt.stacking_to_layup2(angles)
        self.lam = clt.UnorderedLaminate(stseq, self.mat)
        self.load = np.array([308.3, -22.2, 449.2])
        self.sol = self.lam.get_linear_response(self.load, 0)
        self.temp = 0.
        
    def test_symmetric(self):
        self.assertTrue(self.lam.is_symmetric()) 
        

    def test_qbar_0(self):
        # Qbar, 0 deg
        expect = self.expect_qbar_0
        layer = self.lam.get_layer_by_angle(0)
        assert_allclose(layer.Qbar(), expect, atol=1)
              

    def test_qbar_45(self):
        # Qbar, 45 deg
        expect = self.expect_qbar_45
        layer = self.lam.get_layer_by_angle(45)
        assert_allclose(layer.Qbar(), expect, atol=1)


    def test_qbar_m45(self):
        # Qbar, 45 deg
        expect = self.expect_qbar_m45
        layer = self.lam.get_layer_by_angle(-45)
        assert_allclose(layer.Qbar(), expect, atol=1)


    def test_qbar_90(self):
        # Qbar, 90 deg
        expect = self.expect_qbar_90
        layer = self.lam.layers[3]
        assert_allclose(layer.Qbar(), expect, atol=1)


    def test_a(self):
        # A matrix
        expect = self.expect_a
        assert_allclose(self.lam.A()/self.lam.thickness(), expect, atol=1)

    def test_strain(self):
        # global strain and curvature
        expect = np.array([2262.0e-6, -673.0e-6, 10807.0e-6])
        #sol = clt6.linear_solve(self.lam, self.load)
        eps = self.sol.eps0()
        assert_allclose(eps, expect, rtol=1e-4)

    def test_strain_0(self):
        # ply strains, laminate direction
        expect = np.array([2262.0e-6, -673.0e-6, 10807.0e-6])
        layer = self.lam.get_layer_by_angle(0)
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)

    def test_strain_45(self):
        # ply strains, laminate direction
        expect = np.array([6198.0e-6, -4609.0e-6, -2935.0e-6])
        layer = self.lam.get_layer_by_angle(45)
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)

    def test_strain_m45(self):
        # ply strains, laminate direction
        expect = np.array([-4609.0e-6, 6198.0e-6, 2935.0e-6])
        layer = self.lam.get_layer_by_angle(-45)
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)

    def test_strain_90(self):
        # ply strains, laminate direction
        expect = np.array([-673.0e-6, 2262.0e-6, -10807.0e-6])
        layer = self.lam.get_layer_by_angle(90)
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)


    def test_stress_0(self):
        # ply stress, ply C/S
        expect = 10 * np.array([29.42, 0.06, 5.03])
        layer = self.lam.get_layer_by_angle(0)
        assert_allclose(self.sol.sigma_l(layer), expect, atol=0.05)

    def test_stress_45(self):
        # ply stress, ply C/S
        expect = 10 * np.array([80.17, -1.14, -1.36])
        layer = self.lam.get_layer_by_angle(45)
        assert_allclose(self.sol.sigma_l(layer), expect, atol=0.05)

    def test_stress_m45(self):
        # ply stress, ply C/S
        expect = 10 * np.array([-59.17, 2.14, 1.36])
        layer = self.lam.get_layer_by_angle(-45)
        assert_allclose(self.sol.sigma_l(layer), expect, atol=0.05)

    def test_stress_90(self):
        # ply stress, ply C/S
        expect = 10 * np.array([-8.42, 0.95, -5.03])
        layer = self.lam.get_layer_by_angle(90)
        assert_allclose(self.sol.sigma_l(layer), expect, atol=0.05)
        
    def test_margin_0(self):
        expect = 1.40
        layer = self.lam.get_layer_by_angle(0)
        sigma12 = self.sol.sigma_l(layer)
        all_rf_fmode = plainstrength.layer_strength_analysis_a(self.mat, 'TsaiHill', sigma12)
        r, fmode = min(all_rf_fmode, key=lambda x: x[0])
        print(r, fmode)
        self.assertAlmostEqual(r, expect, delta=0.005)
        
    def test_margin_45(self):
        expect = 1.42 
        layer = self.lam.get_layer_by_angle(45)
        sigma12 = self.sol.sigma_l(layer)
        all_rf_fmode = plainstrength.layer_strength_analysis_a(self.mat, 'TsaiHill', sigma12)
        r, fmode = min(all_rf_fmode, key=lambda x: x[0])
        print(r, fmode)
        self.assertAlmostEqual(r, expect, delta=0.005)
        
    def test_margin_m45(self):
        expect = 1.31
        layer = self.lam.get_layer_by_angle(-45)
        sigma12 = self.sol.sigma_l(layer)
        all_rf_fmode = plainstrength.layer_strength_analysis_a(self.mat, 'TsaiHill', sigma12)
        r, fmode = min(all_rf_fmode, key=lambda x: x[0])
        print(r, fmode)
        self.assertAlmostEqual(r, expect, delta=0.005)  
        
    def test_margin_90(self):
        expect = 1.42
        layer = self.lam.get_layer_by_angle(90)
        sigma12 = self.sol.sigma_l(layer)
        all_rf_fmode = plainstrength.layer_strength_analysis_a(self.mat, 'TsaiHill', sigma12)
        r, fmode = min(all_rf_fmode, key=lambda x: x[0])
        print(r, fmode)
        self.assertAlmostEqual(r, expect, delta=0.005)
        
        
###########################################################################


class MTS006_V_Example3(unittest.TestCase):
    
    def setUp(self):
        # aerospatiale example, V9.3
        mat = TransverseIsotropicPlyMaterial(name='T300/BSL914',
            E11=130000., E22=4650., G12=4650., nu12=0.35, 
            a11t=-1e-6, a22t=40e-6, t=0.13) 
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        plydef = [(a, mat) for a in angles]

        self.lam = clt.MembraneLaminate(plydef)
        load = np.zeros(3)
        self.dtemp = -160
        self.sol = self.lam.get_linear_response(load, self.dtemp)
        
    def test_a(self):
        expect = 10 * np.array([[5558, 1642, 0],
                           [1642, 5558, 0],
                           [0, 0, 1956]])
        result = self.lam.A()
        assert_allclose(result, expect, atol=20)
        
    def test_ntunit(self):
        expect = 10*np.array([6.232e-3, 6.232e-3, 0])
        result = self.lam.thermal_force(1)
        assert_allclose(result, expect, atol=1e-4)
        
    def test_global_deformation(self):
        expect = np.array([-13.85e-5, -13.85e-5, 0])
        #layer = self.sol.layers[99]
        result = self.sol.eps_g(2)
        assert_allclose(result, expect, atol=1e-6)
        
    def test_stress_0(self):
        expect = 10*np.array([-2.88, 2.88, 0])
        result = self.sol.sigma_l(self.lam.layers[0])
        assert_allclose(result, expect, atol=0.1)
        
    def test_stress_45(self):
        layer = self.lam.layers[1]
        expect = self.sol.sigma_l(layer)
        result = self.sol.sigma_l_r(layer)
        assert_allclose(expect, result)
        
    def test_strain_45(self):
        layer = self.lam.layers[1]
        expect = self.sol.eps_l(layer)
        result = self.sol.eps_l_r(layer)
        assert_allclose(expect, result)        
    
    def test_stress_m45(self):
        layer = self.lam.layers[5]
        expect = self.sol.sigma_l(layer)
        result = self.sol.sigma_l_r(layer)
        assert_allclose(expect, result)
        
    def test_strain_m45(self):
        layer = self.lam.layers[5]
        expect = self.sol.eps_l(layer)
        result = self.sol.eps_l_r(layer)
        assert_allclose(expect, result)     
        
    def test_stress_90(self):
        layer = self.lam.layers[3]
        expect = self.sol.sigma_l(layer)
        result = self.sol.sigma_l_r(layer)
        assert_allclose(expect, result)
        
    def test_strain_90(self):
        layer = self.lam.layers[3]
        expect = self.sol.eps_l(layer)
        result = self.sol.eps_l_r(layer)
        assert_allclose(expect, result)          
######################################################################################    
        

###############################################

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()