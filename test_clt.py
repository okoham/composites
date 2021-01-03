# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:17:34 2012

@author: oliver

Categorise these tests:
A) "Unit Tests" related to the function of the software
- correct input/output format
- do functions raise an error, ...
- are results formally correct?

B) "Technical Tests" -> are these already "Integration Tests"?
- are the results technically correct?
"""

import unittest
import numpy as np 
from numpy.testing import assert_array_almost_equal, assert_allclose
import math
import clt
from material import TransverseIsotropicPlyMaterial
import plainstrength


class MaterialTest(unittest.TestCase):
  
    def test_material_init(self):
        # mech. load only, 
        m = TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
            E11=154000., E22=8500., G12=4200., nu12=0.35, nu23=0.45,
            a11t=0.15e-6, a22t=28.7e-6, t=0.184,
            F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        self.assertIsInstance(m, TransverseIsotropicPlyMaterial)

################################################################    


class BerthelotExample_14_4_3_1(unittest.TestCase):
    
    mat3 = TransverseIsotropicPlyMaterial(E11=46000., E22=10000., G12=4600., 
                                          nu12=0.31, t=3.)
    mat5 = TransverseIsotropicPlyMaterial(E11=46000., E22=10000., G12=4600., 
                                          nu12=0.31, t=5.)    
    lam = clt.Laminate([(45, mat3), (0, mat5)])
    
    expect_qbar45 = 1000 * np.array([[20.481, 11.282, 9.192],
                                     [11.282, 20.482, 9.192],
                                     [9.192, 9.192, 12.716]])
    
    expect_a = 1000 * np.array([[296.35, 49.676, 27.576],
                                [49.676, 112.51, 27.576],
                                [27.576, 27.576, 61.147]])
    
    expect_b =  1000 * np.array([[198.75, -60.87, -68.94],
                                 [-60.87, -77.01, -68.94],
                                 [-68.94, -68.94, -60.87]])
    
    def test_symm(self):
        self.assertFalse(self.lam.is_symmetric()) 
        
    def test_layer_angle(self):
        # second layer, 45
        self.assertEqual(self.lam.layers[0].angle(), math.radians(45))
        
    def test_qbar45(self):
        layer = self.lam.layers[0]
        assert_allclose(layer.Qbar(), self.expect_qbar45, 1e-4)
                
    def test_a(self):
        assert_allclose(self.lam.A(), self.expect_a, 1e-4)

    def test_b(self):
        assert_allclose(self.lam.B(), self.expect_b, 1e-4)
        
######################################
    
class TsaiTestCase(unittest.TestCase):
    """test layer Q, S matrices - based on Tsai, Theory of composites design,
    Eq. 2.29
    """
    
    def setUp(self):
        """elastic properties: table 3.2, strength values: table 8.3
        thermal and moisture expansion properties: table 4.4
        """
        self.cfrp_t300 = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=181000., E22=10300., G12=7170., nu12=0.28, 
            a11t=0.02e-6, a22t=22.5e-6, b11m=0., b22m=0.6,
            F11t=1500., F11c=1500., F22t=40., F22c=246., F12s=68.,
            name="CFRP/T300/N5208", 
            verbose_name="T300/N5208, from Tsai, Theory of composites design, Tables 3.2, 4.4, 8.3")
    
        self.bfrp = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=204000., E22=18500., G12=5590., nu12=0.23, 
            a11t=6.1e-6, a22t=30.3e-6, b11m=0., b22m=0.6,
            F11t=1260., F11c=-2500., F22t=61., F22c=-202., F12s=67.,
            name="BFRP/B(4)/N5505", 
            verbose_name="BFRP/B(4)/N5505, from Tsai, Theory of composites design, Table 3.2")

        self.cfrp_as = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=138000., E22=8960., G12=7100., nu12=0.30, 
            a11t=-0.3e-6, a22t=28.1e-6, b11m=0., b22m=0.44,
            F11t=1447., F11c=-1447., F22t=52., F22c=-206., F12s=93.,
            name="CFRP/AS/3501", 
            verbose_name="CFRP/AS/3501, from Tsai, Theory of composites design, Table 3.2")

        self.gfrp = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=38600., E22=8270., G12=4140., nu12=0.26, 
            a11t=8.6e-6, a22t=22.1e-6, b11m=0., b22m=0.6,
            F11t=1062., F11c=-610., F22t=31., F22c=-118., F12s=72.,
            name="GFRP/E-Glass/Epoxy", 
            verbose_name="GFRP/E-Glass/Epoxy, from Tsai, Theory of composites design, Table 3.2")

        self.kfrp = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=76000., E22=5500., G12=2300., nu12=0.34, 
            a11t=-4.0e-6, a22t=79.0e-6, b11m=0., b22m=0.6,
            F11t=1400., F11c=-235., F22t=12., F22c=-53., F12s=34.,
            name="KFRP/Kev-49/Epoxy", 
            verbose_name="KFRP/Kev-49/Epoxy, from Tsai, Theory of composites design, Table 3.2")

        self.cfrtp = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=134000., E22=8900., G12=5100., nu12=0.28, 
            F11t=2130., F11c=-1100., F22t=80., F22c=-200., F12s=160.,
            name="CFRTP/AS4/PEEK", 
            verbose_name="CFRTP/AS4/PEEK, from Tsai, Theory of composites design, Table 3.2")

        self.cfrp_im6 = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=203000., E22=11200., G12=8400., nu12=0.32, 
            F11t=3500., F11c=-1540., F22t=56., F22c=-150., F12s=98.,
            name="CFRP/IM6/Epoxy", 
            verbose_name="CFRP/IM6/Epoxy, from Tsai, Theory of composites design, Table 3.2")

        self.cfrp_t300_4 = TransverseIsotropicPlyMaterial(t=0.1, 
            E11=148000., E22=9650., G12=4550., nu12=0.3, 
            F11t=1314., F11c=-1200., F22t=43., F22c=-168., F12s=48.,
            name="CFRP/T300/Fbrt_934/4-mil_tp", 
            verbose_name="CFRP/T300/Fbrt_934/4-mil_tp, from Tsai, Theory of composites design, Table 3.2")

        self.ccrp_t300_13 = TransverseIsotropicPlyMaterial(t=0.325, 
            E11=74000., E22=74000., G12=4550., nu12=0.05, 
            F11t=499., F11c=-352., F22t=458., F22c=-352., F12s=46.,
            name="CCRP/T300/Fbrt_934/13-mil_c", 
            verbose_name="CCRP/T300/Fbrt_934/13-mil_c, from Tsai, Theory of composites design, Table 3.2")

        self.ccrp_t300_7 = TransverseIsotropicPlyMaterial(t=0.175, 
            E11=66000., E22=66000., G12=4100., nu12=0.04, 
            F11t=375., F11c=-279., F22t=368., F22c=-278., F12s=46.,
            name="CCRP/T300/Fbrt_934/7-mil_c", 
            verbose_name="CCRP/T300/Fbrt_934/7-mil_c, from Tsai, Theory of composites design, Table 3.2")

        self.lam = clt.Laminate([(0., self.cfrp_t300)])
        
        
    def test_layer_Q(self):
        # compare matrix norm of Q 
        expect = np.array([[181810., 2897., 0], [2897., 10346., 0.], [0., 0., 7170.]])
        result = self.lam.layers[0].Q()
        assert_allclose(result, expect, 1e-2)
        
    def test_layer_S(self):
        # compare matrix norm of compliance matrix S 
        expect = np.array([[5.52e-6, -1.55e-6, 0], 
                           [-1.55e-6, 97.09e-6, 0.], 
                           [0., 0., 139.47e-6]])
        result = self.lam.layers[0].S()
        assert_allclose(result, expect, 1e-2)

        
    def test_layer_A1(self):
        # compare matrix norm of matrix A of one-ply laminate
        # Tsai Table 4.1
        # B(4), 0°
        lam_bfrp_0 = clt.Laminate([(0., self.bfrp)])
        expect = np.array([[25.62e3, 0.53e3, 0], 
                            [0.53e3, 2.32e3, 0.], 
                            [0., 0., 0.7e3]])
        result = lam_bfrp_0.A()
        assert_allclose(result, expect, 1e-2)


    def test_layer_A2(self):
        # compare matrix norm of matrix A of one-ply laminate
        # Tsai Table 4.1
        # E-glass, 90°
        lam_gfrp_90 = clt.Laminate([(90., self.gfrp)])
        expect = np.array([[1.05e3, 0.27e3, 0.], 
                           [0.27e3, 4.90e3, 0.], 
                           [0., 0., 0.52e3]])
        result = lam_gfrp_90.A()
        assert_allclose(result, expect, atol=10)


    def test_layer_A3(self):
        # compare matrix norm of matrix A of one-ply laminate
        # Tsai Table 4.1
        # E-glass, 90°
        lam_kfrp_45 = clt.Laminate([(45., self.kfrp)])
        expect = np.array([[2.97e3, 2.40e3, 2.22e3], 
                            [2.40e3, 2.97e3, 2.22e3], 
                            [2.22e3, 2.22e3, 2.45e3]])
        result = lam_kfrp_45.A()
        assert_allclose(result, expect, 1e-2)


    def test_layer_A4(self):
        # compare matrix norm of matrix A of one-ply laminate
        # Tsai Table 4.1
        # E-glass, 90°
        lam_kfrp_m45 = clt.Laminate([(-45., self.kfrp)])
        expect = np.array([[2.97e3, 2.40e3, -2.22e3], 
                           [2.40e3, 2.97e3, -2.22e3], 
                           [-2.22e3, -2.22e3, 2.45e3]])
        result = lam_kfrp_m45.A()
        assert_allclose(result, expect, 1e-2)
        
    @unittest.skip('reference correct???')
    def test_ex(self):
        # test Ex for [0_3/90 laminate], T300/5208
        # Tsai Eq. 4.15
        lam = clt.Laminate([(0., self.cfrp_t300), (0., self.cfrp_t300), 
                             (0., self.cfrp_t300), (90., self.cfrp_t300)])
        ref = 138880.
        val = lam.Ex()
        self.assertAlmostEqual(val, ref)

    @unittest.skip('reference correct???')
    def test_ey(self):
        # test Ex for [0_3/90 laminate], T300/5208
        # Tsai Eq. 4.15
        lam = clt.Laminate([(0., self.cfrp_t300), (0., self.cfrp_t300), 
                             (0., self.cfrp_t300), (90., self.cfrp_t300)])
        ref = 53150.
        val = lam.Ey()
        self.assertAlmostEqual(val, ref)

    @unittest.skip('reference correct???')
    def test_nu(self):
        # test Ex for [0_3/90 laminate], T300/5208
        # Tsai Eq. 4.15
        lam = clt.Laminate([(0., self.cfrp_t300), (0., self.cfrp_t300), 
                             (0., self.cfrp_t300), (90., self.cfrp_t300)])
        ref = 0.054
        val = lam.nuxy()
        self.assertAlmostEqual(val, ref)


###############################################


class ESDU_94003_Ex1_TestCase(unittest.TestCase):
    
    def setUp(self):
        carbon = TransverseIsotropicPlyMaterial(t=0.04, 
                E11=207000., E22=7600., G12=5000., nu12=0.3,
                a22t=30e-6, a11t=0,
                name="ESDU 94003 Carbon")
        alu = TransverseIsotropicPlyMaterial(t=1.0,
                E11=73000., E22=73000., G12=28100., nu12=0.3,
                a11t=22e-6, a22t=22e-6,
                name="ESDU 94003 Alu")
        stacking = [(0., carbon), (15., carbon), (0., carbon), (-15., carbon), 
                    (0., carbon), (0., alu), (0., carbon), (-15., carbon), 
                    (0., carbon), (15., carbon), (0., carbon)]
        self.lam = clt.Laminate(stacking)
        
    def test_symm(self):
        self.assertTrue(self.lam.is_symmetric()) 
                
    def test_A(self):
        expect = np.array([[159.2e3, 26.92e3, 0], 
                            [26.92e3, 83.48e3, 0.], 
                            [0., 0., 32.0e3]], dtype=np.float32)
        assert_array_almost_equal(self.lam.A()/1000, expect/1000, decimal=1)

        
    def test_D(self):
        expect = np.array([[35.4e3,  3.03e3, 0.351e3], 
                            [3.03e3, 7.87e3, 0.034e3], 
                            [0.351e3, 0.034e3, 3.76e3]], dtype=np.float32)
        assert_allclose(self.lam.D(), expect, atol=100)

    @unittest.skip('Values in ESDU example do not make sense to me')
    def test_thermal_stress_al(self):
        expect = np.array([151, 3.4, -14.4])
        sol = self.lam.get_linear_response(np.zeros(6), dtemp=-(180-30))
        layer = sol.layers[1]
        result = sol.sigma_l(layer)
        #result = sol.eps_kappa()
        assert_allclose(result, expect)

    @unittest.skip('Values in ESDU example do not make sense to me')
    def test_thermal_stress_layer2(self):
        expect = np.array([11, 4.4, -16.6])
        sol = self.lam.get_linear_response(np.zeros(6), dtemp=-(180+30))
        layer = sol.layers[1]
        result = sol.sigma_l(layer)
        #result = sol.eps_kappa()
        assert_allclose(result, expect)
######################################################################################    
  

class CltTestMTS006C(unittest.TestCase):
    
    """Aerospatiale's "Hill" criterion is actually Tsai-Hill."""
    
    mat = TransverseIsotropicPlyMaterial(name='exmat2',
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
        plydef = [(a, self.mat) for a in angles]

        self.lam = clt.Laminate(plydef)
        self.load = np.array([308.3, -22.2, 449.2, 0, 0, 0])
        self.sol = self.lam.get_linear_response(self.load)
        self.temp = 0.
        
    def test_symm_4(self):
        self.assertTrue(self.lam.is_symmetric()) 
        

    def test_qbar_0(self):
        # Qbar, 0 deg
        expect = self.expect_qbar_0
        layer = self.lam.layers[2]
        assert_array_almost_equal(layer.Qbar(), expect, decimal=2)

    def test_qbar_45(self):
        # Qbar, 45 deg
        expect = self.expect_qbar_45
        layer = self.lam.layers[0]
        assert_array_almost_equal(layer.Qbar(), expect, decimal=1)

    def test_qbar_m45(self):
        # Qbar, 45 deg
        expect = self.expect_qbar_m45
        layer = self.lam.layers[1]
        assert_array_almost_equal(layer.Qbar(), expect, decimal=1)


    def test_qbar_90(self):
        # Qbar, 90 deg
        expect = self.expect_qbar_90
        layer = self.lam.layers[3]
        assert_array_almost_equal(layer.Qbar(), expect, decimal=2)

    def test_a(self):
        # A matrix
        expect = self.expect_a
        assert_array_almost_equal(self.lam.A()/self.lam.thickness(), expect, decimal=1)

    def test_strain(self):
        # global strain and curvature
        expect = np.array([2262.0e-6, -673.0e-6, 10807.0e-6, 0, 0, 0])
        #sol = clt.linear_solve(self.lam, self.load)
        eps = self.sol.eps_kappa()
        assert_array_almost_equal(eps, expect, decimal=6)

    def test_strain_0(self):
        # ply strains, laminate direction
        expect = np.array([2262.0e-6, -673.0e-6, 10807.0e-6])
        layer = self.lam.layers[2]
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)

    def test_strain_45(self):
        # ply strains, laminate direction
        expect = np.array([6198.0e-6, -4609.0e-6, -2935.0e-6])
        layer = self.lam.layers[0]       
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)

    def test_strain_m45(self):
        # ply strains, laminate direction
        expect = np.array([-4609.0e-6, 6198.0e-6, 2935.0e-6])
        layer = self.lam.layers[1]        
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)

    def test_strain_90(self):
        # ply strains, laminate direction
        expect = np.array([-673.0e-6, 2262.0e-6, -10807.0e-6])
        layer = self.lam.layers[3]
        assert_allclose(self.sol.eps_l(layer), expect, rtol=1e-4)


    def test_stress_0(self):
        # ply stress, ply C/S
        expect = 10 * np.array([29.42, 0.06, 5.03])
        layer = self.lam.layers[2]
        assert_allclose(self.sol.sigma_l(layer), expect, atol=0.05)

    def test_stress_45(self):
        # ply stress, ply C/S
        expect = 10 * np.array([80.17, -1.14, -1.36])
        layer = self.lam.layers[0]
        assert_allclose(self.sol.sigma_l(layer), expect, rtol=1e-2)

    def test_stress_m45(self):
        # ply stress, ply C/S
        expect = 10 * np.array([-59.17, 2.14, 1.36])
        layer = self.lam.layers[1]
        assert_allclose(self.sol.sigma_l(layer), expect, rtol=1e-2)

    def test_stress_90(self):
        # ply stress, ply C/S
        expect = 10 * np.array([-8.42, 0.95, -5.03])
        layer = self.lam.layers[3]
        assert_allclose(self.sol.sigma_l(layer), expect, rtol=1e-2)

###############################################

class CltTest(unittest.TestCase):
    
    mat1 = TransverseIsotropicPlyMaterial(name='aerospatiale, d5, example',
        E11=130000., E22=4650., G12=4650., nu12=0.35,
        t=0.13,)  
    
    m1 = TransverseIsotropicPlyMaterial(name='TEST',
        E11=154000., E22=8500., G12=4200., nu12=0.35, nu23=0.45,
        a11t=0.15e-6, a22t=28.7e-6, t=0.25,
        Xt=2000., Xc=1000., Yt=50., Yc=200., Sl=100.)   
    
    m11 = TransverseIsotropicPlyMaterial(name='TEST 1 mm',
        E11=154000., E22=8500., G12=4200., nu12=0.35, nu23=0.45,
        a11t=0.15e-6, a22t=28.7e-6, t=1.0,
        Xt=2000., Xc=1000., Yt=50., Yc=200., Sl=100.)      
    
  
    def test_clt_init(self):
        # mech. load only, 
        plydef = [(angle, self.m1) for angle in (0, 90, 0)]
        # plydef is a list of (alpha, t, mat) tuples
        lam = clt.Laminate(plydef)
        self.assertEqual(lam.thickness(), 0.75)
                
        
    def test_clt_a0_temp(self):
        # result sould be alpha_t
        plydef = [(0., self.m11)]
        lam = clt.Laminate(plydef)
        mload = np.zeros(6)
        temps = 1.
        sol = lam.get_linear_response(mload, temps)
        # s: array(6). s[0] = a11t, s[1] = a22t, rest 0
        expect = np.array([self.m1.a11t, self.m1.a22t, 0., 0., 0., 0.])
        assert_array_almost_equal(sol.eps_kappa(), expect)

    def test_clt_a0_nx(self):
        # result sould be alpha_t
        #m11 = self.m11
        plydef = [(0., self.m11)]
        lam = clt.Laminate(plydef)
        mload = np.array([1,0,0,0,0,0])
        #temps = 0.
        sol = lam.get_linear_response(mload) #, mload)lam.solve(mload, temps)
        expect = np.array([1./self.m11.E11, -self.m11.nu12/self.m11.E11, 0., 0., 0., 0.])
        assert_allclose(sol.eps_kappa(), expect, rtol=1e-6)


    def test_d(self):
        # (2/2/2/2)
        # exampe mts006, D5
        pd2 = [0, 45, -45, 90, 90, -45, 45, 0]
        pd2 = [(angle, self.mat1) for angle in pd2] 
        lam = clt.Laminate(pd2)
        # Qbar, 0 deg
        expect = 10.0 * np.array([[858, 123, 55], 
                                  [123, 194, 55], 
                                  [55, 55, 151]])
        assert_allclose(lam.D(), expect, atol=6)      
        
    def test_kappa(self):
        # (2/2/2/2)
        # exampe mts006, D5
        pd2 = [0, 45, -45, 90, 90, -45, 45, 0]
        pd2 = [(angle, self.mat1) for angle in pd2] 
        lam = clt.Laminate(pd2)
        # Qbar, 0 deg
        load = np.array([0, 0, 0, 100, 0, -50])
        #temp = 0.
        expect = np.array([0, 0, 0, 13.82e-3, 2.283e-3, -38.98e-3])
        sol = lam.get_linear_response(load)
        assert_allclose(sol.eps_kappa(), expect, atol=0.001)

##############################################

class test_cap(unittest.TestCase):
    """
    compare to ETHZ CAP results
    """
    
    def setUp(self):
        #
        self.mat = TransverseIsotropicPlyMaterial(name='T300_Epoxy',
            E11=135000., E22=10000., G12=5000., nu12=0.27, nu23=0.30,
            a11t=-0.6e-6, a22t=40.0e-6, t=0.125,
            Xt=1450., Xc=1400., Yt=55., Yc=170., Sl=90.)
        #    
        self.plydef = [(20., self.mat),  (-20., self.mat),  
                      (20., self.mat),  (-20., self.mat)]
        #
        self.lam = clt.Laminate(self.plydef)
        
    def test_symm(self):
        self.assertTrue(self.lam.is_symmetric())         
    
    def test_abd(self):
        # A matrix
        expect = np.array([[54299.5, 7573.5, 0.0, 0.0, 0.0, -1094.1],
                           [7573.5, 6161.8, 0.0, 0.0, 0.0, -168.1],
                           [0.0, 0.0, 8716.1, -1094.1, -168.1, 0.0],
                           [0.0, 0.0, -1094.1, 1131.2, 157.8, 0.0],
                           [0.0, 0.0, -168.1, 157.8, 128.4, 0.0],
                           [-1094.1, -168.1, 0.0, 0.0, 0.0, 181.6]])
        assert_allclose(self.lam.ABD(), expect, atol=0.5)
        
    def test_slam_btm(self):
        load = np.array([0, 0, 120, 340, 0, 0])
        #temp = 0
        expect = np.array([[-7654.7, 77.6, -2058.9],
                           [-6859.0, -427.0, 2538.9],
                           [2021.1, 310.6, 1006.3],
                           [2816.8, -194.1, -526.3]])
        sol = self.lam.get_linear_response(load)
        sig = [sol.sigma_g(layer, relpos=0.0) for layer in sol.layers]
        assert_allclose(sig, expect, rtol=1e-3)

    def test_smat_top(self):
        # local stresses
        load = np.array([0, 0, 120, 340, 0, 0])
        #temp = 0
        expect = np.array([[-2802.9,  180.2,  564.5],
                           [-2467.9,  136.2,  221.1],
                           [ 7738.6, -452.5, -122.3],
                           [ 8073.6, -496.6,  907.9]])
        sol = self.lam.get_linear_response(load)
        sig = [sol.sigma_l(layer, relpos=1.0) for layer in sol.layers]
        assert_allclose(sig, expect, rtol=1e-3)
    

class TransverseShearTest(unittest.TestCase):
    
    def setUp(self):
        # hsb 37103-03 B 1991, example
        # G23 = 0.5*E22/(1. + nu23) = 4600
        # => nu23 = (0.5 E22/G23) - 1
        
        self.mat = TransverseIsotropicPlyMaterial(name='T300_Epoxy',
            E11=141000., E22=9400., G12=4600., nu12=0.30, nu23=0.46875, t=0.25)
        #    
        self.plydef = [(0., self.mat),  (90., self.mat),  
                       (90., self.mat),  (0., self.mat)]
        #
        self.lam = clt.Laminate(self.plydef)

    def test_q0(self):
        expect = np.array([[141850,  2840,  0],
                           [2840,  9460,  0],
                           [0, 0, 4600]])
        q0 = self.lam.layers[0].Q()
        assert_allclose(q0, expect, atol=5)

    @unittest.skip('Transverse Shear not implemented yet')
    def test_cbar_0(self):
        # fails. hsb uses different order of stresses???
        expect = np.array([[3200,  0],
                           [0,  4600]])
        c0 = self.lam.Cbar[0]
        assert_allclose(c0, expect)


    def test_d(self):
        expect = np.array([[10442,  236,  0],
                           [236,  2167,  0],
                           [0, 0, 383]])
        d = self.lam.D()
        assert_allclose(d, expect, atol=1)
        
    @unittest.skip('Transverse Shear not implemented yet')
    def test_Fts(self):
        expect = np.array([[3900,  0],
                           [0,  3900]])
        res = self.lam.Fts()
        assert_allclose(res, expect)

    @unittest.skip('Transverse Shear not implemented yet')
    def test_Hts(self):
        # hsb order is tauxz, tauyx -> changed here
        expect = np.array([[2540,  0],
                           [0,  3010]])
        res = self.lam.Hts()
        assert_allclose(res, expect)

    @unittest.skip('Transverse Shear not implemented yet')
    def test_kts(self):
        expect = np.array([[1,  0],
                           [0,  1]])
        res = self.lam.kts()
        assert_allclose(res, expect)
        
################################################################################


class Jones_4_5_5(unittest.TestCase):
    
    expect_a = np.array([[37207, 6976.7, 0],
                         [6976.7, 74405, 0],
                         [0, 0, 13137]])

    expect_sigtunit = np.array([0.41049, 0.43407, 0, 0, 0, 0]) # unit stress!!!
    expect_sigg_0 = np.array([0.4409, -0.1977, 0]) # for delta T = 1
    expect_sigg_90 = np.array([-0.08819, 0.03954, 0])

    expect_Nxfail_90_Tm111 = np.array([23.44, 0, 0])
    expect_Nxfail_90_T0 = np.array([36.68, 0, 0])
    
    def setUp(self):
        mat = TransverseIsotropicPlyMaterial(E11=53780, E22=17930, nu12=0.25, G12=8620,
                                         a11t=6.3e-6, a22t=20.52e-6, t=0.127,
                                         F11t=1035, F11c=1035, F22t=27.6, F22c=138, F12s=41.4)
        sseq = [0] + [90]*10 + [0]
        self.lam = clt.Laminate([(a, mat) for a in sseq])
        dtemp = 1
        self.sol = self.lam.get_linear_response(np.zeros(6), dtemp) #clt.linear_solve(self.lam, np.zeros(6), dtemp)
        
    def test_a(self):
        assert_allclose(self.lam.A(), self.expect_a, atol=1)

    def test_thickness(self):
        self.assertEqual(self.lam.thickness(), 1.524)


    def test_unit_thermal_load(self):
        ntmt = np.zeros(6)
        ntmt[:3] = self.lam.thermal_force(1)
        ntmt[3:] = self.lam.thermal_moment(1)
        assert_allclose(ntmt, self.expect_sigtunit*self.lam.thickness(), atol=1e-5)

    def test_sigma_g_0(self):
        layer = self.sol.layers[0]
        assert_allclose(self.sol.sigma_g(layer), self.expect_sigg_0, atol=1e-3)

    def test_sigma_g_90(self):
        layer = self.sol.layers[1]
        assert_allclose(self.sol.sigma_g(layer), self.expect_sigg_90, atol=1e-3)
        
    @unittest.skip('something wrong here. 10% deviation is too much ...')
    def test_fail_0_dt111(self):
        # expected failure load, for laminate cured at 132 and used at 21
        # DT = -111°C
        # pure x load
        # Tsai-Hill failure criterion
        expect_Nxfail = np.array([43.37, 0, 0, 0, 0, 0])*self.lam.thickness()
        # now we expect an RF of 1.00 in the outer plies, for Tsai-Hill
        # there is pure thermal loading; we put the full stress vector in the analysis
        psa = plainstrength.PlainStrengthAnalysisA(self.lam, 'TsaiHill')
        r = psa.all_rf(expect_Nxfail, -111)
        # get the ones for 0° layers
        r0 = [x.r for x in r if x.angle == 0] #filter(lambda x: x[0] == 0, r)
        print(r0)
        assert_allclose(r0, [1, 1], atol=0.01)
        
    def test_fail_0_dt0(self):
        # expected failure load, laminate cured and used at 21°C
        # DT = 0°C
        # pure x load
        # Tsai-Hill failure criterion
        expect_Nxfail = np.array([209.3, 0, 0, 0, 0, 0])*self.lam.thickness()
        psa = plainstrength.PlainStrengthAnalysisA(self.lam, 'TsaiHill')
        r = psa.all_rf(expect_Nxfail, 0)
        # now we expect an RF of 1.00 in the outer plies, for Tsai-Hill
        # get the ones for 0° layers
        r0 = [x.r for x in r if x.angle == 0] #filter(lambda x: x[0] == 0, r)
        print(r0)
        assert_allclose(r0, [1, 1], atol=0.01)

################################################################################

class BerthelotExample_12_2_3_3(unittest.TestCase):

    def setUp(self):
        self.mat = TransverseIsotropicPlyMaterial(F11t=1400, F22t=35, F12s=70,
                                                  F11c=910, F22c=110, t=0.13,
                                                  E11=46000, E22=10000, G12=4600, nu12=0.31)
        self.stress = np.array([100, 100/12, 0])

    def test_max_stress(self):
        fclasses = plainstrength.failure_classes_a['MaxStress']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_max_strain(self):
        fclasses = plainstrength.failure_classes_a['MaxStrain']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_hoffmann(self):
        fclasses = plainstrength.failure_classes_a['Hoffmann']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_tsaihill(self):
        fclasses = plainstrength.failure_classes_a['TsaiHill']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)
        
    def test_tsaiwu_a(self):
        fclasses = plainstrength.failure_classes_a['TsaiWu/0.0']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_tsaiwu_b(self):
        fclasses = plainstrength.failure_classes_a['TsaiWu/-0.5']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_Airbus(self):
        fclasses = plainstrength.failure_classes_a['Airbus']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_hashin_a(self):
        fclasses = plainstrength.failure_classes_a['Hashin A']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)
        
    def test_hashin_b(self):
        fclasses = plainstrength.failure_classes_a['Hashin B']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)

    def test_puck(self):
        fclasses = plainstrength.failure_classes_a['Puck']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)
        
    def test_modpuck(self):
        fclasses = plainstrength.failure_classes_a['ModPuck']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)
        
    def test_puckactionplane(self):
        fclasses = plainstrength.failure_classes_a['Puck C']
        scrit_all = [fclass.critical_stress(self.mat, self.stress) for fclass in fclasses]
        print(scrit_all)        
                

        
################################################################################

class FailureCriteria(unittest.TestCase):
    """Test ply failure criteria, for a single ply."""
    
    def setUp(self):
        self.mat = TransverseIsotropicPlyMaterial(t=0.125, 
            E11=181000., E22=10300., G12=7170., nu12=0.28, 
            a11t=0.02e-6, a22t=22.5e-6, b11m=0., b22m=0.6,
            F11t=1500., F11c=1500., F22t=40., F22c=246., F12s=68.,
            name="CFRP/T300/N5208", 
            verbose_name="T300/N5208, from Tsai, Theory of composites design, Tables 3.2, 4.4, 8.3")
      
        self.s1 = np.array([100, 0, 0])
        self.s2 = np.array([0, 100, 0])
        self.s3 = np.array([0, 0, 100])
        self.s4 = np.array([-100, 0, 0])
        self.s5 = np.array([0, -100, 0])
        self.s6 = np.array([0, 0, -100])
        
        self.st2 = np.array([0, -10, 0])
        
        
    def test_s1_hoffmann(self):
        #fclasses = clt.failure_classes_a['Hoffmann']
        #scrit_all = [fclass.critical_stress(self.mat, self.s1) for fclass in fclasses]
        print(plainstrength.layer_strength_analysis_a(self.mat, 'Hoffmann', self.s1))

    def test_s1_hoffmann_r(self):
        #fclasses = clt.failure_classes_a['Hoffmann']
        #scrit_all = [fclass.critical_stress(self.mat, self.s1, self.st2) for fclass in fclasses]
        print(plainstrength.layer_strength_analysis_a(self.mat, 'Hoffmann', self.s1, self.st2))
        #print(scrit_all)        


# FIXME: this whole thing does not make sense. I have no expected values.
class DegradedLaminate(unittest.TestCase):
    """This set of tests is based on the Aerospatiale example. It compares the 
    properties and results of the original laminate to a degraded one.
    """
    def setUp(self):
            self.mat = TransverseIsotropicPlyMaterial(name='exmat2',
                E11=130000., E22=4650., G12=4650., nu12=0.35, nu23=0.45,
                a11t=0.15e-6, a22t=28.7e-6, t=0.13,
                F11t=1200., F11c=1000., F22t=50., F22c=120., F12s=75.)
            self.angles = [45, -45, 0, 90, 0, 90, 45, -45, 0, 90, 90, 0, -45, 45, 90, 0, 90, 0, -45, 45]
            plydef = [(a, self.mat) for a in self.angles]
            self.lam = clt.Laminate(plydef)
            self.lamd = clt.Laminate(plydef)
            for layer in self.lamd.layers:
                layer.set_matrix_failure(True)
            self.load = np.array([308.3, -22.2, 449.2, 0, 0, 0])
            self.sol = self.lam.get_linear_response(self.load)
            self.sold = self.lamd.get_linear_response(self.load)
            
            # put into it's own class!
            layup = clt.stacking_to_layup2(self.angles)
            self.ulam = clt.UnorderedLaminate(layup, self.mat)
            self.ulamd = clt.UnorderedLaminate(layup, self.mat)
            #self.ulamd = plainstrength.make_degraded_laminate(self.ulam)
            for layer in self.ulamd.layers:
                layer.set_matrix_failure(True)           
            
    def test_thickness(self):
        self.assertEqual(self.lam.thickness(), self.lamd.thickness())

    def test_num_layers(self):
        self.assertEqual(self.lam.num_layers(), self.lamd.num_layers())
        
    def test_angles(self):
        angles = [layer.angle_deg() for layer in self.lamd.layers]
        self.assertListEqual(angles, self.angles)
        
    def test_ex(self):
        expect = self.lam.Ex()
        self.assertAlmostEqual(self.lamd.Ex(), expect, delta=0.1*expect)

    def test_ey(self):
        expect = self.lam.Ey()
        self.assertAlmostEqual(self.lamd.Ey(), expect, delta=0.1*expect)

    def test_gxy(self):
        expect = self.lam.Gxy()
        self.assertAlmostEqual(self.lamd.Gxy(), expect, delta=0.2*expect)
        
    @unittest.skip('should be removed')
    def test_s12_45(self):
        s12_intact = self.sol.sigma_l(self.lam.layers[0])
        s12_degraded = self.sold.sigma_l(self.lamd.layers[0])
        assert_allclose(s12_intact, s12_degraded, atol=153)

    @unittest.skip('should be removed')
    def test_s12_m45(self):
        s12_intact = self.sol.sigma_l(self.lam.layers[1])
        s12_degraded = self.sold.sigma_l(self.lamd.layers[1])
        assert_allclose(s12_intact, s12_degraded, atol=150)

    @unittest.skip('should be removed')
    def test_s12_0(self):
        s12_intact = self.sol.sigma_l(self.lam.layers[2])
        s12_degraded = self.sold.sigma_l(self.lamd.layers[2])
        assert_allclose(s12_intact, s12_degraded, atol=50)

    @unittest.skip('should be removed')
    def test_s126_90(self):
        s12_intact = self.sol.sigma_l(self.lam.layers[3])
        s12_degraded = self.sold.sigma_l(self.lamd.layers[3])
        assert_allclose(s12_intact, s12_degraded, atol=50)

################################################################################

class UtilityTests(unittest.TestCase):
    
    def setUp(self):
            self.mat = TransverseIsotropicPlyMaterial(name='exmat2',
                E11=130000., E22=4650., G12=4650., nu12=0.35, nu23=0.45,
                a11t=0.15e-6, a22t=28.7e-6, t=0.13,
                F11t=1200., F11c=1000., F22t=50., F22c=120., F12s=75.)
            self.angles = [45, -45, 0, 90, 0, 90, 45, -45, 0, 90, 90, 0, -45, 45, 90, 0, 90, 0, -45, 45]
            plydef = [(a, self.mat) for a in self.angles]
            self.lam = clt.Laminate(plydef)
            self.lamd = clt.Laminate(plydef)
            for layer in self.lamd.layers:
                layer.set_matrix_failure(True)
            self.load = np.array([308.3, -22.2, 449.2, 0, 0, 0])
            self.sol = self.lam.get_linear_response(self.load)
            self.sold = self.lamd.get_linear_response(self.load) 
            
    def test_polar_ex(self):
        print(clt.stiffness_polar(self.lam, 'Ex'))
        
    def test_polar_ey(self):
        print(clt.stiffness_polar(self.lam, 'Ey'))
        
    def test_polar_A11(self):
        print(clt.stiffness_polar(self.lam, 'A11'))
        
    def test_polar_error(self):
        self.assertRaises(ValueError, clt.stiffness_polar, self.lam, 'ey')
        

if __name__ == '__main__':
    unittest.main()