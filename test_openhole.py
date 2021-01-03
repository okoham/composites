# -*- coding: utf-8 -*-
"""
Created on 25.09.2014
@author: Oliver
"""

from __future__ import division, print_function
import unittest
import math
import numpy as np
import openhole
from numpy.testing import assert_allclose
import material
import clt


class Sharma(unittest.TestCase):
    
    def setUp(self):
        graphite = material.TransverseIsotropicPlyMaterial(name='Graphite/Epoxy',
           E11=181000., E22=10300., G12=7170., nu12=0.28, t=1.0)
        glass = material.TransverseIsotropicPlyMaterial(name='Glass/Epoxy',
           E11=47400., E22=16200., G12=7000., nu12=0.26, t=1.0)
        
        self.lam_grahite_0 = clt.MembraneLaminate([(0, graphite)])   
        self.lam_grahite_90 = clt.MembraneLaminate([(90, graphite)])   
        self.lam_grahite_0_90 = clt.MembraneLaminate([(0, graphite), (90, graphite)])   
        self.lam_grahite_p_m = clt.MembraneLaminate([(45, graphite), (-45, graphite)])
        
        self.lam_glass_0 = clt.MembraneLaminate([(0, glass)])   
        self.lam_glass_90 = clt.MembraneLaminate([(90, glass)])   
        self.lam_glass_0_90 = clt.MembraneLaminate([(0, glass), (90, glass)])   
        self.lam_glass_p_m = clt.MembraneLaminate([(45, glass), (-45, glass)])
                
    def test_roots_graphite_0(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_grahite_0, 5)   
        actual = oh.roots()
        desired = [0.8566j, 4.8936j]
        assert_allclose(actual, desired, atol=1e-4)
        
    def test_roots_graphite_90(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_grahite_90, 5)   
        actual = oh.roots()
        desired = [0.2043j, 1.1674j]
        assert_allclose(actual, desired, atol=1e-4)        
        
    def test_roots_graphite_0_90(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_grahite_0_90, 5)   
        actual = oh.roots()
        desired = [0.2747j, 3.6403j]
        assert_allclose(actual, desired, atol=1e-4)      
        
    def test_roots_graphite_p_m(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_grahite_p_m, 5)   
        actual = oh.roots()
        desired = [-0.8597+0.5109j, 0.8597+0.5109j]
        assert_allclose(actual, desired, atol=1e-4)                

    def test_roots_glass_0(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_glass_0, 5)   
        actual = oh.roots()
        desired = [0.7139j, 2.3960j]
        assert_allclose(actual, desired, atol=1e-4)
        
    def test_roots_glass_90(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_glass_90, 5)   
        actual = oh.roots()
        desired = [0.4174j, 1.4007j]
        assert_allclose(actual, desired, atol=1e-4)        
        
    def test_roots_glass_0_90(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_glass_0_90, 5)   
        actual = oh.roots()
        desired = [0.4965j, 2.0142j]
        assert_allclose(actual, desired, atol=1e-4)      
        
    def test_roots_glass_p_m(self):
        oh = openhole.CircularHoleInfiniteAnisotropicPlate(self.lam_glass_p_m, 5)   
        actual = oh.roots()
        desired = [-0.6045+0.7966j, 0.6045+0.7966j]
        assert_allclose(actual, desired, atol=1e-4)                

class Example_pm45(unittest.TestCase):

    def setUp(self):
        mat = material.TransverseIsotropicPlyMaterial(name='T300/BSL914 (new)',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.5,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        sseq = [(a, mat) for a in [45,-45,-45,45]]
        lam = clt.MembraneLaminate(sseq)        
        self.radius = 20
        self.oh = openhole.EllipticHoleInfiniteAnisotropicPlate(lam, self.radius, self.radius)

    @unittest.skip('need a proper reference')
    def test_kt_inf(self):
        desired = np.array([3, 0, 0]) 
        actual = self.oh.unit_stress_uniax(0, 0, self.radius)
        print(actual)
        assert_allclose(actual, desired)
        
        
class Example_0_90(unittest.TestCase):

    def setUp(self):
        mat = material.TransverseIsotropicPlyMaterial(name='T300/BSL914 (new)',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.25,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        sseq = [(a, mat) for a in [0,90,90,0]]
        lam = clt.MembraneLaminate(sseq)        
        self.radius = 20
        self.oh = openhole.EllipticHoleInfiniteAnisotropicPlate(lam, self.radius, self.radius)

    @unittest.skip('need a proper reference')
    def test_kt_inf(self):
        desired = np.array([3, 0, 0]) 
        actual = self.oh.unit_stress_uniax(0, 0, self.radius)
        print(actual)
        assert_allclose(actual, desired)

class QIExample(unittest.TestCase):

    def setUp(self):
        mat = material.TransverseIsotropicPlyMaterial(name='T300/BSL914 (new)',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.25,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        sseq = [(a, mat) for a in [45,-45,0,90,90,0,-45,45]]
        self.lam = clt.MembraneLaminate(sseq)        
        self.radius = 20
        #self.oh = openhole.EllipticHoleInfiniteAnisotropicPlate(lam, self.radius, self.radius)

    def test_raises(self):
        self.assertRaises(openhole.IsotropicMaterialError, openhole.EllipticHoleInfiniteAnisotropicPlate, 
                          self.lam, self.radius, self.radius)
        

class MTS006Example1(unittest.TestCase):
    """Aerospatiale Stress Manual, Chapter K
    """
    
    def setUp(self):
        """Create test environment, isotropic Alu sheet, 1 mm thick, circular hole
        with 5 mm radius."""
        mat = material.TransverseIsotropicPlyMaterial(name='T300/BSL914 (new)',
           E11=62560., E22=34100., G12=18820., nu12=0.4191, t=0.25,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        # (6, 4, 4, 2)
        sseq = [(a, mat) for a in [45,-45,0,90,0,45,-45,0,0,-45,45,0,90,0,-45,45]]
        lam = clt.MembraneLaminate(sseq)        
        self.radius = 20
        self.oh = openhole.EllipticHoleInfiniteAnisotropicPlate(lam, self.radius, self.radius)

    #@unittest.skip('reference inaccurate')
    def test_kt_inf(self):
        # 3.28 vs. 3.07 -> is MTS006 that inaccurate???
        desired = np.array([3.28, 0, 0]) 
        actual = self.oh.unit_stress_uniax(0, 0, self.radius)
        print(actual)
        assert_allclose(actual, desired)
        
    def test_sx0_sx180(self):
        """for Nx loading, solution should be symmetric about y axis"""
        beta = 0
        x = self.radius + 3
        y = 0
        s0 = self.oh.unit_stress_uniax(beta, x, y)
        s180 = self.oh.unit_stress_uniax(beta, -x, y)
        assert_allclose(s0, s180)
        

class CircularOpenHoleIsotropic2(unittest.TestCase):
    """Verifies correctness of open hole solution for iso plates,
    using the anisotropic method!!!
    """
    
    def setUp(self):
        """Create test environment, isotropic Alu sheet, 1 mm thick, circular hole
        with 5 mm radius."""
        #E = 70000 
        #nu = 0.3
        #G = E / (2*(1+nu))
        #mat = material.TransverseIsotropicPlyMaterial(t=1, E11=E, E22=E, G12=G, nu12=nu)
        #lam = clt.MembraneLaminate([(0, mat)])
        self.radius = 5
        self.oh = openhole.CircularHoleInfiniteIsotropicPlate(self.radius)
                
    def test_uniax_kt3(self):
        #actual = self.oh.unit_stress_uniax(0, 0, self.radius)
        actual = self.oh.cart_stress(0, self.radius, [1, 0, 0])
        desired = [3, 0, 0]
        print(actual)
        assert_allclose(actual, desired, atol=1e-6)
        
    def test_uniax_sr_90(self):
        """sigma_r at 90°.
        ref. peterson (2nd), 4.7
        test this from r=a to r=4*a, 40 points
        """
        a = self.radius
        r = np.linspace(a, 5*a, 50)
        desired = 1.5*((a/r)**2 - (a/r)**4)
        farfield_load = [1, 0, 0]
        result1 = [self.oh.cart_stress(0, ri, farfield_load) for ri in r]
        # get s_y components
        actual = [x[1] for x in result1]
        assert_allclose(actual, desired, atol=1e-6)
        
    def test_uniax_stheta_90(self):
        """sigma_theta at 90°.
        ref. peterson (2nd), 4.7
        test this from r=a to r=4*a, 40 points
        """
        a = self.radius
        r = np.linspace(a, 5*a, 50)
        desired = 0.5*(2 + (a/r)**2 + 3*(a/r)**4)
        farfield_load = [1, 0, 0]
        result1 = [self.oh.cart_stress(0, ri, farfield_load) for ri in r]
        # get s_x components
        actual = [x[0] for x in result1]
        assert_allclose(actual, desired)   
        
    def test_uniax_tau_90(self):
        """sigma_r at 90°.
        ref. peterson (2nd), 4.7
        test this from r=a to r=4*a, 40 points
        """
        a = self.radius
        r = np.linspace(a, 5*a, 50)
        desired = np.zeros_like(r)
        farfield_load = [1, 0, 0]
        result1 = [self.oh.cart_stress(0, ri, farfield_load) for ri in r]
        # get tau_r_theta components
        actual = [x[2] for x in result1]
        assert_allclose(actual, desired, atol=1e-6)   

    def test_uniax_sr_0(self):
        """sigma_r at 0°.
        ref. peterson (2nd), 4.8
        test this from r=a to r=4*a, 40 points
        """
        a = self.radius
        r = np.linspace(a, 5*a, 50)
        desired = 0.5*(2 - 5*(a/r)**2 + 3*(a/r)**4)
        farfield_load = [1, 0, 0]
        result1 = [self.oh.cart_stress(ri, 0, farfield_load) for ri in r]
        # get s_r components
        actual = [x[0] for x in result1]
        print(actual)
        assert_allclose(actual, desired, atol=1e-6)   
        
    def test_uniax_stheta_0(self):
        """sigma_theta at 0°.
        ref. peterson (2nd), 4.8
        test this from r=a to r=4*a, 40 points
        """
        a = self.radius
        r = np.linspace(a, 10*a, 50)
        desired = 0.5*((a/r)**2 - 3*(a/r)**4)
        farfield_load = [1, 0, 0]
        result1 = [self.oh.cart_stress(ri, 0, farfield_load) for ri in r]
        # get s_theta components
        actual = [x[1] for x in result1]
        print(actual)        
        assert_allclose(actual, desired, atol=1e-6)     
        
    def test_uniax_tau_0(self):
        """tau along x axis°.
        ref. peterson (2nd), 4.8
        test this from r=a to r=4*a, 40 points
        """
        a = self.radius
        r = np.linspace(a, 10*a, 50)
        desired = np.zeros_like(r)
        farfield_load = [1, 0, 0]
        result1 = [self.oh.cart_stress(0, ri, farfield_load) for ri in r]
        # get tau_r_theta components
        actual = [x[2] for x in result1]
        print(actual)        
        assert_allclose(actual, desired, atol=1e-6)             


class CircularOpenHoleOrtho(unittest.TestCase):
    """Verifies correctness of open hole solution for ortho plates,
    using the anisotropic method!!!
    """
    
    def setUp(self):
        """Create test environment, isotropic Alu sheet, 1 mm thick, circular hole
        with 5 mm radius."""
        
        mat = material.TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.127,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        sseq = [(a, mat) for a in [45,-45,0,90,0,90,0,-45,45]]
        #self.lam = Laminate(sseq)
        lam = clt.MembraneLaminate(sseq)        
        #E = 70000 
        #nu = 0.3
        #G = E / (2*(1+nu))
        #mat = material.TransverseIsotropicPlyMaterial(t=1, E11=E, E22=E, G12=G, nu12=nu)
        #lam = clt6.MembraneLaminate([(0, mat)])
        self.radius = 5
        self.oh = openhole.EllipticHoleInfiniteAnisotropicPlate(lam, self.radius, self.radius)

class CircularOpenHoleIsotropic(unittest.TestCase):
    """Verifies correctness of open hole solution for iso plates
    """
    
    def test_uniax_kt3(self):
        """Test the tangential stress at r=a, theta=90: this is Kt=3"""
        a = 1234.6
        r = a 
        theta = math.radians(90)
        sx = 123
        expect = np.array([0, 3, 0]) * sx
        result = openhole.stress_hole_isotropic_x(a, r, theta, sx)
        print(result/sx)
        assert_allclose(result, expect)
        
    def test_uniax_sr_90(self):
        """sigma_r at 90°.
        ref. peterson (2nd), 4.7
        test this from r=a to r=4*a, 40 points
        """
        a = 5
        r = np.linspace(a, 4*a, 40)
        sx = 1
        expect = 1.5*((a/r)**2 - (a/r)**4)
        theta = math.radians(90)
        result1 = [openhole.stress_hole_isotropic_x(a, ri, theta, sx) for ri in r]
        # get s_r components
        result = [x[0] for x in result1]
        assert_allclose(result, expect)      
        
    def test_uniax_stheta_90(self):
        """sigma_theta at 90°.
        ref. peterson (2nd), 4.7
        test this from r=a to r=4*a, 40 points
        """
        a = 5
        r = np.linspace(a, 4*a, 40)
        sx = 1
        expect = 0.5*(2 + (a/r)**2 + 3*(a/r)**4)
        theta = math.radians(90)
        result1 = [openhole.stress_hole_isotropic_x(a, ri, theta, sx) for ri in r]
        # get s_theta components
        result = [x[1] for x in result1]
        assert_allclose(result, expect)
        
    def test_uniax_tau_90(self):
        """sigma_r at 90°.
        ref. peterson (2nd), 4.7
        test this from r=a to r=4*a, 40 points
        """
        a = 5
        r = np.linspace(a, 4*a, 40)
        sx = 1
        expect = np.zeros_like(r)
        theta = math.radians(90)
        result1 = [openhole.stress_hole_isotropic_x(a, ri, theta, sx) for ri in r]
        # get tau_r_theta components
        result = [x[2] for x in result1]
        assert_allclose(result, expect, atol=1e-15)     
        
        
    def test_uniax_sr_0(self):
        """tau at 0°.
        ref. peterson (2nd), 4.8
        test this from r=a to r=4*a, 40 points
        """
        a = 5
        r = np.linspace(a, 4*a, 40)
        sx = 1
        expect = 0.5*(2 - 5*(a/r)**2 + 3*(a/r)**4)
        theta = 0
        result1 = [openhole.stress_hole_isotropic_x(a, ri, theta, sx) for ri in r]
        # get s_r components
        result = [x[0] for x in result1]
        assert_allclose(result, expect)      
        
    def test_uniax_stheta_0(self):
        """sigma_theta at 0°.
        ref. peterson (2nd), 4.8
        test this from r=a to r=4*a, 40 points
        """
        a = 5
        r = np.linspace(a, 4*a, 40)
        sx = 1
        expect = 0.5*((a/r)**2 - 3*(a/r)**4)
        theta = 0
        result1 = [openhole.stress_hole_isotropic_x(a, ri, theta, sx) for ri in r]
        # get s_theta components
        result = [x[1] for x in result1]
        assert_allclose(result, expect)
        
    def test_uniax_tau_0(self):
        """sigma_r at 0°.
        ref. peterson (2nd), 4.8
        test this from r=a to r=4*a, 40 points
        """
        a = 5
        r = np.linspace(a, 4*a, 40)
        sx = 1
        expect = np.zeros_like(r)
        theta = 0
        result1 = [openhole.stress_hole_isotropic_x(a, ri, theta, sx) for ri in r]
        # get tau_r_theta components
        result = [x[2] for x in result1]
        assert_allclose(result, expect, atol=1e-15)   
        
    def test_biax_kt2(self):
        """sx = sy -> Kt = 2 at the boundary"""
        a = 23.4
        r = a 
        theta = math.radians(90)
        sx = 123
        expect = np.array([0, 2, 0]) * sx
        result = openhole.stress_hole_isotropic(a, r, theta, sx, sx, 0)
        print(result/sx)
        assert_allclose(result, expect)
        
    def test_shear_kt4(self):
        """sx = - sy -> Kt = 4 at the boundary"""
        a = 23.4
        r = a 
        theta = math.radians(90)
        sx = 123
        expect = np.array([0, 4, 0]) * sx
        result = openhole.stress_hole_isotropic(a, r, theta, sx, -sx, 0)
        print(result/sx)
        assert_allclose(result, expect)
        
class OpenHoleStress:
    
    def setUp(self):
        mat = material.TransverseIsotropicPlyMaterial(name='T300/BSL914 (new)',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.125,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        stacking = [(a, mat) for a in self.sseq]
        self.lam = clt.MembraneLaminate(stacking)        
        self.dia = 6.35
        self.d0 = 0.5
        
    def test_Nx_tension(self):
        # critial position at theta = 0?
        load = [100, 0, 0]
        res = openhole.open_hole_stress(self.lam, load, self.dia, self.d0)
        for line in res:
            print(line)
    
    def test_Nx_compression(self):
        # critial position at theta = 0?
        load = [-100, 0, 0]
        res = openhole.open_hole_stress(self.lam, load, self.dia, self.d0)
        for line in res:
            print(line)

    def test_Ny_tension(self):
        # critial position at theta = 0?
        load = [0, 100, 0]
        res = openhole.open_hole_stress(self.lam, load, self.dia, self.d0)
        for line in res:
            print(line)
    
    def test_Ny_compression(self):
        # critial position at theta = 0?
        load = [0, -100, 0]
        res = openhole.open_hole_stress(self.lam, load, self.dia, self.d0)
        for line in res:
            print(line)

    def test_Nxy(self):
        # critial position at theta = 0?
        load = [0, 0, 100]
        res = openhole.open_hole_stress(self.lam, load, self.dia, self.d0)
        for line in res:
            print(line)        
    
class OpenHoleStressQI(OpenHoleStress, unittest.TestCase):
    sseq = [45,-45,0, 90, 90, 0, -45,45]

class OpenHoleStressOR(OpenHoleStress, unittest.TestCase):
    # FIXME: stresses are incorrect...
    # 6, 2, 2, 2
    sseq = [45,-45,0,0, 90, 0, 0, 90, 0,0, -45,45]
    
    
        
class OpenHoleStrength:
    
    def setUp(self):
        mat = material.TransverseIsotropicPlyMaterial(name='T300/BSL914 (new)',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.125,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        stacking = [(a, mat) for a in self.sseq]
        self.lam = clt.MembraneLaminate(stacking)        
        self.dia = 6.35
        self.d0 = 0.5
        
    def test_Nx_tension(self):
        # critial position at theta = 0?
        load = [100, 0, 0]
        res = openhole.open_hole_strength(self.lam, load, self.dia, self.d0, self.theory)
        for line in res:
            print(line)
    
    def test_Nx_compression(self):
        # critial position at theta = 0?
        load = [-100, 0, 0]
        res = openhole.open_hole_strength(self.lam, load, self.dia, self.d0, self.theory)
        for line in res:
            print(line)

    def test_Ny_tension(self):
        # critial position at theta = 0?
        load = [0, 100, 0]
        res = openhole.open_hole_strength(self.lam, load, self.dia, self.d0, self.theory)
        for line in res:
            print(line)
    
    def test_Ny_compression(self):
        # critial position at theta = 0?
        load = [0, -100, 0]
        res = openhole.open_hole_strength(self.lam, load, self.dia, self.d0, self.theory)
        for line in res:
            print(line)

    def test_Nxy(self):
        # critial position at theta = 0?
        load = [0, 0, 100]
        res = openhole.open_hole_strength(self.lam, load, self.dia, self.d0, self.theory)
        for line in res:
            print(line)


# class OpenHoleStrengthQIHashinA(OpenHoleStrength, unittest.TestCase):
#     sseq = [45,-45,0, 90, 90, 0, -45,45]
#     theory = 'Hashin A'
        
#class OpenHoleStrengthQIAirbus(OpenHoleStrength, unittest.TestCase):
#    sseq = [45,-45,0, 90, 90, 0, -45,45]
#    theory = 'Airbus'

# class OpenHoleStrengthORAirbus(OpenHoleStrength, unittest.TestCase):
#     # FIXME: stresses seem to be incorrect...
#     # 6, 2, 2, 2
#     sseq = [45,-45,0,0, 90, 0, 0, 90, 0,0, -45,45]
#     theory = 'Airbus'
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()