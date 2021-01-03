# -*- coding: utf-8 -*-
"""
Created on 03.10.2014
@author: Oliver
"""

from __future__ import division, print_function
import unittest
from material import TransverseIsotropicPlyMaterial
from plainstrength import PlainStrengthAnalysisA, layer_strength_enveloppe_2d, offaxis_layer_strength
import numpy as np
from clt import MembraneLaminate
import matplotlib.pyplot as plt


# Sl = 55 instead of the 41 in the table. See text in Jones.
glass = TransverseIsotropicPlyMaterial(name='Glass/Epoxy, Jones Table 2-3',
        E11=54000., E22=18000., G12=9000., nu12=0.25, t=1,
        a11t=0, a22t=0,
        F11t=1035., F11c=1035., F22t=28., F22c=138., F12s=55.)

carbon = TransverseIsotropicPlyMaterial(name='Graphite/Epoxy, Jones Table 2-3',
        E11=207000., E22=5000., G12=2600., nu12=0.25, t=1,
        F11t=1035., F11c=689., F22t=41., F22c=117., F12s=69.)

mat = carbon
criterion = 'MaxStress'

desired = {(-1, -1, -1): mat.F12s,  # shear strength
           (-1, -1,  0): mat.F22c,  # transverse compressive strength
           (-1, -1,  1): mat.F12s,  # shear strength

           (-1, 0, -1): mat.F12s,  # shear strength
           (-1, 0,  0): mat.F11c,  # longitudinal compressive strength
           (-1, 0,  1): mat.F12s,  # shear strength

           (-1, 1, -1): mat.F22t,  # transverse tensile strength
           (-1, 1,  0): mat.F22t,  # transverse tensile strength
           (-1, 1,  1): mat.F22t,  # transverse tensile strength
           
           (0, -1, -1): mat.F12s,  # shear strength
           (0, -1,  0): mat.F22c,  # transverse compressive strength
           (0, -1,  1): mat.F12s,  # shear strength

           (0, 0, -1): mat.F12s,  # shear strength
           #(0, 0,  0): mat.F11c,  # longitudinal compressive strength
           (0, 0,  1): mat.F12s,  # shear strength

           (0, 1, -1): mat.F22t,  # transverse tensile strength
           (0, 1,  0): mat.F22t,  # transverse tensile strength
           (0, 1,  1): mat.F22t,  # transverse tensile strength               

           (1, -1, -1): mat.F12s,  # shear strength
           (1, -1,  0): mat.F22c,  # transverse compressive strength
           (1, -1,  1): mat.F12s,  # shear strength

           (1, 0, -1): mat.F12s,  # shear strength
           (1, 0,  0): mat.F11t,  # longitudinal tensile strength
           (1, 0,  1): mat.F12s,  # shear strength

           (1, 1, -1): mat.F22t,  # transverse tensile strength
           (1, 1,  0): mat.F22t,  # transverse tensile strength
           (1, 1,  1): mat.F22t,  # transverse tensile strength               
}

class OffaxisStrength(unittest.TestCase):
    # UD layer

    def test_tension_compression(self):    
        res_t = offaxis_layer_strength(mat, 'tension', criterion)
        res_c = offaxis_layer_strength(mat, 'compression', criterion)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        alpha = np.array([item[0] for item in res_t])
        strength_t = np.array([item[1] for item in res_t])
        strength_c = np.array([item[1] for item in res_c])
        ax.plot(alpha, strength_t, 'b-')
        ax.plot(alpha, strength_c, 'r-')
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlabel('alpha')
        ax.set_ylabel('strength')
        plt.show()
 

class Enveloppe2D(unittest.TestCase):
    # 2d enveloppes for UD layer
    
    def test_s1_s2(self):    
        res = layer_strength_enveloppe_2d(mat, 's1', 's2', criterion)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        values = np.array([item[0] for item in res])
        ax.plot(values[:,0], values[:,1], 'ro')
        ax.plot(values[:,0], values[:,1], 'r-')
        #print(res)
        ax.grid()
        ax.set_xlabel('s1')
        ax.set_ylabel('s2')
        plt.show()
        
    
    def test_s2_s6(self):    
        res = layer_strength_enveloppe_2d(mat, 's2', 's6', criterion)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        values = np.array([item[0] for item in res])
        ax.plot(values[:,1], values[:,2], 'ro')
        ax.plot(values[:,1], values[:,2], 'r-')
        #print(res)
        ax.grid()
        ax.set_xlabel('s2')
        ax.set_ylabel('s6')
        plt.show()
        
    def test_s1_s6(self):    
        res = layer_strength_enveloppe_2d(mat, 's1', 's6', criterion)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        values = np.array([item[0] for item in res])
        ax.plot(values[:,0], values[:,2], 'ro')
        ax.plot(values[:,0], values[:,2], 'r-')
        #print(res)
        ax.grid()
        ax.set_xlabel('s1')
        ax.set_ylabel('s6')
        plt.show()        
#################################################################################
# Only generic stuff down here

class _SimpleTest:
    
    def setUp(self):
        lam = MembraneLaminate([(0, mat)])
        self.psa = PlainStrengthAnalysisA(lam, criterion)
        
    def test_it(self):
        res = self.psa.min_rf(self.load, 0)
        self.assertAlmostEqual(res.r, desired[self.load])


class T_mmm(_SimpleTest, unittest.TestCase):
    load = (-1, -1, -1)

class T_mm0(_SimpleTest, unittest.TestCase):
    load = (-1, -1, 0)

class T_mmp(_SimpleTest, unittest.TestCase):
    load = (-1, -1, 1)

class T_m0m(_SimpleTest, unittest.TestCase):
    load = (-1, 0, -1)

class T_m00(_SimpleTest, unittest.TestCase):
    load = (-1, 0, 0)

class T_m0p(_SimpleTest, unittest.TestCase):
    load = (-1, 0, 1)

class T_mpm(_SimpleTest, unittest.TestCase):
    load = (-1, 1, -1)

class T_mp0(_SimpleTest, unittest.TestCase):
    load = (-1, 1, 0)

class T_mpp(_SimpleTest, unittest.TestCase):
    load = (-1, 1, 1)

##

class T_0mm(_SimpleTest, unittest.TestCase):
    load = (0, -1, -1)

class T_0m0(_SimpleTest, unittest.TestCase):
    load = (0, -1, 0)

class T_0mp(_SimpleTest, unittest.TestCase):
    load = (0, -1, 1)

class T_00m(_SimpleTest, unittest.TestCase):
    load = (0, 0, -1)

#class T_000(_SimpleTest, unittest.TestCase):
#    load = (0, 0, 0)

class T_00p(_SimpleTest, unittest.TestCase):
    load = (0, 0, 1)

class T_0pm(_SimpleTest, unittest.TestCase):
    load = (0, 1, -1)

class T_0p0(_SimpleTest, unittest.TestCase):
    load = (0, 1, 0)

class T_0pp(_SimpleTest, unittest.TestCase):
    load = (0, 1, 1)

##

class T_pmm(_SimpleTest, unittest.TestCase):
    load = (1, -1, -1)

class T_pm0(_SimpleTest, unittest.TestCase):
    load = (1, -1, 0)

class T_pmp(_SimpleTest, unittest.TestCase):
    load = (1, -1, 1)

class T_p0m(_SimpleTest, unittest.TestCase):
    load = (1, 0, -1)

class T_p00(_SimpleTest, unittest.TestCase):
    load = (1, 0, 0)

class T_p0p(_SimpleTest, unittest.TestCase):
    load = (1, 0, 1)

class T_ppm(_SimpleTest, unittest.TestCase):
    load = (1, 1, -1)

class T_pp0(_SimpleTest, unittest.TestCase):
    load = (1, 1, 0)

class T_ppp(_SimpleTest, unittest.TestCase):
    load = (1, 1, 1)

#    
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()