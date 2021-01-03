# -*- coding: utf-8 -*-
"""
Created on 05.10.2014
@author: Oliver

Mil-Hdbk-17-3F Table 5.3.3.2(b) provides values for the apparent modulus in bending for
different stacking sequences.

The tests in this module compare if the calculated 

* membrane modulus
* flexural modulus

are equal to the values given in MIL-HDBK-17, within a tolerance of 1%.

"""

from __future__ import division, print_function
import unittest
from clt import Laminate
from material import TransverseIsotropicPlyMaterial as Material

# create material
MAT = Material(E11=138000, E22=9700, G12=4500, nu12=0.31, t=0.14,
               name='T300/934')

# desired accuracy: 1%
TOL = 0.01 

# utility function -> 
def make_symm_stacking(half_stacking, mat):
    angles = half_stacking + list(reversed(half_stacking))
    return [(a, mat) for a in angles]

class GenericTest:
    
    half_sseq = NotImplemented # to be defined on subclass
    em_desired = 52900 # membrane modulus, same for all
    ef_desired = NotImplemented # to be defined on subclass
    
    def setUp(self):
        ss = make_symm_stacking(self.half_sseq, MAT)
        self.lam = Laminate(ss)    

    def test_symm(self):
        self.assertTrue(self.lam.is_symmetric())
        
    def test_em(self):
        em_actual = self.lam.Ex()
        delta = abs(em_actual - self.em_desired) / self.em_desired
        print(em_actual, self.ef_desired, delta)
        self.assertLessEqual(delta, TOL)

    def test_ef(self):
        ef_actual = self.lam.Exf()
        delta = abs(ef_actual - self.ef_desired) / self.ef_desired
        print(ef_actual, self.ef_desired, delta)
        self.assertLessEqual(delta, TOL)        


class BendingModulusMil17Case1(GenericTest, unittest.TestCase):
    half_sseq = [0, 0, 45, -45, 45, -45, 90, 90]
    ef_desired = 88200
    
class BendingModulusMil17Case2(GenericTest, unittest.TestCase):
    half_sseq = [0, 45, -45,90]*2
    ef_desired = 69600
    
class BendingModulusMil17Case3(GenericTest, unittest.TestCase):
    half_sseq = [45, -45, 0, 0, 45, -45, 90, 90]
    ef_desired = 53800
    
class BendingModulusMil17Case4(GenericTest, unittest.TestCase):
    half_sseq = [45, -45, 0, 90]*2
    ef_desired = 44900
    
class BendingModulusMil17Case5(GenericTest, unittest.TestCase):
    half_sseq = [45, -45, 45, -45, 0, 0, 90, 90]
    ef_desired = 30700
    
class BendingModulusMil17Case6(GenericTest, unittest.TestCase):
    half_sseq = [45, -45, 45, -45, 90, 90, 0, 0]
    ef_desired = 23600
    
class BendingModulusMil17Case7(GenericTest, unittest.TestCase):
    half_sseq = [90, 90, 45, -45, 45, -45, 0, 0]
    ef_desired = 22400
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()