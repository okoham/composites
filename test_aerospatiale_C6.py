# -*- coding: utf-8 -*-
"""
Created on 05.10.2014
@author: Oliver

Aeropatiale Composite Stress Manual: Using Tsai-Hill Failure criterion

"""

from __future__ import division, print_function
import unittest
import numpy as np
#from numpy.testing import assert_allclose
from material import TransverseIsotropicPlyMaterial as Material
from clt import Laminate
from plainstrength import PlainStrengthAnalysisA

# TODO: add other laminates types (MembraneLaminate, ...)

MAT = Material(name='T300/BSL914 (new)',
            E11=130000., E22=4650., G12=4650., nu12=0.35, t=0.13,
            F11t=1200., F11c=1000., F22t=50., F22c=120., F12s=75.)

angles = (45, -45, 0, 90, 0, 90, 45, -45, 0, 90, 90, 0, -45, 45, 90, 0, 90, 0, -45, 45)
SSEQ = [(a, MAT) for a in angles]

TOL = 0.005 # 

# expected RFs from example
Expected_RFs = {0: 1.40, 
                45: 1.42, 
                -45: 1.31, 
                90: 1.42}


class CltTestMTS006CReserveFactors:

    def setUp(self):
        # aerospatiale example, C6
        lam = Laminate(SSEQ)
        self.load = np.array([308.3, -22.2, 449.2, 0, 0, 0]) # N/mm & N
        self.psa = PlainStrengthAnalysisA(lam, 'TsaiHill')
        
    def test_rf(self):
        # 
        alpha = angles[self.lid]
        desired = Expected_RFs[alpha]
        sres = self.psa.minrf_in_layer(self.lid, self.load, 0)
        print(self.lid, alpha, desired, sres.r)
        self.assertAlmostEqual(sres.r, desired, delta=TOL)


class Layer0(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 0

class Layer1(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 1

class Layer2(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 2

class Layer3(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 3

class Layer4(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 4

class Layer5(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 5

class Layer6(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 6

class Layer7(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 7

class Layer8(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 8

class Layer9(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 9

class Layer10(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 10

class Layer11(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 11

class Layer12(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 12

class Layer13(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 13

class Layer14(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 14

class Layer15(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 15

class Layer16(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 16

class Layer17(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 17

class Layer18(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 18

class Layer19(CltTestMTS006CReserveFactors, unittest.TestCase):
    lid = 19


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()