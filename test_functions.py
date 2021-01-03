# -*- coding: utf-8 -*-
"""
Created on 06.10.2014
@author: Oliver

non-technical tests for basic classes
"""

from __future__ import division, print_function
import unittest
#import numpy as np 
#from numpy.testing import assert_allclose
#import math
import clt
#from clt import Laminate
from material import TransverseIsotropicPlyMaterial as Material
#import plainstrength

MAT = Material(name='Dummy Carbon', t=0.27,
                     Ex=100000, Ey=10000, Gxy=5000, nuxy=0.3)

# Material
# - change property
# set unplanned property

class TestMaterialPropertyChange(unittest.TestCase):
    
    def setUp(self):
        self.m = MAT
        
    @unittest.skip('not implemented')
    def test_set_something(self):
        self.assertRaises(AttributeError, setattr, self.m, 'otto', 2000)
        
    def test_set_E11(self):
        self.assertRaises(AttributeError, setattr, self.m, 'E11', 2000)
        
    def test_set_E22(self):
        self.assertRaises(AttributeError, setattr, self.m, 'E22', 2000)
        
    def test_set_E33(self):
        self.assertRaises(AttributeError, setattr, self.m, 'E33', 2000)
        
    def test_set_G12(self):
        self.assertRaises(AttributeError, setattr, self.m, 'G12', 2000)
        
    def test_set_G13(self):
        self.assertRaises(AttributeError, setattr, self.m, 'G13', 2000)
        
    def test_set_G23(self):
        self.assertRaises(AttributeError, setattr, self.m, 'G23', 2000)
        
    def test_set_nu12(self):
        self.assertRaises(AttributeError, setattr, self.m, 'nu12', 2000)
        
    def test_set_nu21(self):
        self.assertRaises(AttributeError, setattr, self.m, 'nu21', 2000)
        
    def test_set_nu13(self):
        self.assertRaises(AttributeError, setattr, self.m, 'nu13', 2000)
        
    def test_set_nu31(self):
        self.assertRaises(AttributeError, setattr, self.m, 'nu31', 2000)
        
    def test_set_nu23(self):
        self.assertRaises(AttributeError, setattr, self.m, 'nu23', 2000)
        
    def test_set_nu32(self):
        self.assertRaises(AttributeError, setattr, self.m, 'nu32', 2000)
        
    def test_set_a11t(self):
        self.assertRaises(AttributeError, setattr, self.m, 'a11t', 2000)       
        


# Layer
# - initialise with nonsense properties
# - change angle
# - change material
# - change laminate reference
# - set property that does not belong there
# - all functions: return proper type
# - all functions: raise error if incorrect type of parameters


class TestLayer(unittest.TestCase):
    
    def setUp(self):
        self.lam = clt.Laminate([(0, MAT), (90, MAT)])
        
    def test_initialise_complex_angle(self):
        #layer = clt.Layer(self.lam, 1+2j, MAT)
        self.assertRaises(TypeError, clt.Layer, self.lam, 1+2j, MAT)
        
    @unittest.skip('ääähhh... do I need to change that ?')
    def test_change_angle(self):
        layer = self.lam.layers[0]
        # SHIT! I can replace the angle function with an integer...
        setattr(layer, 'angle', 23)
        print(layer.angle, layer.angle())
        self.assertRaises(TypeError, setattr, layer, 'angle', 23)
        
    @unittest.skip('not implemented')
    def test_set_something(self):
        layer = self.lam.layers[0]
        self.assertRaises(AttributeError, setattr, layer, 'otto', 23)

    @unittest.skip('unfair - changing a pseudo-private attribute')
    def test_change_laminate_reference(self):
        other_laminate = clt.Laminate([(34, MAT), (42, MAT)])
        layer = self.lam.layers[0]
        self.assertRaises(AttributeError, setattr, layer, '_laminate', other_laminate)
        
    def test_thickness_return_type(self):
        layer = self.lam.layers[0]
        t = layer.thickness()
        self.assertIsInstance(t, float)
        
    def test_angle_return_type(self):
        layer = self.lam.layers[0]
        result = layer.angle()
        self.assertIsInstance(result, float)
        
    

class TestLaminate(unittest.TestCase):

# Laminate
# - try to initialise abstract superclass
# - initialise with 0 layers
# - initialise with something not a sseq defintion
# - add layer
# - change layer attribute
# - remove layer
# - set property that is not planned
# - validate_layer function: always return error except for proper use
# - get_z: test nonsense arguments
# - A, Ex, ... : test nonsense arguments for offaxis
# - thermal_force, swelling force: 
# - all functions: return proper type
# - solve: raise error if singular
# - degradation state: cannot have fibre failure without matrix failure? 

    @unittest.skip('not implemented')
    def test_init_superclass(self):
        raise NotImplementedError
    
    def test_init_without_layers(self):
        self.assertRaises(ValueError, clt.Laminate, [])
        
    def test_init_with_nonsense(self):
        self.assertRaises(ValueError, clt.Laminate, [1, 2, 5])
        
    @unittest.skip('not implemented')
    def test_set_illegal_property(self):
        lam = clt.Laminate([(0, MAT), (90, MAT)])
        self.assertRaises(AttributeError, setattr, lam, 'otto', 2000)

    def test_add_layer(self):
        lam = clt.Laminate([(0, MAT), (90, MAT)])
        newlayer = clt.Layer(lam, 45, MAT)
        self.assertRaises(TypeError, tuple.__add__, lam.layers, newlayer)      
        
    def test_replace_layers(self):
        lam = clt.Laminate([(0, MAT), (90, MAT)])
        newlayers = (clt.Layer(lam, 45, MAT), clt.Layer(lam, -45, MAT))
        self.assertRaises(AttributeError, setattr, lam, 'layers', newlayers)
    
    
# Response
# - change internal attributes
# - set attributes that don't belong there
# - all functions: return proper type and shape
# - call functions with nonsense arguments
# - response is created for an undamaged laminate, then someone changes the 
#   damage state of the laminate -> is that possible? does it lead to wrong 
#   results? 
#   



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()