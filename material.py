# -*- coding: utf-8 -*-
"""
Created on 20.09.2014
@author: Oliver Koch (n52e09@gmail.com)

Notes on material properties

- nu12 is the larger one: -1 x resultant strain in 2 due to applied strain in 1 direction.
- first index: ursache, second index: wirkung
"""


from __future__ import division, print_function
import numpy as np
import math


NaN = np.NaN

# FIXME: introduce two classes of materials: one with a thickness property,
# the other one without.

# das ist jetzt eine beschraenkte definition fuer transvers isotrope
# werkstoffe, die nur plane-stress belastungen erfahren
# sollte als sub-klasse einer allgemeinen material-klasse definiert sein.
# VDI2014: nu_12 ist die kleinere, nu_21 die größere Querkontraktionszahl

# FIXME: add validation functions for material properties
# - formal validation
# - physical/mechanical validation


class TransverseIsotropicPlyMaterial(object):
    """Transverse isotropic material. 
    - Ply Material: has a thickness property
    - axis 1 is the fibre direction, 2 transverse to the fribre in the plane of the 
      lamina, 3 normal to the plane of the lamina.
    """
    
    _matkeys = ['composite', 'UD', 'carbon']

    def __init__(self, t=NaN, E11=NaN, E22=NaN, G12=NaN, nu12=NaN,
                 a11t=0, a22t=0, b11m=0, b22m=0, nu23=0.45,
                 F11t=NaN, F11c=NaN, F22t=NaN, F22c=NaN, F12s=NaN,
                 F33t=NaN, F33c=NaN, Fils=NaN,
                 fvc=NaN, # fibre volume content
                 name="", verbose_name="", **keywords):

        self._mattype = 'composite'
        self.name = name
        self.verbose_name = verbose_name

        self._t = float(t)
        assert math.isfinite(self._t) 
        self._E11 = abs(float(E11))
        self._E22 = abs(float(E22))
        self._G12 = abs(float(G12))
        self._nu12 = abs(float(nu12))
        self._nu23 = abs(float(nu23))

        self._a11t = float(a11t)
        self._a22t = float(a22t)
        self._a33t = float(a22t)

        self.b11m = float(b11m)
        self.b22m = float(b22m)
        self.b33m = float(b22m)

        self.F11t = abs(float(F11t))
        self.F11c = abs(float(F11c))
        self.F22t = abs(float(F22t))
        self.F22c = abs(float(F22c))
        self.F12s = abs(float(F12s))
        self.F33t = abs(float(F33t))
        self.F33c = abs(float(F33c))
        self.Fils = abs(float(Fils))

    def stiffness_matrix(self):
        """Return 6x6 3D stiffness matrix C of transverse isotropic material, 
        Axis 0 is the axis for rotational symmetry."""
        return np.linalg.inv(self.compliance_matrix())

    def compliance_matrix(self):
        """Return 6x6 3D compliance matrix S of transverse isotropic material
        Axis 0 is the axis for rotational symmetry."""
        # berthelot, 10.25
        S = np.zeros((6,6))
        S[0,0] = 1/self.E11
        S[0,1] = S[1,0] = -self.nu12/self.E11
        S[0,2] = S[2,0] = -self.nu13/self.E11
        S[1,1] = 1/self.E22
        S[1,2] = S[2,1] = -self.nu23/self.E22
        S[2,2] = 1/self.E33
        S[3,3] = 1/self.G23
        S[4,4] = 1/self.G13
        S[5,5] = 1/self.G12
        return S
    
    def compliance_matrix_3(self):
        """Return the 3x3 compliance matrix S for plane stress state.
        Components 1, 2, and 6.
        """
        return np.array([[1/self.E11, -self.nu21/self.E22, 0],
                          [-self.nu21/self.E22, 1/self.E22, 0],
                          [0, 0, 1/self.G12]], dtype=np.float32)

    @property
    def t(self):
        """Ply thickness"""
        return self._t

    @property
    def E11(self):
        """Modulus of elasticity, 1 direction."""
        return self._E11

    @property
    def E22(self):
        """Modulus of elasticity, 2-direction."""
        return self._E22

    @property
    def E33(self):
        """Modulus of elasticity in thickness (3) direction: E33 = E22 """
        return self._E22

    @property
    def G12(self):
        """In-plane shear modulus G12"""
        return self._G12

    @property
    def G13(self):
        """Transverse shear modulus G13 = G12"""
        return self._G12

    @property
    def G23(self):
        """Transverse shear modulus G23"""
        return 0.5*self._E22/(1. + self._nu23)

    @property
    def nu12(self):
        return self._nu12

    @property
    def nu13(self):
        return self._nu12

    @property
    def nu23(self):
        return self._nu23

    @property
    def nu32(self):
        return self._nu23

    @property
    def nu21(self):
        return self._nu12 * self._E22/self._E11

    @property
    def nu31(self): return self.nu21
    
    @property
    def a11t(self): return self._a11t
    
    @property
    def a22t(self): return self._a22t
    
    @property
    def a33t(self): return self._a22t
    
    @property
    def a12t(self): return 0.0
    
    def __repr__(self):
        return "Transverse Isotropic Ply Material - %s" % self.name

    def as_dict(self):
        return dict(name=self.name, E11=self.E11, E22=self.E22, G12=self.G12,
                    nu12=self.nu12, a11t=self.a11t, a22t=self.a22t,
                    b11m=self.b11m, b22m=self.b22m,
                    F11t=self.F11t, F11c=self.F11c, F22t=self.F22t,
                    F22c=self.F22c, F12s=self.F12s,
                    F33t=self.F33t, F33c=self.F33c, Fils=self.Fils,
                    verbose_name=self.verbose_name, t=self.t)

