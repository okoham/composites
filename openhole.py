# -*- coding: utf-8 -*-
"""
Created on 25.09.2014
@author: Oliver

functions for stress distributions in plates with holes
"""

from __future__ import division, print_function
import numpy as np
import math
import cmath
from clt import tmat_sig
from plainstrength import failure_analysis_b

def finite_width_correction():
    pass

def stress_hole_isotropic_x(a, r, theta, sigmax):
    """Return stress at point (r, theta): [sigma_r, sigma_theta, tau_r_theta]!
    Hole of radius a in infinite isotropic plate
    Timoshenko/Goodier, Theory of Elasticity.
    
    :Parameters:
    - r, theta ... coordinates at which stress is to be calculated, theta in radians 
    - a ... hole radius
    - sigmax ... uniform stress in x direction
    
    """
    assert a > 0
    assert r >= a 
     
    ar = a/r
    k1 = 0.5*(1 - ar**2) + 0.5*(1 + 3*ar**4 - 4*ar**2)*math.cos(2*theta)
    k2 = 0.5*(1 + ar**2) - 0.5*(1 + 3*ar**4)*math.cos(2*theta)
    k6 = -0.5*(1 - 3*ar**4 + 2*ar**2)*math.sin(2*theta)
    return sigmax * np.array([k1, k2, k6])
    
    
def stress_hole_isotropic(a, r, theta, s1, s2, s6):
    # s1 -> evaluate at theta
    # s2 -> evaluate at theta-90
    # s6 -> +s6 at theta-45 & -s6 at theta+45 
    
    s = stress_hole_isotropic_x(a, r, theta, s1)
    s += stress_hole_isotropic_x(a, r, theta-math.pi/2, s2)
    s += stress_hole_isotropic_x(a, r, theta-math.pi/4, s6/2)
    s -= stress_hole_isotropic_x(a, r, theta+math.pi/4, -s6/2)
    return s


def kty_circular_hole_orthotropic(A):
    # kollar, 10.74
    A11, A12, A16 = A[0]
    A22 = A[1,1]
    A26 = A[1,2]
    A66 = A[2,2]
    assert A16 == 0 and A26 == 0, 'Material not orthotropic' 
    
    temp = math.sqrt(A11*A22) - A12 + (A11*A22 - A12**2)/(2*A66)
    temp *= 2/A11
    return 1 + math.sqrt(temp) 
    
class IsotropicMaterialError(Exception):
    pass
    
class EllipticHoleInfiniteAnisotropicPlate(object):
    """Model for an infinite anisotropic plate with an elliptic hole.
    Pure elastic membrane analysis, 2D.
    Hole axes parallel to plate x, y axes. 
    Lehknitskij
    NOTE: this model does >>not<< works for isotropic (or quasi-isotropic) materials. 
    For these materials, the characteristic equation has identical roots: s1 = s2 = j.
    Need to switch to the isotropic model then.
    """ 
    def __init__(self, lam, a, b):
        # TODO: dont need the laminate, just A matrix. Change this.
        assert a > 0
        assert b > 0
        self._a = a 
        self._b = b 
        self._lam = lam
        self._roots = self.roots()
        if np.allclose(self._roots[0], self._roots[1], atol=1e-3):
            raise IsotropicMaterialError('roots equal')

    def kt_x(self):
        """Return elastic stress concentration factor, for loading in x."""
        pass
    
    def kt_y(self):
        pass
    
    def kt_shear(self):
        pass
    
    def kt_complex(self, stress):
        """Return stress concentration factor (max stress at boundary) for 
        combined load state (Nx, Ny, Nxy)."""
        pass
    
    def roots(self):
        """return the roots of the characteristic equation.
        possible cases for roots:
        Case 1:
            mu1 = a1 + j*b1
            mu2 = a2 + j*b2
            mu3 = cc(mu1)
            mu4 = cc(mu2)
        Case 2:
            mu1 = mu2 = a + j*b
            mu3 = mu4 = cc(mu1)
        For the orthotropic case (A16 = A26 = 0): purely imaginary roots, ai = 0 
        
        """
        # LTH-FL 33100-05, or better: lehknitskij
        # NOTE: this is for fluxes, not stresses
        # get coefficients out of compliance matrix
        a = np.linalg.inv(self._lam.A())
        c4 = a[0,0] # a11
        c3 = -2*a[0,2] # -2*a16
        c2 = 2*a[0,1] + a[2,2]  # 2*a12 + a66
        c1 = -2*a[1,2] # -2*a26
        c0 = a[1,1] # a22
        # build polynomial and calculate roots
        # TODO: that algorithm is not too accurate -> find anything better??
        poly = np.polynomial.Polynomial([c0, c1, c2, c3, c4])
        roots = poly.roots()
        # verify that they are correct: value <= TOL
        TOL = 1e-9
        assert all([abs(poly(r)) < TOL for r in roots])
        # the 4 roots are either complex, then there is two pairs of congujate complex numbers.
        # Or they are purely imaginary. 
        # In either case we want the roots with positive imaginary parts only.
        #print('number of roots found:', len(roots))
        #print(roots)
        proots = roots[roots.imag > 0]
        
        # make sure its 2 roots. Should be ...
        assert len(proots) == 2
        # set real part to zero if < 1e-9
        proots.real[np.abs(proots.real) < 1e-9] = 0
        return np.sort_complex(proots)
        #r1 = np.max(proots)
        #r2 = np.min(proots)
        #return np.array([r1, r2])
  

    def cart_stress(self, x, y, nxy_ff):
        """Return stresses (Nx, Ny, Nxy) at point (x,y) due
        to far field stress sxy.
        
        :Parameters:
        - `sxy` (3,) array of far field fluxes, cartesic coordinate system, s_x, s_y, tau_xy
        """
        # FIXME: real stress state: 
        # Nx,farfield -> p=Nx, beta=0
        # Ny, farfield -> p=Ny, beta=90
        # Nxy, farfield -> p = Nxy, beta=45
        #                  p = -Nxy, beta=-45
        # requires 4 calculations        
        # different formulation: principal loading directions
        # requires only 2 calculations!   
        sx, sy, tau = nxy_ff
        s = sx * self.unit_stress_uniax(0, x, y)
        s += sy * self.unit_stress_uniax(math.pi/2, x, y)
        s += tau * self.unit_stress_uniax(math.pi/4, x, y)
        s -= tau * self.unit_stress_uniax(-math.pi/4, x, y)
        return s

    def unit_stress_uniax_2(self, phi, x, y):
        m = math.cos(phi)
        n = math.sin(phi)
        a = self._a
        b = self._b
        # Lekhnitskii, eq. 38.12
        alpha1 = -0.5*n * (a*n - 1j*b*m)
        beta1 = 0.5*m * (a*n - 1j*b*m)
        mu1, mu2 = self._roots
        z1 = x + mu1*y 
        z2 = x + mu2*y
         
        A1 = (beta1 - mu2*alpha1) / (mu1-mu2)
        A2 = (beta1 - mu1*alpha1) / (mu1-mu2)
        B1 = a - 1j*mu1*b 
        B2 = a - 1j*mu2*b
        C1 = cmath.sqrt(z1**2 - a**2 - mu1**2*b**2)
        C2 = cmath.sqrt(z2**2 - a**2 - mu2**2*b**2)
        Phi1prime = - A1*B1 / (C1*(z1 + C1))
        Phi2prime = A2*B2 / (C2*(z2 + C2))

        sx = m**2 + 2*(mu1**2*Phi1prime + mu2**2*Phi2prime).real
        sy = n**2 + 2*(Phi1prime + Phi2prime).real
        tau = m*n - 2*(mu1*Phi1prime + mu2*Phi2prime).real
        
        return np.array([sx, sy, tau])

    # FIXME: THIS IS WRONG!!!
    # for circular hole and loading direction beta=0:
    # sx @ 0° != sx @ 180°
    def unit_stress_uniax(self, beta, x, y):
        """Return unit fluxes due to uniaxial unit loading in direction beta."""
        # will actually return fluxes
        # beta in radians
        # s1, s2, are the two roots
        
        s1, s2 = self._roots
        z1 = x + s1*y 
        z2 = x + s2*y 
        a, b = self._a, self._b

        n = cmath.sin(beta)
        n2 = cmath.sin(2*beta)
        m = cmath.cos(beta)

        # calculate Phi_0_prime (z1)
        kappa1 = b * (s2*n2 + 2*m**2) + 1j*a * (2*s2*n**2 + n2)
        # FIXME: the s1-s2 term make a problem if roots are identical ...
        phi = -1j*kappa1 / (4 * (s1-s2) * (a + 1j*s1*b))
        phi *= 1 - z1/cmath.sqrt(z1**2 - a**2 - s1**2*b**2)

        # calculate Psi_0_prime (z2)
        kappa2 = b * (s1 * n2 + 2*m**2) + 1j*a * (2*s1*n**2 + n2)
        psi = -1j*kappa2 / (4 * (s2-s1) * (a + 1j*s2*b))
        psi *= 1 - z2/cmath.sqrt(z2**2 - a**2 - s2**2*b**2)

        f0 = phi + psi
        f1 = s1*phi + s2*psi
        f2 = s1**2*phi + s2**2*psi
        
        nx = m**2 + 2*f2.real
        ny = n**2 + 2*f0.real
        nxy = m*n - 2*f1.real
        
        # FIXME: remove this as soon as function works correctly
        #print(x, y, beta, f2.real, f0.real, f1.real)
        
        return np.array([nx, ny, nxy])
    
    
# FIXME: this delivers incorrect stresses for 90 ... 270°
class CircularHoleInfiniteAnisotropicPlate(EllipticHoleInfiniteAnisotropicPlate):
    
    def __init__(self, lam, r):
        self._a = self._b = r 
        self._lam = lam
        self._roots = self.roots()
        if np.allclose(self._roots[0], self._roots[1], atol=1e-3):
            raise IsotropicMaterialError('roots equal')   
        
class CircularHoleInfiniteAnisotropicPlateSharmea(CircularHoleInfiniteAnisotropicPlate):
    """Stress concentrations around Circular/Elliptical/Triangular Cutouts in 
    Infinite Composite Plates. Dharmendra S. Sharma, Proceedings of the World 
    Congress on Engineering 2011 Vol III. WCE 2011, July 6-8, 2011, London, U.K.
    """
    pass
    # choose the value of biaxial load factor lambda
    # choose value of load angle, beta
    # calculate compliance coefficient aij
    # calculate complex parameters of anisotropy s1 = alpha1 + j beta1 and s2
    # calculate constants a1, b1, a2, b2, B, B', C', K1, K2, K3, K4, a3, b3, a4, b4
    # evaluate stress functions and their derivatives 
    # evaluate stresses
        
class CircularHoleInfiniteIsotropicPlate(object):
    # Timoshenko, theory of elasticity
    
    def __init__(self, radius):
        assert radius > 0
        self._radius = radius
        
    def kt_x(self):
        """Return elastic stress concentration factor, for loading in x.
        max value along boundary"""
        return 3.0
    
    def kt_y(self):
        return 3.0
    
    def kt_shear(self):
        pass
    
    def kt_complex(self, stress):
        """Return stress concentration factor (max stress at boundary) for 
        combined load state (Nx, Ny, Nxy)."""
        pass
    
    def unit_stress_uniax(self, beta, x, y):
        """Return unit fluxes due to uniaxial unit loading in direction beta."""    
        pass
    
    def cart_stress(self, x, y, nxy_ff):
        """Return stress (sx, sy, tauxy) at point (x, y) for given far field 
        stress (sx_ff, sy_ff, tauxy_ff)
        """
        r = math.sqrt(x**2 + y**2)
        assert r >= self._radius
        theta = math.atan2(y, x)
        s1, s2, s6 = nxy_ff
        srt = stress_hole_isotropic(self._radius, r, theta, s1, s2, s6)
        tsi = np.linalg.inv(tmat_sig(theta))
        return np.dot(tsi, srt)
    
def open_hole_stress(lam, nxy_ff, hole_dia, d0, dtemp=0.0):
    """Return stresses around hole."""
        
    # create open hole instance
    radius = hole_dia/2
    try:
        hole = CircularHoleInfiniteAnisotropicPlate(lam, radius)
    except IsotropicMaterialError:
        hole = CircularHoleInfiniteIsotropicPlate(radius)
        
    angles = np.arange(0, 360, 5) # 5° step
    
    all_res = []
    
    for theta in angles:
        theta_rad = np.radians(theta)
        x = (radius + d0) * math.cos(theta_rad)
        y = (radius + d0) * math.sin(theta_rad)
        ntheta = hole.cart_stress(x, y, nxy_ff)
        # dtemp = 0
        #strength = failure_analysis_b(lam, ntheta, dtemp, theory)
        all_res.append((theta, ntheta))
        
    return all_res

def open_hole_strength(lam, nxy_ff, hole_dia, d0, theory, dtemp=0.0):
    """Analyse open hole strength using strength analysis B method.
    evaluate stresses at hole edge plus distance d0
    """
        
    # create open hole instance
    radius = hole_dia/2
    try:
        hole = CircularHoleInfiniteAnisotropicPlate(lam, radius)
    except IsotropicMaterialError:
        hole = CircularHoleInfiniteIsotropicPlate(radius)
        
    angles = np.arange(0, 360, 5) # 5° step
    
    all_res = []
    
    for theta in angles:
        theta_rad = np.radians(theta)
        x = (radius + d0) * math.cos(theta_rad)
        y = (radius + d0) * math.sin(theta_rad)
        ntheta = hole.cart_stress(x, y, nxy_ff)
        # dtemp = 0
        strength = failure_analysis_b(lam, ntheta, dtemp, theory)
        all_res.append((theta, strength))
        
    return all_res
    

if __name__ == '__main__':
    pass