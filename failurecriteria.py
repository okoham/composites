"""
Created on 20.09.2014
@author: Oliver

Methods for strength analysis for PLANE STRESS in UD REINFORCED LAYERS!
"""

from __future__ import division, print_function
import math
import numpy as np

NAN = np.nan


###############################################################################

class ResidualStressFailure(Exception):
    """Error raised if failure occurs due to residual stresses alone."""
    pass

def solve_quadratic(sigma12m, sigma12r, F11, F22, F12, F66, F1, F2):
    """Return Strength/Stress Ratio. Solves a quadratic equation for Strength 
    ratio. includes mechanical stresses (to be scaled), and residual stresses 
    which are kept constant.
    
    Solves the equation
    
    .. math::
    
      a 
    
    Equation to solve (for r):
    F11*(r*s1m + s1r)**2 + F22*(r*s2m + s2r)**2 + F66*(r*s6m + s6r)**2 + \
    F12*(r*s1m + s1r)*(r*s2m + s2r) + F1*(r*s1m + s1r) + F2*(r*s2m + s2r) = 1
    
    F11, ... are the coefficients
    
    Tsai, Ch. 9
    """
    s1m, s2m, s6m = sigma12m
    s1r, s2r, s6r = sigma12r
    am = F11*s1m**2 + F12*s1m*s2m + F22*s2m**2 + F66*s6m**2
    ar = F11*s1r**2 + F12*s1r*s2r + F22*s2r**2 + F66*s6r**2
    bm = F1*s1m + F2*s2m
    br = F1*s1r + F2*s2r
    bmix = 2*(F11*s1m*s1r + F12*s1m*s2r + F12*s2m*s1r + F22*s2m*s2r + F66*s6m*s6r)
    a = am
    b = bm + bmix
    c = -1 + ar + br
    # FIXME: raise some meaningful error if a == 0!
    # can be due to two reasons: 
    # - all stresses zero
    # - all quadratic coefficients zero
    # alternatively: return solution for the linear equation 
    # FIXME: make sure that solution is real, and positive -> makes no sense otherwise
    return -b/(2*a) + math.sqrt((b/(2*a))**2 - c/a)


class PlaneStressFailureCriterion(object):
    """Superclass for plane Stress failure criteria. 
    
    Subclasses must implement the `r` method.
    Subclasses must also override the attribute `failure_type`, with "FF" for 
    fibre failure anf "MF" for matrix failure. Criteria that do not this 
    distinction (e.g. Tsai-Wu) should keep the default value (None).  
    """
    
    failure_type = None
    
    # abstract methods that need to be implemented by subclasses 
    
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        """Return the strength ratio and failure mode for tensile strength in 1-direction 
        
        :Parameters:
        - mat: TransverseIsotropicPlyMaterial instance
        - sigma12m: stresses in principal material coordinates (\sigma_1^m, \sigma_2^m,
                    \sigma_6^m) due to mechanical load. np.array (3,), float
        - sigma12r: residual stresses in principal material coordinates (\sigma_1^r, 
                    \sigma_2^r, \sigma_6^r). np.array (3,), float. Default zero.
        
        :Returns:
        - r: strength ratio, float, r >= 0, or `None` if tensile failure cannot
                occur under the given load conditions.
        - fmode: string describing the failure mode. 
         
        """        
        raise NotImplementedError
    
    # Concrete methods implemented here 
    
    def is_matrix_failure(self):
        """Return True if this failure criterion describes matrix failure."""
        return self.failure_type == 'MF'
    
    def is_fibre_failure(self):
        """Return True if this failure criterion describes fibre failure."""
        return self.failure_type == 'FF'

    def fi(self, mat, sigma12m, sigma12r=np.zeros(3)):
        """Return the Failure Index."""
        r, fmode = self.r(mat, sigma12m, sigma12r=np.zeros(3))
        if r is None:
            return (None, fmode)
        elif r == 0:
            return (float('inf'), fmode)
        else:
            return (1/r, fmode)
    
    def critical_stress(self, mat, sigma12m, sigma12r=np.zeros(3)):
        """Return the critical stress that leads to failure."""
        r, fmode = self.r(mat, sigma12m, sigma12r)
        if r is not None:
            return (sigma12r + sigma12m*r, fmode) 
        else:
            return (None, fmode)    
        
    
###############################################################################  

"""
Maximum Stress Criterion
------------------------

Failure is considered to occur when any one of the three stresses \sigma_1,
\sigma_2 or \sigma_6 reaches the allowable value:

.. math::
    \sigma_1 = X_t or -X_c
    or \sigma_2 = Y_t or -Y_c
    or abs \sigma_6 = S_l
    
The maximum stress criterion is implemented with 5 individual classes 

.. math::
    \left( \frac{\sigma_1}{X_t} \right)^2 = 1

.. math::
    \left( \frac{\sigma_1}{X_c} \right)^2 = 1

.. math::
    \left( \frac{\sigma_2}{Y_t} \right)^2 = 1

.. math::
    \left( \frac{\sigma_2}{Y_c} \right)^2 = 1

.. math::
    \left( \frac{\sigma_6}{S_l} \right)^2 = 1

For UD plies, failure in 1 direction is considered fibre failure. 
Failure in 2 direction and shear failure are considered matrix failures. 

"""

class MaxStressTension1(PlaneStressFailureCriterion):
    """Maximum Stress failure criterion: tensile strength in 1-direction.
    This is a fibre failure criterion.
    """
    failure_type = 'FF'
    
    
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11m = sigma12m[0]
        s11r = sigma12r[0]    
        F11 = 1/mat.F11t**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, 0, 0, 0)  
        r = rtent if (rtent*s11m + s11r) >= 0 else None  
        return (r, 'MaxStress_T1_FF')


class MaxStressCompression1(PlaneStressFailureCriterion):
    """Maximum Stress failure criterion: compressive strength in 1-direction.
    This is a fibre failure criterion.
    """
    failure_type = 'FF'

    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11m = sigma12m[0]
        s11r = sigma12r[0]    
        F11 = 1/mat.F11c**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, 0, 0, 0)  
        r = rtent if (rtent*s11m + s11r) < 0 else None  
        return (r, 'MaxStress_C1_FF')    


class MaxStressTension2(PlaneStressFailureCriterion):
    """Maximum Stress failure criterion: Tensile strength in 2-direction.
    This is a matrix failure criterion.
    """    
    failure_type = 'MF'

    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s22m = sigma12m[1]
        s22r = sigma12r[1]
        F22 = 1/mat.F22t**2
        rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, 0, 0, 0)  
        r = rtent if (rtent*s22m + s22r) >= 0 else None  
        return (r, 'MaxStress_T2_MF')    


class MaxStressCompression2(PlaneStressFailureCriterion):
    """Maximum Stress failure criterion: Compressive strength in 2-direction.
    This is a matrix failure criterion.
    """    
    failure_type = 'MF'

    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s22m = sigma12m[1]
        s22r = sigma12r[1]    
        F22 = 1/mat.F22c**2
        rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, 0, 0, 0)  
        r = rtent if (rtent*s22m + s22r) < 0 else None  
        return (r, 'MaxStress_C2_MF')


class MaxStressShear(PlaneStressFailureCriterion):
    """Maximum Stress failure criterion: Shear strength.
    This is a matrix failure criterion.
    """    
    failure_type = 'MF'
    
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)): 
        F66 = 1/mat.F12s**2
        r = solve_quadratic(sigma12m, sigma12r, 0, 0, 0, F66, 0, 0)  
        return (r, 'MaxStress_S_MF')
    
        

###############################################################################

# MAX STRAIN
# convert stresses to strains
# convert strengths to strain allowables
# use quadratic criterion on strains.
# criterion is applicable if sign of resultant strain matches the stength type

class MaxStrainTension1(PlaneStressFailureCriterion):
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        S = mat.compliance_matrix_3() # compliance matrix
        eps12m = np.dot(S, sigma12m)  # calculate strains
        eps12r = np.dot(S, sigma12r)
        e11m = eps12m[0]    
        e11r = eps12r[0]    
        Xet = mat.F11t/mat.E11 # allowable strain
        G11 = 1/Xet**2
        rtent = solve_quadratic(eps12m, eps12r, G11, 0, 0, 0, 0, 0)
        r = rtent if (rtent*e11m + e11r) >= 0 else None  # make sure that resulting strain correct sign
        return (r, 'MaxStrain_T1_FF')

class MaxStrainCompression1(PlaneStressFailureCriterion):
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        S = mat.compliance_matrix_3()
        eps12m = np.dot(S, sigma12m)
        eps12r = np.dot(S, sigma12r)
        e11m = eps12m[0]
        e11r = eps12r[0]
        Xec = mat.F11c/mat.E11
        G11 = 1/Xec**2
        rtent = solve_quadratic(eps12m, eps12r, G11, 0, 0, 0, 0, 0)  
        r = rtent if (rtent*e11m + e11r) < 0 else None  
        return (r, 'MaxStrain_C1_FF')


class MaxStrainTension2(PlaneStressFailureCriterion):
    failure_type = 'MF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        S = mat.compliance_matrix_3() # compliance matrix
        eps12m = np.dot(S, sigma12m)  # calculate strains
        eps12r = np.dot(S, sigma12r)
        e22m = eps12m[1]    
        e22r = eps12r[1]    
        Yet = mat.F22t/mat.E22 # allowable strain
        G22 = 1/Yet**2
        rtent = solve_quadratic(eps12m, eps12r, 0, G22, 0, 0, 0, 0)  
        r = rtent if (rtent*e22m + e22r) >= 0 else None  # make sure that resulting strain correct sign
        return (r, 'MaxStrain_T2_MF')


class MaxStrainCompression2(PlaneStressFailureCriterion):
    failure_type = 'MF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        S = mat.compliance_matrix_3() # compliance matrix
        eps12m = np.dot(S, sigma12m)  # calculate strains
        eps12r = np.dot(S, sigma12r)
        e22m = eps12m[1]    
        e22r = eps12r[1]    
        Yec = mat.F22c/mat.E22 # allowable strain
        G22 = 1/Yec**2
        rtent = solve_quadratic(eps12m, eps12r, 0, G22, 0, 0, 0, 0)  
        r = rtent if (rtent*e22m + e22r) < 0 else None  # make sure that resulting strain correct sign
        return (r, 'MaxStrain_C2_MF')


class MaxStrainShear(PlaneStressFailureCriterion):
    failure_type = 'MF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)): 
        # can use stress here - no interaction with other components
        F66 = 1/mat.F12s**2
        r = solve_quadratic(sigma12m, sigma12r, 0, 0, 0, F66, 0, 0)  
        return (r, 'MaxStrain_S_MF')

###############################################################################  


class Hoffmann(PlaneStressFailureCriterion):
    """
    Hoffmann Criterion
    ------------------
    
    The Hoffmann criterion is an interactive criterion. It accounts for the 
    different material strength behaviour under tension or compression.
    It does not allow to identify failure modes.
    
    For plane stress, Hoffmann's criterion is given as:
    
    .. math::
    
        \frac{\sigma_1^2}{X_t X_c} + \frac{\sigma_2^2}{Y_t Y_c} - \frac{\sigma_1 \sigma_2}{X_t X_c} + \frac{X_c - X_t}{X_t X_c} \sigma_1 + \frac{Y_c - Y_t}{Y_t Y_c} \sigma_2 + \frac{\sigma_6^2}{S_{LT}^2} = 1 
    
    Ref. Berthelot, eq. 12.45
    
    Hoffmann's criterion is implemented as a single class. Only one failure mode is
    retured: `Hoffmann_I`.
    """
    
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        """Return Strength/Stress ratio for Hoffmann's criterion."""
        
        F11 = 1/(mat.F11t*mat.F11c)
        F12 = -1/(mat.F11t*mat.F11c)
        F22 = 1/(mat.F22t*mat.F22c)
        F66 = 1/mat.F12s**2
        F1 = (mat.F11c - mat.F11t)/(mat.F11t*mat.F11c)
        F2 = (mat.F22c - mat.F22t)/(mat.F22t*mat.F22c)
        if np.allclose(sigma12m, 0):
            rf = float('inf')
        else:
            rf = solve_quadratic(sigma12m, sigma12r, F11, F22, F12, F66, F1, F2)
        return (rf, 'Hoffmann_I')



###############################################################################  


class TsaiHill(PlaneStressFailureCriterion):
    """
    Tsai-Hill Criterion
    ------------------
    
    The Tsai-Hill criterion is an interactive criterion. It accounts for the 
    different material strength behaviour under tension or compression.
    It does not allow to identify failure modes.
    
    For plane stress, the Tsai-Hill criterion is given as:
    
    .. math::
    
        \left (\frac{\sigma_1}{X} \right)^2 + \left (\frac{\sigma_2}{Y} \right)^2 - \frac{\sigma_1 \sigma_2}{X^2} + \left (\frac{\sigma_6}{S_{LT}} \right)^2 = 1 
    
    Ref. Berthelot, eq. 12.41 and ESDU 83014
    
    The strength properties X and Y are replaced by X_t, X_c and Y_t, Y_c, depending on
    the sign of \sigma_1 and \sigma_2. This is implemented as an a-posteriori check:
    r ratios are calculated for all combinations of Xt/Xc, Yt/Yc. Then the resulting 
    stresses at failure \sigma = \sigma^R + r \sigma^M are calculated. Then the 
    result for the correct combination is returned.  
    
    The Tsai-Hill criterion is implemented as a single class. Failure mode codes are:
    
    - tsaihill_I_XT_YT: resulting stress \sigma_1 at failure is tensile, X_t was 
      used for X. resulting stress \sigma_2 at failure is tensile, Y_t was 
      used for Y. 
    - tsaihill_I_XT_YC: resulting stress \sigma_1 at failure is >= 0, X_t was 
      used for X. resulting stress \sigma_2 at failure is < 0, Y_c was 
      used for Y. 
    - tsaihill_I_XC_YT: resulting stress \sigma_1 at failure is < 0, X_c was 
      used for X. resulting stress \sigma_2 at failure is tensile, Y_t was 
      used for Y. 
    - tsaihill_I_XC_YC: resulting stress \sigma_1 at failure is < 0, X_c was 
      used for X. resulting stress \sigma_2 at failure is < 0, Y_c was 
      used for Y. 
    
    """ 
    
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        """Return strength ratio and failure mode for the Tsai-Hill criterion."""
        s11m, s22m = sigma12m[[0, 1]]
        s11r, s22r = sigma12r[[0, 1]]
        
        F11_T = 1/mat.F11t**2 # depends on sign of sigma_1
        F11_C = 1/mat.F11c**2
        F12_T = -1/mat.F11t**2 # depends on sign of sigma_1
        F12_C = -1/mat.F11c**2 
        F22_T = 1/mat.F22t**2 # depends on sign of sigma_2
        F22_C = 1/mat.F22c**2
        F66 = 1/mat.F12s**2
        
        # case 1: XT / YT
        rtt = solve_quadratic(sigma12m, sigma12r, F11_T, F22_T, F12_T, F66, 0, 0)
        if (rtt*s11m + s11r) >= 0 and (rtt*s22m + s22r) >= 0:
            return (rtt, 'tsaihill_I_XT_YT')
        
        # case 2: XT / YC
        rtc = solve_quadratic(sigma12m, sigma12r, F11_T, F22_C, F12_T, F66, 0, 0)
        if (rtc*s11m + s11r) >= 0 and (rtc*s22m + s22r) < 0:
            return (rtc, 'tsaihill_I_XT_YC')
        
        # case 3: XC / YT
        rct = solve_quadratic(sigma12m, sigma12r, F11_C, F22_T, F12_C, F66, 0, 0)
        if (rct*s11m + s11r) < 0 and (rct*s22m + s22r) >= 0:
            return (rct, 'tsaihill_I_XC_YT')
        
        # case 4: XC / YC
        rcc = solve_quadratic(sigma12m, sigma12r, F11_C, F22_C, F12_C, F66, 0, 0)
        if (rcc*s11m + s11r) < 0 and (rcc*s22m + s22r) < 0:
            return (rcc, 'tsaihill_I_XC_YC')
    

###############################################################################  

"""
The Tasi-Wu failure criterion is an interactive criterion, that does not 
distinguish between failure modes. Fir a plane stress state, failure is 
described by:

.. math::

    F_1 \sigma_1 + F_2 \sigma_2 + F_6 \sigma_6 + F_{11} \sigma_1^2 + F_{22} \sigma_2^2 + F_{66}  \sigma_6^2 + 2 F_{12} \sigma_1 \sigma_2 = 1
    
The parameters are:

.. math::

    F_1 = \frac{1}{X_t} - \frac{1}{X_c}
    
.. math::

    F_2 = \frac{1}{Y_t} - \frac{1}{Y_c}
    
.. math::

    F_6 = 0
    
.. math::

    F_{11} = \frac{1}{X_t X_c}
     
.. math::

    F_{22} = \frac{1}{Y_t Y_c}
     
.. math::

    F_{66} = \frac{1}{S_l^2}
     
.. math::

    F_{12} = F_{xy}^{*} \sqrt{F_{11} F_{22}}
     
- F_6 is zero because of symmetry reasons. 
- Fxystar has to be betermined from biaxial tensile tests. 
- the factor Fxystar in F_12 can vary between -0.5 and 0.0
- offer two variants of the criterion: TsaiWu/0.0 and TsaiWu/-0.5
- Berthelot 12.48
"""

class TsaiWu(PlaneStressFailureCriterion):
    """The Tsai-Wu interactive criterion."""

    def __init__(self, fxystar):
        """Return a new instance of the Tsai-Wu failure criterion.
        
        :Parameters:
        - fxystar: float, -0.5 <= fxystar <= 0.0, constant used in F12
        """
        assert -0.5 <= fxystar <= 0.0
        self.fxystar = fxystar

    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        """Return strength ratio and failure mode for the Tsai-Hill criterion."""
    
        F1 = 1/mat.F11t - 1/mat.F11c
        F2 = 1/mat.F22t - 1/mat.F22c
        F11 = 1/(mat.F11t*mat.F11c)
        F22 = 1/(mat.F22t*mat.F22c)
        F66 = 1/mat.F12s**2
        F12 = self.fxystar * math.sqrt(F11*F22)
        
        r = solve_quadratic(sigma12m, sigma12r, F11, F22, F12, F66, F1, F2)
        return (r, 'TsaiWu{}_I'.format(self.fxystar))


###############################################################################

# Airbus modified Puck criterion for matrix failure. 2d criterion.
# RP0416030 iss. 2.1, Eq. 14
# 
# title = "Ply Failure Criterion - Airbus modified Puck 2D Matrix Failure"
# code = "MF_Mod_Puck_Airbus"
# fmode = MATRIX_FAILURE


class YamadaSunTension(PlaneStressFailureCriterion):
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11 = sigma12m[0]
        F11 = 1/mat.F11t**2
        F66 = 1/mat.F12s**2
        prelim_r = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, F66, 0, 0)
        r = prelim_r if prelim_r*s11 >= 0 else None 
        return (r, 'YamadaSun_FF_T')

class YamadaSunCompression(PlaneStressFailureCriterion):
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11 = sigma12m[0]
        F11 = 1/mat.F11c**2
        F66 = 1/mat.F12s**2
        prelim_r = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, F66, 0, 0)
        r = prelim_r if prelim_r*s11 < 0 else None 
        return (r, 'YamadaSun_FF_C')

class ModPuckAirbus(PlaneStressFailureCriterion):
    failure_type = 'MF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # this is taken from the visual basic source...
        F11 = 1/(2*mat.F11t)**2 # yes, they include this term in there... 
        F22 = 1/(mat.F22t*mat.F22c)
        F66 = 1/mat.F12s**2
        F2 = 1/mat.F22t - 1/mat.F22c
        r = solve_quadratic(sigma12m, sigma12r, F11, F22, 0, F66, 0, F2)
        return (r, 'AirbusModPuck_MF_I')


###############################################################################

# Hashin 2D, according to MIL-HDBK-17-3F, eq. 5.2.4(f) to (i)
# 
# 1. Fibre failure, tensile:
#     (f*sigma_11/F11t)**2 + (f*sigma_12/F12s)**2 = 1
#     -> f**2 (r_11**2 + r_12**2) = 1
#     -> f = 1 / sqrt(r_11**2 + r_22**2)
#     
# Transverse shear strength F23s, or ST: F23s = 0.378 F22c.
# This corresponds to a typical fracture angle of 53°.
# Ref. Carlos G. Dávila, Pedro P. Camanho: Failure Criteria for FRP Laminates
# in Plane Stress. NASA/TM-2003-212663

"""
Hashin Criterion

"""

class HashinFibreTension(PlaneStressFailureCriterion):
    """Fibre tensile strength. Hashin."""
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11m = sigma12m[0]
        s11r = sigma12r[0]
        F11 = 1/mat.F11t**2
        F66 = 1/mat.F12s**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, F66, 0, 0)
        r = rtent if (rtent*s11m + s11r) >= 0 else None 
        return (r, 'Hashin2D_FF_T')
  
  
class HashinFibreCompressionA(PlaneStressFailureCriterion):
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11m = sigma12m[0]
        s11r = sigma12r[0]
        F11 = 1/mat.F11c**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, 0, 0, 0)
        r = rtent if (rtent*s11m + s11r) < 0 else None 
        return (r, 'Hashin2D_FF_C')
    
    
class HashinFibreCompressionB(PlaneStressFailureCriterion):
    failure_type = 'FF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s11m = sigma12m[0]
        s11r = sigma12r[0]
        F11 = 1/mat.F11c**2
        F66 = 1/mat.F12s**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, F66, 0, 0)
        r = rtent if (rtent*s11m + s11r) < 0 else None 
        return (r, 'Hashin2DMod_FF_C')


class HashinMatrixTension(PlaneStressFailureCriterion):
    failure_type = 'MF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # handle cases of positive or zero tensile stress
        s22m = sigma12m[1]
        s22r = sigma12r[1]
        F22 = 1/mat.F22t**2
        F66 = 1/mat.F12s**2
        rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, F66, 0, 0)
        r = rtent if (rtent*s22m + s22r) >= 0 else None 
        return (r, 'Hashin2D_MF_T')


class HashinMatrixCompression(PlaneStressFailureCriterion):
    failure_type = 'MF'
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        s22m = sigma12m[1]
        s22r = sigma12r[1]
        F23s = 0.378 * mat.F22c # corresponds to fracture angle of 53°
        k = mat.F22c/(2*F23s)
        
        F22 = 1/(2*F23s)**2
        F66 = 1/mat.F12s**2
        F2 = (k**2 - 1)/mat.F22c
        rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, F66, 0, F2)
        r = rtent if (rtent*s22m + s22r) < 0 else None     
        return (r, 'Hashin2D_MF_C')
    
    
###############################################################################  

class PuckFibreTension(PlaneStressFailureCriterion):
    failure_type = "FF"
    
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # ESDU 83014
        s11m = sigma12m[0]
        s11r = sigma12r[0]
        F11 = 1/mat.F11t**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, 0, 0, 0)
        r = rtent if (rtent*s11m + s11r) >= 0 else None
        return (r, 'Puck_FF_T')

class PuckFibreCompression(PlaneStressFailureCriterion):
    failure_type = "FF"
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # ESDU 83014
        s11m = sigma12m[0]
        s11r = sigma12r[0]
        F11 = 1/mat.F11c**2
        rtent = solve_quadratic(sigma12m, sigma12r, F11, 0, 0, 0, 0, 0)
        r = rtent if (rtent*s11m + s11r) < 0 else None     
        return (r, 'Puck_FF_C')

class PuckMatrixTension(PlaneStressFailureCriterion):
    failure_type = "MF"
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # ESDU 83014
        s22m = sigma12m[1]
        s22r = sigma12r[1]
        F22 = 1/mat.F22t**2
        F66 = 1/mat.F12s**2
        rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, F66, 0, 0)
        r = rtent if (rtent*s22m + s22r) >= 0 else None     
        return (r, 'Puck_MF_T')

class PuckMatrixCompression(PlaneStressFailureCriterion):
    failure_type = "MF"
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # ESDU 83014
        s22m = sigma12m[1]
        s22r = sigma12r[1]
        F22 = 1/mat.F22c**2
        F66 = 1/mat.F12s**2
        rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, F66, 0, 0)
        r = rtent if (rtent*s22m + s22r) < 0 else None     
        return (r, 'Puck_MF_C')


###############################################################################  

class ModPuckMatrixFailure(PlaneStressFailureCriterion):
    failure_type = "MF"
    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # ESDU 83014
        F22 = 1/(mat.F22t*mat.F22c)
        F66 = 1/mat.F12s**2
        F2 = 1/mat.F22t - 1/mat.F22c
        r = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, F66, 0, F2)
        fmode = 'ModPuck_MF' #if (r*s22m + s22r) >= 0 else 'ModPuck_MF_C'
        return (r, fmode)


###############################################################################

# LaRC03: residual stresses not supported for the time being ???
# Would require iteration, because some parameters (PhiC, ..) depend on stress
# state. 

def lambda022(mat):
    # parameter required for calculation of in-situ strength of thin embedded 
    # ply in LaRC03 model.
    return 2*(1/mat.E22 - mat.nu21**2/mat.E11)

def lambda044(mat):
    # parameter required for calculation of in-situ strength of thin embedded 
    # ply in LaRC03 model.
    return 1/mat.G12

def YTis(mat, is_thin_embedded_ply):
    """Return in-situ transverse tension strength of thin embedded ply, 
    LaRC03 failure criterion.
    Requires a material that has thickness property, as well as GIc value!
    """
    if is_thin_embedded_ply:
        return math.sqrt(8*mat.G1CL / (math.pi*mat.t*lambda022(mat)))
    else: # thick ply, eq. 26
        return 1.12*math.sqrt(2)*mat.F22t

def SLis(mat, is_thin_embedded_ply):
    """Return in-situ longitudinal shear strength of thin embedded ply, 
    LaRC03 failure criterion.
    Requires a material that has thickness property, as well as GIIc value!
    """
    if is_thin_embedded_ply:
        return math.sqrt(8*mat.G2CL / (math.pi*mat.t*lambda044(mat)))
    else: # eq. 26
        return math.sqrt(2) * mat.F12s

def toughness_ratio(mat, is_thin_embedded_ply):
    """Return toughness ratio g for the ply. 
    """
    if is_thin_embedded_ply is True:
        return mat.G1CL
    elif is_thin_embedded_ply is False:
        return 1.12**2 * (lambda022(mat)/lambda044(mat)) * (mat.F22t/mat.F12s)**2
    else:
        raise ValueError('must provide True or False for is_thin_embedded_py parameter')
    
def fibre_misaligment_stresses(sig):
    pass

def phi():
    """Total mialignment angle. Required for LaRC03 failure criterion."""
    pass

def phiC(mat):
    pass

def etaL(mat):
    pass

def etaT(mat):
    pass

def r_larc03_ff_t(mat, sigma12m, sigma12r=np.zeros(3)):
    # LaRC03 #3: fibre failure, tension
    S = mat.compliance_matrix_3() # compliance matrix
    eps12m = np.dot(S, sigma12m)  # calculate strains
    eps12r = np.dot(S, sigma12r)
    s11m = sigma12m[0]    
    s11r = sigma12r[0]    
    Xet = mat.F11t/mat.E11 # allowable strain
    G11 = 1/Xet**2
    rtent = solve_quadratic(eps12m, eps12r, G11, 0, 0, 0, 0, 0)  
    r = rtent if (rtent*s11m + s11r) >= 0 else None  # make sure that resulting stress has correct sign
    return (r, 'LaRC03#3_FF_T')

def r_larc03_ff_c_mc(mat, sigma12m, sigma12r=np.zeros(3)):
    # LaRC03 #4, Fibre compression, in case of matrix compression
    pass
    
def r_larc03_ff_c_mt(mat, sigma12m, sigma12r=np.zeros(3), thin_embedded=None):
    # LaRC03 #5, Fibre compression, in case of matrix tension
    # FIXME: not the typical stresses, but "sigma_22^m" -> stresses in fibre 
    # misalignment frame
    s1m, s2m = sigma12m[[0, 1]]
    s1r, s2r = sigma12r[[0, 1]]
    g = toughness_ratio(mat, thin_embedded)
    F22 = g/YTis(mat)**2
    F66 = 1/SLis(mat)**2
    F2 = (1-g)/YTis
    rtent = solve_quadratic(sigma12m, sigma12r, 0, F22, 0, F66, 0, F2) 
    r = rtent if (rtent*s2m + s2r >= 0) and (rtent*s1m + s1r < 0) else None  # make sure that resulting stress has correct sign
    return (r, 'LaRC03#5_FF_C_(MT)')    


class LaRC03():
    """Ref. Carlos G. Dávila, Pedro P. Camanho: Failure Criteria for FRP Laminates
    in Plane Stress. NASA/TM-2003-212663
    Thick ply: >= 0.7 mmm, for E-glass/epoxy and carbon/epoxy
    """
    pass


class Cuntze():
    pass

####################################################################

"""
Puck Action Plane Criteria (for plane stress state): "PuckC"
The implementation follows the description in Schürmann

Notation:

- R_{\perp}^+ ... F_{22t}, Y_t ... transverse tensile strength -> \sigma_n
- R_{\perp}^- ... F_{22c}, Y_c ... transverse compressive strength -> \sigma_n
- R_{\perp\parallel} ... F_{12s}, S_l ... in-plane shear strength -> \tau_{n1}
- R_{\perp\perp}^A ... shear strength in action plane, corresponds to \tau_{nt}

Slope parameters (Table 17.1):

======= ====== ====== ============= ============= 
        p_sp_+ p_sp_- p_ss_+        p_ss_- 
======= ====== ====== ============= =============
GFRP-UD 0.30   0.25   0.20 ... 0.25 0.20 ... 0.25
CFRP-UD 0.35   0.30   0.25 ... 0.30 0.25 ... 0.30
======= ====== ====== ============= =============

Transverse shear strength (eq. 16.17):

.. math::
 
    R_{\perp\perp}^A = \frac{R_{\perp}^-}{2 (1 + p_{\perp\perp}^- }
    
If numerical search for fracture angle in plane stress state is to be avaoided, 
the transver shear strength follows from (17.50):

.. math::

     R_{\perp\perp}^A = \frac{R_{\perp\parallel}}{2 p_{\perp\parallel}^-} \left( \sqrt{ 1 + 2p_{\perp\parallel}^{-} \frac{R_{\perp}^{-}}{R_{\perp\parallel}}} - 1 \right)

General formulation:

Fracture criterion for tension, sigma_n >= 0, eq. 17.29

.. math::

    \left( \frac{\tau_{n\psi}{R_{\perp\psi}^A} \right)^2 + 2 \frac{p^+_{\perp\psi} \sigma_n}{R_{\perp\psi}^A} + \left( 1 - 2 \frac{p^+_{\perp\psi} R^+_{\perp}}{R_{\perp\psi}^A} \right) \cot \frac{\sigma_n^2}{(R_{\perp}^+)^2} = 1
    
Fracture criterion for compression, sigma_n < 0, eq. 17.32

.. math::

   \left( \frac{\tau_{n\psi}{R_{\perp\psi}^A} \right)^2 + 2 \frac{p^-_{\perp\psi}}{R^A_{\perp\psi}} \sigma_n = 1

Strength in fracture plane (eq. 17.34):

.. math::

    \left( \frac{1}{R_{\perp\psi}^A} \right)^2 = \left( \frac{\cos \psi}{R_{\perp\perp}^A \right)^2 + \left( \frac{\sin \psi}{R_{\perp\parallel} \right)^2 

Angle psi is defined by the stress state in the fracture plane (eq.17.35): 

.. math::

    \cos \psi = \frac{\tau_{nt}}{\tau_{n\psi}} 
    
.. math::

    \sin \psi = \frac{\tau_{n1}}{\tau_{n\psi}} 

Formulation for plane stress state:

Coupling of slope parameters. this eliminates the need for searching for the 
fracture angle in Mode C. 
For plane stress, \theta_{fp} can be determined analytically.

sigma_n -> sigma_2
tau_n1 -> tau_21
tau_nt = 0
tau_npsi = sqrt(tau_n1**2 + tau_nt**2) = tau_n1 = tau_21


.. math::

    \frac{p^-_{\perp\perp}}{R_{\perp\perp}^A} = \frac{p^-_{\perp\parallel}}{R_{\perp\parallel}^A}

In modes A and B the angle of the fracture plane theta_fp is zero. That results in:
sigma_n -> sigma_2,
tau_n1 -> tau_21, 
tau_nt = 0

In mode C, the angle of the fracture plane can be determined analytically (17.45):

.. math::

    \cos \theta_{fp} = \sqrt{\frac{R_{\perp\perp}^A}{-\sigma_2^*}}

- \sigma_2^* is the compressive stress at fracture in mode C. 

Schürmann gives the solutions for fE "Anstrengung". 
From fE = 1/fS = ( L + sqrt(L**2 + 4Q)) / 2 we can determine the linear and
quadratic term L, Q of the fracture condition curve.

Mode A:
L/2 = pspt*s2/Rsp
(L/2)**2 + Q = (1 - pspt*Rst/Rsp)**2 * (s2/Rst)**2 + (s6/Rsp)**2 

 
"""

    
class PuckActionPlaneMF(PlaneStressFailureCriterion):
    failure_type = 'MF'
    # carbon
    #pspc = 0.30
    #pspt = 0.35
    #psst = 0.25 # ... 0.30
    #_pssc = 0.25 # ... 0.30

    def __init__(self, pspc=0.30, pspt=0.35, etaw1=1.6, detaw1=1.0):
        self._pspc = pspc
        self._pspt = pspt
        self._etaw10 = etaw1
        self._detaw1 = detaw1

    def _pssc(self, mat):
        # vdi2014, eq. A13 -> dependent value due to parameter coupling
        return self._rssa(mat) * self._pspc/mat.F12s

    def _rssa(self, mat):
        """Return transverse shear strength of action plane.
        
        This is calculated for plane stress state, with coupling of slope 
        parameters. 
        """
        # Schürmann, 17.50
        # Strength of fracture plane - valid for plane stress state with 
        # assumed parameter coupling
        Sl = mat.F12s
        Yc = mat.F22c 
        return 0.5*(Sl/self._pspc) * (math.sqrt(1 + 2*self._pspc*Yc/Sl) - 1)
    
    def _tau21c(self, mat):
        return mat.F12s * math.sqrt(1 + 2*self._pssc(mat))

    @staticmethod
    def _fracture_angle(Rssa, F12s, pssc, s2, s6):
        # degrees
        if s2 == 0:
            return float('nan')
        x = (Rssa/F12s)**2 * (s6/s2)**2 + 1
        x /= 2*(1 + pssc)
        cos_theta_fp = math.sqrt(x)
        theta_fp = math.acos(cos_theta_fp)
        return math.degrees(theta_fp)

    def _r_mode_a(self, mat, sigma12m, sigma12r, X, etaw10, detaw1):
        # failure related to fracture plane
        # Returns Anstrengung fe (== failure index k)
        # Elliptical
        # Theta_fp = 0
        # sigma_2 >= 0
        #
        # VDI2014, Table A1
        # All strength terms are positive values!!!
        # 
        s1m, s2m, s6m = sigma12m
        s1r, s2r, s6r = sigma12r
        Sl = mat.F12s
        Yt = mat.F22t
    
        k1 = 1 - 2*self._pspt*Yt/Sl
        k2 = etaw10 - detaw1*s1r/X
        k3 = 2*self._pspt*detaw1
        k4 = 2*self._pspt
        
        qa = (s6m/Sl)**2
        qa -= k4 * s1m*s2m/(Sl*X)
        qa += k1 * (s2m/Yt)**2
        qa -= detaw1**2 * (s1m/X)**2
        
        la = k3 * s1m*s2r/(Sl*X)
        la += k2*k4 * s2m/Sl
        la += 2*k1 * s2r*s2m/Sl**2
        la += 2*k2*detaw1 * s1m/X
        la += 2 * s6m*s6r/Sl**2
        
        ca = k1 * (s2r/Yt)**2
        ca -=  k2**2 
        ca += (s6r/Sl)**2
        ca += k2 * 2*self._pspt*s2r/Sl
        
        try:
            r = (math.sqrt(la**2 - 4*qa*ca) - la)/(2*qa) 
            if r < 0: r = 0
        except ValueError: # math domain error -> comples result
            r = float('nan')
        cx = 'XT' if X > 0 else 'XC'
        return (r, 'PuckFP_MF_ModeA_{}'.format(cx))
    
    def _r_mode_b(self, mat, sigma12m, sigma12r, X, etaw10, detaw1):
        # Parabolical
        # Theta_fp = 0
        # sigma_2 < 0 and 0 <= abs(sigma_2/tau_21) <= abs(Rssa/tau_21c)
        #assert fibre_load in ('tension, compression')
        
        s1m, s2m, s6m = sigma12m
        s1r, s2r, s6r = sigma12r
        Sl = mat.F12s

        k2 = etaw10 - detaw1*s1r/X
        k3 = 2*self._pspc*detaw1
        k4 = 2*self._pspc   
        
        qb = (s6m/Sl)**2
        qb -= k3*s1m*s2m/(Sl*X)
        qb -= (detaw1*s2m/X)**2
        
        lb = 2*s6r*s6m/Sl**2
        lb -= k4*s1m*s2r/(Sl*X)
        lb += k4*s2m*k2/Sl
        lb += 2*detaw1*s1m*k2/X
        
        cb = (s6r/Sl)**2 
        cb -= k2**2
        cb += k4*s2r*k2/Sl 
    
        try:
            r = (math.sqrt(lb**2 - 4*qb*cb) - lb)/(2*qb) 
            if r < 0: r = 0
        except ValueError: # math domain error -> complex result
            r = float('nan')

        cx = 'XT' if X > 0 else 'XC'
        return (r, 'PuckFP_MF_ModeB_{}'.format(cx))    
    

    def _r_mode_c(self, mat, sigma12m, sigma12r, X, etaw10, detaw1):
        # fracture plane angle theta_fp != 0
        # sigma_2 < 0 and 0 <= abs(tau_21/sigma_2) <= abs(tau_21c/Rssa)

        s1m, s2m, s6m = sigma12m
        s1r, s2r, s6r = sigma12r
        Sl = mat.F12s
        #Yt = mat.F22t
        Yc = mat.F22c
        Rssa = self._rssa(mat)
        pssc = self._pssc(mat)

        Sa = Sl + self._pspc*Rssa
        
        qc = 0.25 * s6m**2/Sa**2
        qc += (s2m/Yc)**2
        qc -= detaw1*s1m*s2m/(Yc*X)
        
        lc = 0.5*s6m*s6r/Sa**2
        lc -= detaw1*s1r*s2m/(Yc*X)
        lc -= detaw1*s1m*s2r/(Yc*X)
        lc += etaw10*s2m/Yc
        lc += 2*s2m*s2r/Yc**2
        
        cc = 0.25 * (s6r/Sa)**2
        cc += etaw10*s2r/Yc
        cc += (s2r/Yc)**2
        cc -= detaw1*s2r*s1r/(Yc*X)
        
        try:
            r = (math.sqrt(lc**2 - 4*qc*cc) - lc)/(2*qc) 
            theta_fp = PuckActionPlaneMF._fracture_angle(Rssa, Sl, pssc, s2r+r*s2m, s6r+r*s6m)
            if r < 0: r = 0
        except ValueError: # math domain error -> complex result
            r = float('nan')        
            theta_fp = float('nan')
            
        cx = 'XT' if X > 0 else 'XC'
        return (r, 'PuckFP_MF_ModeC_{}_FP{:.1f}°'.format(cx, theta_fp)) 
    

    def _get_fmode(self, mat, s2, s6):
        Rssa = self._rssa(mat)
        Tau21c = self._tau21c(mat)

        if s2 >= 0:
            return 'A'
        elif 0 <= abs(s6/s2) <= abs(Tau21c/Rssa):
            return 'C'
        else:
            return 'B'

    def r(self, mat, sigma12m, sigma12r=np.zeros(3)):
        # VDI 2014, Fig. A6
        
        from itertools import product
        
        s1m, s2m, s6m = sigma12m
        s1r, s2r, s6r = sigma12r
        
        # 3 modes: A, B, C
        # 2 signs: tension/compression
        # 2 s1 influences: (1.6, 1), (1, 0)
        # => 12 calculations
        
        weak_vals = {True: (self._etaw10, self._detaw1), 
                     False: (1.0, 0)}
        
        strength_vals = {'tension': mat.F11t, 
                         'compression': -mat.F11c} 
        
        modes = {'A': self._r_mode_a, 
                 'B': self._r_mode_b,
                 'C': self._r_mode_c}
        
        results = []
        for (kmode, kX, kweak) in product(modes, strength_vals, weak_vals):
            etaw0, detaw = weak_vals[kweak]
            X = strength_vals[kX]
            func = modes[kmode]
            r, fmode = func(mat, sigma12m, sigma12r, X, etaw0, detaw)
            if not math.isfinite(r):
                continue
            # resulting stresses
            s1c = s1r + r*s1m
            s2c = s2r + r*s2m
            s6c = s6r + r*s6m
            
            # validity checks
            sign_valid = (s1c >= 0 and X > 0) or (s1c < 0 and X < 0)
            weakening_valid = not (((s1r/X >= 0.6) and not kweak) or ((s1r/X < 0.6) and kweak)) 
            mode_valid = kmode == self._get_fmode(mat, s2c, s6c)
            if sign_valid and weakening_valid and mode_valid:
                results.append((r, fmode))

        if len(results) == 1:
            return results[0]
        else:
            # FIXME: what can be the reason for this? Can Fibre Failure lead to this situation? 
            # other possibility: residual stresses alone lead to matrix failure?
            return (None, 'PuckFP_MF')


if __name__ == '__main__':
    pass