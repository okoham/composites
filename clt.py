"""
Created on 18.09.2014
@author: Oliver

This module provides classes and functions for classical laminate analysis
of laminates in plane stress state.
"""

from __future__ import division, print_function
import numpy as np
import math
from material import TransverseIsotropicPlyMaterial


def validate_stacking(sseq):
    """Return the stacking sequence sseq (if validated), or raise ValueError.
    
    A stacking sequence (in the context of this module) is a list of (angle, 
    material) tuples. 
    - angle: in degrees. int or float
    - material: a TransverseIsotropicPlyMaterial instance.
    """
    
    if not len(sseq) >= 1: 
        raise ValueError('need at least 1 layer')
    if not isinstance(sseq, list):
        raise ValueError('sseq not a list')
    if not all(isinstance(item, tuple) for item in sseq):
        raise ValueError('items (layer defs) must be tuples')
    if not all(len(item) == 2 for item in sseq):
        raise ValueError('tuples must be of length 2')
    if not all(isinstance(angle, (int, float)) for (angle, dummy) in sseq):
        raise ValueError('angle must be numeric')
    if not all(isinstance(mat, TransverseIsotropicPlyMaterial) for (dummy, mat) in sseq):
        raise ValueError('mat must be TransverseIsotropicPlyMaterial')
    return sseq


def stacking_to_layup2(stseq):
    """Return a layup definition, from stacking sequence (no materials)"""
    counts = {}
    for ang in stseq:
        if ang in counts:
            counts[ang] += 1
        else:
            counts[ang] = 1
    layup = sorted(counts.items(), key=lambda x: x[0])
    return layup


def stacking_to_layup(stseq):
    """Return a layup definition for stacking sequence. Works only if all
    plies are of same material.
    Raise ValueError if more than one material used in stseq
    """

    material_set = set([mat for (ang, mat) in stseq])
    if len(material_set) > 1:
        raise ValueError('Works only if all plies have same material')

    # get set of unique angles
    # list of tuples: (angle, count) <- angles unique now.
    counts = {}
    for (ang, mat) in stseq:
        if ang in counts:
            counts[ang] += 1
        else:
            counts[ang] = 1
    layup = sorted(counts.items(), key=lambda x: x[0])
    return layup, material_set.pop()


def mirror_symmetric(stseq):
    """Return a symmetric stacking sequence: stseq + reversed(stseq)
    """
    s = list(stseq)
    return s + reversed(s)

def tmat_sig(theta):
    """Return stress transformation matrix for plane stress state, as a (3,3) numpy array.
    Transformation xy -> 12
    
    :Parameters:
      - `theta`: angle is the angle (in radians) between the arbitrary (x,y) system and the 
      (1,2) principal material system of the layer. 
      
    
    .. math::
        \mathbf{T}_s =  \left[ \begin{array}{ccc}
        m^2 & n^2   & -2 m n \\
        n^2 & m^2   & 2 m n \\
        m n & - m n & m^2 - n^2
        \end{array} \right]
        
    where :math:` m = \cos (\theta)` and :math:` n = \sin (\theta)` 
    :math:` \mathbf{\sigma_x} = \mathbf{T}_s \mathbf{\sigma_l}
    
    See Mil-Hdbk-17-F, eq. 5.3.1(k).
    """
    m = math.cos(theta)
    n = math.sin(theta)
    return np.array([[m**2, n**2, -2*m*n],
                     [n**2, m**2, 2*m*n],
                     [m*n, -m*n, m**2 - n**2]])

def tmat_eps(theta):
    """Return strain transformation matrix for plane stress state, as a (3,3) numpy array.
    
    :Parameters:
      - `theta`: float, is the angle (in radians) between the arbitrary (x,y) system and the 
      (1,2) principal material system of the layer. 
      
    .. math::
    
        \mathbf{T}_e =  \left[ \begin{array}{ccc}
        m^2   & n^2    & - m n \\
        n^2   & m^2    & m n \\
        2 m n & -2 m n & m^2 - n^2
        \end{array} \right]
        
    where :math:` m = \cos (\theta)` and :math:` n = \sin (\theta)` 
    :math:` \mathbf{\varepsilon}_x = \mathbf{T}_e \mathbf{\varepsilon}_l`
    
    
    Theta is defined as the rotation from the arbitrary (x,y) system to the 
    (1,2) material system.
    Plane stress state! Only components 1, 2, 6
    See [mil173f]_, eq. ...
    """
    m = math.cos(theta)
    n = math.sin(theta)
    return np.array([[m**2, n**2, -m*n],
                      [n**2, m**2, m*n],
                      [2*m*n, -2*m*n, m**2-n**2]])


def rotated_stiffness_matrix33(m, alpha):
    """Return rotated (3,3) stiffness matrix M' = Ts . M . Ts^T
    M can be Q, or A, or D
    
    alpha is rotation angle about 3 (or z) axis, in radians.
    """
    ts = tmat_sig(alpha)
    return np.dot(ts, np.dot(m, ts.T))


class Layer(object):
    """Class providing methods for an orthotropic layer, pure plane stress 
    state. The Layer always lives in a Laminate.
    
    Off-axis properties of a layer: (info)
    
    .. math::
    
        E_x = \bar{Q}_{11} - \frac{\bar{Q}_{12}^2}{\bar{Q}_{22}}
        
    .. math::
    
        E_y = \bar{Q}_{22} - \frac{\bar{Q}_{12}^2}{\bar{Q}_{11}}
        
    .. math::
    
        \nu_{xy} = \frac{\bar{Q}_{21}}{\bar{Q}_{22}} = \frac{\bar{Q}_{12}}{\bar{Q}_{22}} 
        
    .. math::
        
        G_{xy} = \bar{Q}_{66}
    
    """
    
    __slots__ = '_laminate', '_material', '_theta'

    def __init__(self, laminate, theta, material):
        """Create and return a new layer.
        
        :Parameters:
        - laminate: a _Laminate instance
        - theta: orientation of layer with respect to the laminate reference
                 axis (in radians), float
        - material: a TransverseIsotropicPlyMaterial instance
        """
        if not isinstance(laminate, Laminate):
            raise TypeError('not a laminate instance')
        self._laminate = laminate
        if not isinstance(material, TransverseIsotropicPlyMaterial):
            raise TypeError('not a TransverseIsotropicPlyMaterial instance')
        self._material = material
        if not isinstance(theta, (int, float)):
            raise TypeError('not float/int')
        self._theta = float(theta) 

    def thickness(self):
        """Return the thickness of the layer."""
        return self._material.t

    def angle(self):
        """Return the orientation of the layer, in radians."""
        return self._theta
    
    def angle_deg(self):
        """Return the orientation of the layer, in degrees."""
        return math.degrees(self._theta)    

    def material(self):
        """Return the layer material."""
        return self._material

    def S(self):
        """Return the 3x3 layer compliance matrix S for plane stress state, 
        in (1,2) coordinate system.
        
        For a transverse isotropic material, the layer compliance matrix 
        looks like this:
        
        .. math::
        
            \mathbf{S} = \left[ \begin{array}{ccc}
            S_{11} & S_{12} & 0 \\
            S_{21} & S_{22} & 0 \\
            0      & 0      & S_{66}
            \end{array} right]        
        
        .. math::
        
            \mathbf{S} = \left[ \begin{array}{ccc}
            \frac{1}{E_{11}}          & - \frac{\nu_{21}}{E_{22}} & 0 \\
            - \frac{\nu_{21}}{E_{22}} & \frac{1}{E_{22}}          & 0 \\
            0                         & 0                         & \frac{1}{G_{12}} 
            \end{array} right]
            
        """
        E11 = self._material.E11
        E22 = self._material.E22
        nu21 = self._material.nu21
        G12 = self._material.G12
        return np.array([[1/E11, -nu21/E22, 0],
                          [-nu21/E22, 1/E22, 0],
                          [0, 0, 1/G12]], dtype=np.float32)

    def Q(self):
        """Return the 3x3 layer stiffness matrix Q for plane stress state, 
        in (1,2) coordinate system.
        
        For a transverse isotropic material, axis 1 being the axis of rotational symmmetry: 
        
        .. math::
        
            \mathbf{Q} = \left[ \begin{array}{ccc}
            Q_{11} & Q_{12} & 0 \\
            Q_{21} & Q_{22} & 0 \\
            0      & 0      & Q_{66}
            \end{array} right]          
        
        .. math::
        
            \mathbf{Q} = \left[ \begin{array}{ccc}
            \frac{E_{11}}{\Delta}          & \frac{\nu_{12} E_{22}}{\Delta} & 0 \\
            \frac{\nu_{12} E_{22}}{\Delta} & \frac{E_{22}}{\Delta}          & 0 \\
            0                              & 0                              & G_{12} 
            \end{array} right]
            
        where :math:`\Delta = 1 - \nu_{12} \nu_{21}`
                     
        """
        E11 = self._material.E11
        E22 = self._material.E22
        nu21 = self._material.nu21
        nu12 = self._material.nu12
        G12 = self._material.G12

        D = 1 - nu12*nu21
        return np.array([[E11/D, nu12*E22/D, 0],
                         [nu12*E22/D, E22/D, 0],
                         [0, 0, G12]], dtype=np.float32)

    def at(self):
        """Return the (3,) vector of free thermal expansion coefficients for the 
        layer, in (1,2) layer material coordinate system.
        
        .. math::
        
            \mathbf{\alpha}_T = \left[ \begin{array}{c}
            \alpha_{T11} \\
            \alpha_{T22} \\
            0 
            \end{array} right]        
       
        """
        return np.array([self._material.a11t, self._material.a22t, 0],
                         dtype=np.float32)
        
    def bm(self):
        """Return the (3,) vector of free moisture expansion coefficients for the 
        layer, in (1,2) layer material coordinate system.
        
        .. math::
        
            \mathbf{\beta}_m = \left[ \begin{array}{c}
            \beta_{T11} \\
            \beta_{T22} \\
            0 
            \end{array} right]        
       
        """
        return np.array([self._material.b11m, self._material.b22m, 0],
                         dtype=np.float32)        
        

    def Sbar(self):
        """Return the 3x3 layer compliance matrix \bar{S} for plane stress state, 
        in (x, y) laminate coordinate system.
        
        .. math::
            \bar{\mathbf{S}} = \mathbf{T}_e \mathbf{S} \mathbf{T}_e^T
            
        .. math::
        
            \bar{\mathbf{S}} = \left[ \begin{array}{ccc}
            \bar{S}_{11} & \bar{S}_{12} & \bar{S}_{16} \\
            \bar{S}_{12} & \bar{S}_{22} & \bar{S}_{26} \\
            \bar{S}_{16} & \bar{S}_{26} & \bar{S}_{66}
            \end{array} right]               

        """
        Te = tmat_eps(self._theta)
        return np.dot(Te, np.dot(self.S(), Te.T))

    def Qbar(self):
        """Return the 3x3 layer stiffness matrix \bar{Q} for plane stress state, 
        in (x, y) laminate coordinate system.
        
        .. math::
            \bar{\mathbf{Q}} = \mathbf{T}_s \mathbf{Q} \mathbf{T}_s^T
            
        .. math::
        
            \bar{\mathbf{Q}} = \left[ \begin{array}{ccc}
            \bar{Q}_{11} & \bar{Q}_{12} & \bar{Q}_{16} \\
            \bar{Q}_{12} & \bar{Q}_{22} & \bar{Q}_{26} \\
            \bar{Q}_{16} & \bar{Q}_{26} & \bar{Q}_{66}
            \end{array} right]               
        """     
        return rotated_stiffness_matrix33(self.Q(), self._theta)   

    def atbar(self):
        """Return the (3,) vector of free thermal expansion coefficients for the 
        layer, in (x,y) laminate system.

        .. math::
            \bar{\mathbf{\alpha}}_T = \mathbf{T}_e \mathbf{\alpha}_T 
            
        """
        Te = tmat_eps(self._theta)
        return np.dot(Te, self.at())
    
    def bmbar(self):
        """Return the (3,) vector of free thermal expansion coefficients for the 
        layer, in (x,y) laminate system.

        .. math::
            \bar{\mathbf{\beta}}_M = \mathbf{T}_e \mathbf{\beta}_B 
            
        """
        Te = tmat_eps(self._theta)
        return np.dot(Te, self.bm())    
    

class DegradableLayer(Layer):
    
    k_mf_e22 = 0.01 # Tsai: applies a factor of 0.15 to matrix properties, then uses micromechanics
    k_mf_g12 = 0.01 # Tsai: applies a factor of 0.15 to matrix properties, then uses micromechanics
    k_mf_nu12 = 0.01 # Tsai: applies a factor of 0.15 to matrix properties, then uses micromechanics
    k_ff_e11 = 0.01 # Tsai: 0.01
    
    def __init__(self, laminate, alpha, material):
        """Create and return a new degradable layer. 
        """
        self._laminate = laminate
        self._material = material
        self._theta = alpha #math.radians(alpha)
        self._matrix_failure = False
        self._fibre_failure = False    
    
    def is_matrix_failed(self):
        """Return True if matrix failure has occurred in the Layer."""
        return self._matrix_failure
    
    def is_fibre_failed(self):
        """Return True if fibre failure has occurred in the Layer."""
        return self._fibre_failure
    
    def set_matrix_failure(self, b):
        """Set the matrix failure state of the Layer.
        
        :Parameters:
        - b: bool
        """
        assert b in (True, False)
        self._matrix_failure = b 
    
    def set_fibre_failure(self, b):
        """Set the fibre failure state of the Layer.
        
        :Parameters:
        - b: bool
        """
        assert b in (True, False)
        self._fibre_failure = b 

    def S(self):
        """Return the 3x3 layer compliance matrix S for plane stress state, 
        in (1,2) coordinate system.
        """

        E11 = self._material.E11 * (self.k_ff_e11 if self._fibre_failure else 1.0)
        E22 = self._material.E22 * (self.k_mf_e22 if self._matrix_failure else 1.0)
        nu21 = self._material.nu21 * (self.k_mf_nu12 if self._matrix_failure else 1.0)
        G12 = self._material.G12 * (self.k_mf_g12 if self._matrix_failure else 1.0)
        return np.array([[1/E11, -nu21/E22, 0],
                          [-nu21/E22, 1/E22, 0],
                          [0, 0, 1/G12]], dtype=np.float32)

    def Q(self):
        """Return the 3x3 layer stiffness matrix Q for plane stress state, 
        in (1,2) coordinate system.
        """
        E11 = self._material.E11 * (self.k_ff_e11 if self._fibre_failure else 1.0)
        E22 = self._material.E22 * (self.k_mf_e22 if self._matrix_failure else 1.0)
        nu21 = self._material.nu21 * (self.k_mf_nu12 if self._matrix_failure else 1.0)
        G12 = self._material.G12 * (self.k_mf_g12 if self._matrix_failure else 1.0)
        # FIXME: is nu21 unchanged??? 
        nu12 = self._material.nu12 
        D = 1 - nu12*nu21
        
        return np.array([[E11/D, nu12*E22/D, 0],
                         [nu12*E22/D, E22/D, 0],
                         [0, 0, G12]], dtype=np.float32)


class LayerWithExplicitThickness(DegradableLayer):
    """This class defines a layer, as the standard layer. difference: one may
    define the thickness explicitely; it is not taken from the material property.
    Used for Percentage Laminate and UnorderedLaminate
    """

    def __init__(self, laminate, alpha, material, thickness):
        self._laminate = laminate
        self._material = material
        self._theta = alpha #math.radians(alpha)
        self._thickness = thickness
        self._matrix_failure = False
        self._fibre_failure = False         

    def thickness(self):
        """Return the thickness of the Layer."""
        return self._thickness


# TODO: add a copy method! 
# we should always use a copy of the laminate in progressive failure analysis
# Benefit: we could run laminate strength analyses inn parallel.

class BaseLaminate(object):
    """superclass for laminates"""
    
    def validate_layer(self, layer_or_id):
        """Return a Layer index, or raise an error.
        
        lyr may be a layer instance or an index"""
        if layer_or_id in self._layers:
            return self._layers.index(layer_or_id)
        elif isinstance(layer_or_id, int):
            if layer_or_id >= 0 and layer_or_id < self.num_layers():
                return layer_or_id
            else:
                raise IndexError('index out of range:', layer_or_id)
        else:
            raise IndexError('No such layer')
    
    def is_membrane_isotropic(self):
        """Return True if the laminate is isotropic with respect to in-plane 
        properties.
        """ 
        raise NotImplementedError
    
    def num_layers(self):
        raise NotImplementedError
    
    def is_symmetric(self):
        raise NotImplementedError
    
    def is_balanced(self):
        raise NotImplementedError
    
    def to_layup(self):
        raise NotImplementedError
    
    def to_percentage_layup(self):
        raise NotImplementedError
    
    def to_stacking_sequence(self):
        raise NotImplementedError
    
    # TODO: would an iterator be better?
    @property
    def layers(self):
        """Return the laminate's layers, as a tuple. can be iterated over,
        can be accessed by index.
        """
        return self._layers    
    
    def get_z(self, layer_or_id, relpos=0.5):
        """Return global z position for layer, at relative layer position relpos.
        Global z = 0 is in laminate midplane!
        Layer may be given by instance, or by index.
        """
        lid = self.validate_layer(layer_or_id)
        z = -0.5 * self.thickness()
        z += sum(l.thickness() for l in self._layers[:lid])
        z += relpos * self._layers[lid].thickness()
        return z    
    
    def thickness(self):
        """Return the thickness of the laminate. 
        
        The laminate thickness is the sum of the thicknesses of the layers.
        
        .. math::
        
            t = \sum_{i} t_i
        
        """
        return sum(layer.thickness() for layer in self._layers)    
    
    def A(self, offaxis=0):
        """Return the extensional stiffness matrix A (3x3) of the laminate.
        
        The stiffness matrix relates membrane forces to the mid-plane strains 
        of the laminate.
        
        .. math::
        
            \mathbf{A} = sum_i \bar\mathbf{Q}}_i t_i
            
        .. math::
        
            \mathbf{A} = \left[ \begin{array}{ccc}
            \A_{11} & A_{12} & A_{16} \\
            \A_{12} & A_{22} & A_{26} \\
            \A_{16} & A_{26} & A_{66}
            \end{array} right] 
            
        .. math::
        
            \mathbf{N}= \mathbf{A} \mathbf{\varepsilon}_0
        
        """
        a = np.zeros((3,3), dtype=np.float32)
        for layer in self._layers:
            a += layer.Qbar() * layer.thickness()
        return rotated_stiffness_matrix33(a, -offaxis)    
    
    
    def Ai(self):
        return np.linalg.inv(self.A())
    
    def Ex(self, offaxis=0):
        """Return apparent in-plane elastic modulus Ex of the laminate. 
        If the offaxis angle (in radians) is defined, return apparent 
        modulus in a direction rotated by that angle.
        """
        a = self.A(-offaxis)
        ai = np.linalg.inv(a) 
        return 1.0 / (ai[0,0] * self.thickness())


    def Ey(self, offaxis=0):
        """Return apparent in-plane elastic modulus Ey of the laminate.
        
        If the offaxis angle (in radians) is defined, return apparent 
        modulus in a direction rotated by that angle.

        """
        a = self.A(-offaxis)
        ai = np.linalg.inv(a)  
        return 1.0 / (ai[1,1] * self.thickness())


    def Gxy(self, offaxis=0):
        """Return apparent in-plane shear modulus Gxy of the laminate."""

        a = self.A(-offaxis)
        ai = np.linalg.inv(a)  
        return 1.0 / (ai[2,2] * self.thickness())


    def nuxy(self, offaxis=0):
        """Return the larger in-plane Poisson ratio of the laminate."""
        a = self.A(-offaxis)
        ai = np.linalg.inv(a) 
        return -ai[0,1] / ai[1,1] # FIXME: a_11 or a_22 ???

    def nuyx(self, offaxis=0):
        """Return the smaller in-plane Poisson ratio of the laminate."""
        # schürmann, 10.14
        a = self.A(-offaxis)
        ai = np.linalg.inv(a) 
        return -ai[0,1] / ai[0,0]
    
    def scherzahl(self):
        # wiedemann, 4.1-6 and 4.1-7
        # compliance matrix:
        c = self.Ai() 
        dx = 1 / c[0,0]
        dy = 1 / c[1,1]
        dxy = 1 / (c[0,1] + c[2,2]/2)
        return math.sqrt(dx*dy)/dxy
    

    def thermal_force(self, deltaT):
        """Return (3,) thermal stress resultant."""
        nt = np.zeros(3)
        for layer in self.layers:
            nt += np.dot(layer.Qbar(), layer.atbar()) * layer.thickness()
        return deltaT * nt    
    
    def swelling_force(self, deltaM):
        """Return (3,) swelling stress resultant."""
        nm = np.zeros(3)
        for layer in self.layers:
            nm += np.dot(layer.Qbar(), layer.bmbar()) * layer.thickness()
        return deltaM * nm        
    
    
    def alphaT(self): 
        """Return effective laminate coefficients of thermal expansion."""
        # derivation mil-17
        # calculate thermal force for unit thermal load
        # FIXME: we use alpha_x for the layer and the laminate. fnd better nomenclature.
        nt1 = self.thermal_force(1)
        return np.dot(self.Ai(), nt1)

    def betaM(self): 
        """Return effective laminate coefficients of moisture expansion."""
        # derivation mil-17
        # calculate thermal force for unit thermal load
        # FIXME: we use alpha_x for the layer and the laminate. fnd better nomenclature.
        nm1 = self.swelling_force(1)
        return np.dot(self.Ai(), nm1)
    
    # FIXME: don't need single values
#     def axT(self):
#         """Return the apparent coefficient of thermal expansion in x direction
#         of the laminate."""
#         return self.alphaT()[0]
# 
#     def ayT(self):
#         """Return the apparent coefficient of thermal expansion in y direction
#         of the laminate."""
#         return self.alphaT()[1]
# 
#     def axyT(self):
#         """Return the apparent coefficient of thermal expansion in "xy direction"
#         of the laminate.
#         
#         This is the in-plane shear strain of the laminate if subjected to a 
#         unit temperature (uniform distribution over thickness). 
#         """
#         return self.alphaT()[2]
    

class MembraneLaminate(BaseLaminate):
    """Laminate having in-plane properties only, regardless of the stacking 
    sequence.
    The laminate can deal with a constant temperature distribution.
    """
    
    def __init__(self, layerdef):
        """Create a new Laminate. layerdef is an iterable of (angle, material) 
        tuples.
        """ 
        layers = []
        for (angle_deg, material) in validate_stacking(layerdef):
            angle_rad = math.radians(angle_deg)
            layers.append(DegradableLayer(self, angle_rad, material))
        self._layers = tuple(layers)    
        
    def num_layers(self):
        """Return number of layers."""
        return len(self._layers)        
        
    def get_linear_response(self, mload3, dtemp=0, dmoist=0):
        """Return a LinearSolution for the laminate under pure mechanical 
        loads.
        
        :Parameters:
        - `mload` is a (3,) array of applied forces (Nx, Ny, Nxy)
        - `dtemp`... temperature difference. float, default 0
        """
        
        # eps0: midplane strain
        abd = self.A()
        tload = self.thermal_force(dtemp)
        # swellng loads
        sload = self.swelling_force(dmoist)
        sol = np.linalg.solve(abd, mload3 + tload + sload) # berthelot, eq. 25.20
        return LinearMembraneSolution(self, mload3, dtemp, dmoist, sol)  
    

# this should accept different materials and raise a warning/error if 
# the laminate is too unsymmetric.
# Then make subclasses that deal with unordered and percentage laminates.
# Unordered and percentage laminates should accept different materials.

class UnorderedLaminate(MembraneLaminate):
    """Defines a laminate where layer count and orientations are given,
    but not stacking sequence.
    Works for single material only.
    Works for in-plane properties only.
    """
    
    def is_symmetric(self):
        """Return True.
        
        An unordered Laminate is considered always symmetric.
        """
        return True

    def __init__(self, layup, material):
        """Initialises the laminate.
        Layup: list of (angle, count) tuples.
        """
        assert isinstance(material, TransverseIsotropicPlyMaterial)
        assert all(isinstance(x, tuple) for x in layup)
        assert all(c > 0 for (a, c) in layup)

        self._material = material
        self._layup = layup

        # list of tuples: (angle, count) <- angles unique now.
        d = {}
        for (a, c) in layup:
            if a in d:
                d[a] += c
            else: d[a] = c
        unique_layup = sorted(d.items(), key=lambda x: x[0])
        # convert to radians
        rad_layup = [(math.radians(a), c) for(a, c) in unique_layup]

        layers = []
        for (ang, cnt) in rad_layup:
            lyr = LayerWithExplicitThickness(self, ang, material, cnt*material.t)
            layers.append(lyr)
        self._layers = tuple(layers) # make it immutable

    def get_layer_by_angle(self, alpha_deg):
        """Return layer that has the angle alpha_deg. Return None if no layer
        with that angle.
        """
        alpha_rad = math.radians(alpha_deg)
        angles = [layer.angle() for layer in self._layers]
        try:
            idx = angles.index(alpha_rad)
            return self._layers[idx]
        except ValueError:
            return None

    def num_layers(self):
        # FIXME: should I return len of layup instead???
        """Return number of layers."""
        return len(self._layers)
    


class Laminate(BaseLaminate):
    """Laminate having membrane and flexural properties."""

    def __init__(self, layerdef):
        """Create a new Laminate. layerdef is an iterable of (angle, material) 
        tuples.
        """ 
        layers = []
        for (angle_deg, material) in validate_stacking(layerdef):
            angle_rad = math.radians(angle_deg)
            layers.append(DegradableLayer(self, angle_rad, material))
        self._layers = tuple(layers)

    def is_symmetric(self):
        """Return True if the laminate is symmetric.
        Implemented as B == 0"""
        tol = 1.0e-6
        # normalise B by E t**2. Use max E11 of all layers for E, and laminate thickness for t
        emax = max([layer.material().E11 for layer in self._layers])
        bnorm = self.B() / (emax*self.thickness()**2)
        #print(bnorm)
        return np.all(bnorm <= tol)
    
    def num_layers(self):
        """Return number of layers."""
        return len(self._layers)

    
    def thermal_moment(self, deltaT):
        """Return (3,) thermal moment resultant."""
        mt = np.zeros(3)
        for layer in self.layers:
            za = self.get_z(layer, relpos=0.0)
            ze = self.get_z(layer, relpos=1.0)
            mt += np.dot(layer.Qbar(), layer.atbar()) * (ze**2 - za**2)
        return 0.5*deltaT*mt
    
    def swelling_moment(self, deltaM):
        """Return (3,) swelling moment resultant."""
        mm = np.zeros(3)
        for layer in self.layers:
            za = self.get_z(layer, relpos=0.0)
            ze = self.get_z(layer, relpos=1.0)
            mm += np.dot(layer.Qbar(), layer.bmbar()) * (ze**2 - za**2)
        return 0.5*deltaM*mm    

        
    def deltaT(self): 
        """Return laminate free thermal curvature."""
        # derivation mil-17 except for sign convention for Mt
        mt1 = self.thermal_moment(1)
        di = np.linalg.inv(self.D())
        return np.dot(di, mt1)
    
    def deltaM(self): 
        """Return laminate free swelling curvature."""
        # derivation mil-17 except for sign convention for Mt
        mm1 = self.swelling_moment(1)
        di = np.linalg.inv(self.D())
        return np.dot(di, mm1)
    
    def ABD(self):
        """Return the 6x6 ABD stiffness matrix."""
        b = self.B()
        abd = np.zeros((6,6), dtype=np.float32)
        abd[0:3, 0:3] = self.A()
        abd[0:3, 3:6] = b
        abd[3:6, 0:3] = b
        abd[3:6, 3:6] = self.D()
        return abd

    def ABDi(self):
        """Return the 6x6 laminate compliance matrix (inverse of the stiffness
        matrix).
        """
        return np.linalg.inv(self.ABD())

    def B(self):
        """Return the coupling stiffness matrix B (3x3) of the laminate."""
        b = np.zeros((3,3), dtype=np.float32)
        for layer in self._layers:
            za = self.get_z(layer, relpos=0.0)
            ze = self.get_z(layer, relpos=1.0)
            b += layer.Qbar() * (ze**2 - za**2) / 2.0
        return b

    def D(self):
        """Return the bending stiffness matrix D (3x3) of the laminate."""
        d = np.zeros((3,3), dtype=np.float32)
        for layer in self._layers:
            za = self.get_z(layer, relpos=0.0)
            ze = self.get_z(layer, relpos=1.0)
            d += layer.Qbar() * (ze**3 - za**3) / 3.0
        return d

    def Deff(self):
        """Return effective bending stiffness matrix Deff (3x3). For symmetric
        laminates, Deff == D.
        """
        # Wiedemann, 4.1-27: Deff = D - B.A^-1.B
        b = self.B()
        #ai = np.linalg.inv(self.A())
        return self.D() - np.dot(b, np.dot(self.Ai(), b))


    def Exf(self):
        """Return apparent modulus in bending, about y axis."""
        di = np.linalg.inv(self.Deff())
        return 12.0 / (di[0,0] * self.thickness()**3)

    def Eyf(self):
        """Return apparent modulus in bending, about x axis."""
        di = np.linalg.inv(self.Deff())
        return 12.0 / (di[1,1] * self.thickness()**3)

    def Gxyf(self):
        """Return apparent modulus in twisting, about xy axis."""
        di = np.linalg.inv(self.Deff())
        return 12.0 / (di[2,2] * self.thickness()**3)

    def nuxyf(self, oaa=1.):
        di = np.linalg.inv(self.Deff())
        return -di[0,1] / di[1,1]

    def nuyxf(self, oaa=1.):
        di = np.linalg.inv(self.Deff())
        return -di[0,1] / di[0,0]

    def kreuzsteifigkeit(self):
        # wiedemann, 4.1-21b
        # gilt das nur für symmetrischen Aufbau?
        d = self.D()
        return d[0,1] + 2*d[2,2]

    def kreuzzahl(self):
        # wiedemann, 4.1-22
        # gilt das nur für symmetrischen Aufbau?
        d = self.D()
        return self.kreuzsteifigkeit() / math.sqrt(d[0,0] * d[1,1])
    
    def get_linear_response(self, mload6, dtemp=0., dmoist=0.):
        """Return a LinearSolution for the laminate under pure mechanical loads.
        mload is a (6,) array: (Nx, Ny, Nxy, Mx, My, Mxy).
        
        :Parameters:
        - mload: membrane forces and bending moments, (6,) array of float
        - dtemp: temperature difference. optional, default = 0
        """
        # eps0: midplane strain
        # kappa0: midplane curvature
        
        abd = self.ABD()
        # thermal loads
        tload = np.zeros(6) 
        tload[:3] = self.thermal_force(dtemp)
        tload[3:] = self.thermal_moment(dtemp)
        # swellng loads
        sload = np.zeros(6) 
        sload[:3] = self.swelling_force(dmoist)
        sload[3:] = self.swelling_moment(dmoist)
        # solve it
        sol = np.linalg.solve(abd, mload6+tload+sload) # berthelot, eq. 25.20
        return LinearSolution(self, mload6, dtemp, dmoist, sol[:3], sol[3:])    


class _LinearResponse(object):
    """Superclass for solutions.
    
    The solution stores (and hides) the global displacement eps0, kappa. 
    """

    # strains in laminate system

    def eps_g(self, layer, relpos=0.5):
        """Return layer strains, in laminate coordinate system, as a 
        (3,) numpy array.
        
        Must be implemented by subclass. 
        """
        raise NotImplementedError

    def eps_g_m(self, layer, relpos=0.5):
        """Return engineering strain in layer `layer`, strain due to 
        mechanical loads."""
        raise NotImplementedError

    def eps_g_r(self, layer, relpos=0.5):
        """Return engineering strain in layer `layer`, strain due to 
        residual loads."""
        raise NotImplementedError
    
    # strains in material system

    def eps_l(self, layer, relpos=0.5):
        """Return strains in local (1,2) system: (e11, e22, gamma12) in layer lyr, at
        relative height relpos.
        These are engineering strains: (eps_11, eps_22, gamma_12)
        
        :Parameters:
        - layer: lid or instance
        - relpos: relative z position within layer, 0 <= relpos <= 1
        
        :Returns:
        - eps_l: engineering strains. numpy array (3,), float
        """
        #layer = self._laminate.validate_layer(layer_or_id)
        eps = self.eps_g(layer, relpos=relpos)
        te = tmat_eps(layer.angle())
        return np.linalg.solve(te, eps)


    def eps_l_r(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        epsg = self.eps_g_r(layer, relpos=relpos)
        te = tmat_eps(layer.angle())
        return np.linalg.solve(te, epsg)

    def eps_l_m(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        eps = self.eps_g_m(layer, relpos=relpos)
        te = tmat_eps(layer.angle())
        return np.linalg.solve(te, eps)

    # stresses in laminate system

    def sigma_g(self, layer, relpos=0.5):
        """Return stresses in laminate reference system: (sxx, syy, tauxy) 
        in layer lyr, at relative height relpos.
        These are total stresses sigma_m + sigma_r
        
        :Parameters:
        - layer: lid or instance
        - relpos: relative z position within layer, 0 <= relpos <= 1
        
        :Returns:
        - sigma_g: stresses. numpy array (3,), float
        """        
        #layer = self._laminate.validate_layer(layer_or_id)
        epsg = self.eps_g(layer, relpos=relpos)
        epsg -= self._dtemp*layer.atbar() 
        epsg -= self._dmoist*layer.bmbar()
        return np.dot(layer.Qbar(), epsg)

    def sigma_g_r(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        epsg = self.eps_g_r(layer, relpos=relpos) 
        epsg -= self._dtemp*layer.atbar() 
        epsg -= self._dmoist*layer.bmbar()
        return np.dot(layer.Qbar(), epsg)

    def sigma_g_m(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        epsg = self.eps_g_m(layer, relpos=relpos)  
        epsg -= self._dtemp*layer.atbar() 
        epsg -= self._dmoist*layer.bmbar()
        return np.dot(layer.Qbar(), epsg)


    # stresses in material system 

    def sigma_l(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        epsl = self.eps_l(layer, relpos)
        epsl -= self._dtemp*layer.at() # thermal
        epsl -= self._dmoist*layer.bm() # moisture
        return np.dot(layer.Q(), epsl)
    
    def sigma_l_r(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        epsr = self.eps_l_r(layer, relpos)
        epsr -= self._dtemp*layer.at()
        epsr -= self._dmoist*layer.bm()
        return np.dot(layer.Q(), epsr)

    def sigma_l_m(self, layer, relpos=0.5):
        #layer = self._laminate.validate_layer(layer_or_id)
        epsm = self.eps_l_m(layer, relpos)
        epsm -= self._dtemp*layer.at()
        epsm -= self._dmoist*layer.bm()
        return np.dot(layer.Q(), epsm)



class LinearSolution(_LinearResponse):

    def __init__(self, laminate, mload, dtemp, dmoist, eps0, kappa):
        self._laminate = laminate

        self._eps0 = eps0
        self._kappa = kappa
        
        self._at = self._laminate.alphaT() # free thermal expansion
        self._dt = self._laminate.deltaT() # free thermal curvature

        self._bm = self._laminate.betaM() # free thermal expansion
        self._dm = self._laminate.deltaM() # free thermal curvature

        self._mload = mload
        self._dtemp = dtemp # uniform temperature
        self._dmoist = dmoist # uniform moisture

    @property
    def layers(self):
        return self._laminate.layers

    def eps_kappa(self):
        return np.hstack([self._eps0, self._kappa])

    def eps_g(self, layer, relpos=0.5):
        """Return strains in global (x,y) system in layer , at relative layer height eta.
        lyr can be an integer -> layer id
        or it can be a reference to the layer itself
        These are engineering strains: (eps_xx, eps_yy, gamma_xy)

        Raise ValueError id eta < 0 or eta > 1.0
        Raise IndexError -> Layerid < 0 or >= numlayers
        """
        z = self._laminate.get_z(layer, relpos)
        return self._eps0 + z*self._kappa
    
    def eps_g_m(self, layer, relpos=0.5):
        z = self._laminate.get_z(layer, relpos)
        eps = self._eps0 + z*self._kappa
        eps -= (self._at + z*self._dt) * self._dtemp
        eps -= (self._bm + z*self._dm) * self._dmoist
        return eps

    def eps_g_r(self, layer, relpos=0.5):
        z = self._laminate.get_z(layer, relpos)
        eps = (self._at + z*self._dt) * self._dtemp
        eps += (self._bm + z*self._dm) * self._dmoist
        return eps

    
class LinearMembraneSolution(_LinearResponse):

    def __init__(self, laminate, mload, dtemp, dmoist, eps0):
        """eps0 are the midplane strains in laminate system; for total 
        loading (mech + therm)
        """
        self._laminate = laminate
        self._eps0 = eps0
        self._mload = mload
        self._dtemp = dtemp # uniform temperature
        self._dmoist = dmoist
        self._at = self._laminate.alphaT() # free thermal expansion
        self._bm = self._laminate.betaM()

    @property
    def layers(self):
        return self._laminate._layers
    
    def eps0(self):
        return self._eps0
    
    def eps_g(self, layer, relpos=0.5):
        """Return strains in global (x,y) system in layer lid, at relative layer height eta.
        lyr can be an integer -> layer id
        or it can be a reference to the layer itself
        These are engineering strains: (eps_xx, eps_yy, gamma_xy)

        Raise ValueError id eta < 0 or eta > 1.0
        Raise IndexError -> Layerid < 0 or >= numlayers
        """
        return self._eps0
    
    def eps_g_m(self, layer, relpos=0.5):
        return self._eps0 - self._at*self._dtemp - self._bm*self._dmoist

    def eps_g_r(self, layer, relpos=0.5):
        return self._at * self._dtemp + self._bm*self._dmoist

        
# FIXME: move this into utility module
def stiffness_polar(lam, prop, step=10):
    """Return laminate stiffness property for an off-axis angle (in degrees).
    prop can be one of: Ex, Ey, ...,  A11, A12, ...
    default step: 10 degrees
    function returns values from 0 to 360, including (!) the 360 value. 
    """
    assert isinstance(prop, str)
    angles = np.arange(0, 360+step, step)
    result = []
    # Membrane Moduli
    if prop in ['Ex', 'Ey', 'Gxy', 'nuxy', 'nuyx']:
        func = getattr(lam, prop)
        for angle in angles:
            value = func(offaxis=np.radians(angle))
            result.append((angle, value))
        return result
    # Membrane stiffness
    elif prop in ["A11", "A12", "A16", "A21", "A22", "A26", "A61", "A62", "A66"]:
        dummy, row, col = prop
        map_index = {"1": 0, "2": 1, "6": 2}
        irow = map_index[row]
        icol = map_index[col]
        for angle in angles:
            a = lam.A(offaxis=np.radians(angle))
            result.append((angle, a[irow, icol]))
        return result
    else:
        raise ValueError('no such property: {}'.format(prop))
    
