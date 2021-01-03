
# CLT

## Module CLT

This module provides classes and functions for classical laminate analysis
of laminates in plane stress state.

The main classes defined here are: 

The `Laminate` class stores layers, provides methods to access its layers. It 
provided methods to access inherent properties of the laminate (stiffness
data). It also provides a method for calculating the response of the laminate
to external loads.

The `Layer` class describes the properties of a single layer inside a laminate.
The layer class can store information on damage state.

The `Response` class stores the solution of a classical lamainate analysis.
It provides methods to retrieve streses and strains.


### options to define a laminate (all angles in degrees!)

1. explicit stacking sequence:
   list ot tuple of (angle, material) pairs

1. explicit stacking sequence (uniform material):
    list of angles, plus a Ply material (which has a thickness property)

2. layup:
   list of (angle, count) pairs, plus a Ply material (having a thickness property)

3. percentage layup:
   list of (angle, percentage) pairs, plus a material, plus a thickness



### Stresses in a layer

Stresses in plane stress state, in an arbitrary laminate reference system

.. math::
    \mathbf{\sigma}_x =  \left[ \begin{array}{c}
    \sigma_{xx} \\
    \sigma_{yy} \\
    \tau_{xy}
    \end{array} \right]

Stresses in plane stress state, in the principal material axis system of a 
ply: 

.. math::
    \mathbf{\sigma}_l =  \left[ \begin{array}{c}
    \sigma_{11} \\
    \sigma_{22} \\
    \tau_{12}
    \end{array} \right]
    
In most cases an abbreviated notation is used, :math:` \sigma_1 = \sigma_{11}`, 
:math:` \sigma_2 = \sigma_{22}` and :math:` \sigma_6 = \tau_{12}`  

Transformation of stresses:

.. math::
    \mathbf{\sigma}_x = \mathbf{T}_s \mathbf{\sigma}_l

.. math::
    \mathbf{\sigma}_l = \mathbf{T}_s^{-1} \mathbf{\sigma}_x


### Strains in a layer

Strains in plane stress state, in an arbitrary laminate reference system.
Note that these are engineering strains, :math:` \gamma_{xy} = 2 \varepsilon_{xy}` .

.. math::
    \mathbf{\varepsilon}_x =  \left[ \begin{array}{c}
    \varepsilon_{xx} \\
    \varepsilon_{yy} \\
    \gamma_{xy}
    \end{array} \right]

Strains in the principal material axis system of a ply: 

.. math::
    \mathbf{\varepsilon}_l =  \left[ \begin{array}{c}
    \varepsilon_{11} \\
    \varepsilon_{22} \\
    \gamma_{12}
    \end{array} \right]
    
In most cases an abbreviated notation is used, :math:` \varepsilon_1 = \varepsilon_{11}`, 
:math:` \varepsilon_2 = \varepsilon_{22}` and :math:` \varepsilon_6 = \gamma_{12}`.

Transformation of strains:

.. math::
    \mathbf{\varepsilon}_x = \mathbf{T}_e \mathbf{\varepsilon}_l

.. math::
    \mathbf{\varepsilon}_l = \mathbf{T}_e^{-1} \mathbf{\varepsilon}_x


### Stress-Strain Relationship for a single layer

In principal material axis system:

.. math::
    \mathbf{\varepsilon}_l = \mathbf{S} \mathbf{\sigma}_l
    
.. math::
    \mathbf{\sigma}_l = \mathbf{Q} \mathbf{\varepsilon}_l
    
In arbitrary (laminate) reference system:
    
.. math::
    \mathbf{\varepsilon}_x = \mathbf{\bar{S}} \mathbf{\sigma}_x
    
.. math::
    \mathbf{\sigma}_x = \mathbf{\bar{Q}} \mathbf{\varepsilon}_x
    

### Apparent elastic properties of a laminate

Ref. [schür]_ eq. (10.14) 
 
.. math::
 
    E_x = \frac{1}{a_{11} t}
    
where :math:` a = A^{-1}` is the inserve membrane stiffness matrix 
and t is the laminate thickness.

        
.. math::
 
    E_x = \frac{1}{a_{22} t}
    
     
.. math::
    G_{xy} = \frac{1}{a_{66} t} 
    
.. math::
    \nu_{xy} = - \frac{a_{12}}{a_{22}} 
    
    
### Thermal Expansion

Total strain in lamina is sum of strain due to load and thermal expansion:
\epsilon_l = \epsilon_l^m + \epsilon_l^t

thermal expansion is
\epsilon_l^t = \alpha_l DT

where \alpha_l = (alpha1, alpha2, 0) are the coefficients of free thermal 
expansion of the layer. 

strains due to >>free<< thermal expansion do not cause stresses! 
Stress in layer:
\sigma_l = Q ( \epsilon_l - \alpha_l DT )

In laminate system:
\sigma_x = Qbar ( \epsilon_x - \alpha_x DT )

Thermal force vector:
NT = sum Qbar \alpha_x (ze - za) DT

Thermal moments:
MT = 0.5 * sum Qbar \alpha_x (ze**2 - za**2) DT

Effective coefficients of thermal expansion for the laminate:
\alpha_XL = A^-1 NT / DT
Thermal curvature of the laminate (is that also valid for unsymmetrical ones?):
\delta_XL = D^-1 MT / DT

Solution:
N = A eps0 + B kappa - NT
M = B eps0 + D kappa - MT

Stress and strain recovery:

strains, global system: 
* eps_x =   eps0 + z*kappa
* eps_x_r = (\alpha_XL + z \delta_XL) DT            
* eps_x_m = eps_x - (\alpha_XL + z \delta_XL) DT   

strains, local system:
* eps_l =   Te^-1 * eps_x
* eps_l_r = Te^-1 * eps_x_r
* eps_l_m = Te^-1 * eps_x_m

stresses, global system:
* sigma_x =   Qbar * (eps_x - atbar DT)
* sigma_x_r = Qbar * (eps_x_r - atbar DT)
* sigma_x_m = Qbar * (eps_x_m - atbar DT)

stresses, local system:
* sigma_l = Q * (eps_l - at DT)
* sigma_l_r = Q * (eps_l_r - at DT)
* sigma_l_m = Q * (eps_l_m - at DT)

### References

.. [berth] Jean-Marie Berthelot: Composite Materials. Springer 1999
.. [schür] Helmut Schürmann: Konstruieren mit Faser-Kunststoff-Verbunden. 2. Auflage, Springer 2007
.. [jones] Robert M. Jones: Mechanics of Composite Materials, 2nd ed. Tayloer & Francis 1999
.. [mil173f] Composite Materials Handbook - Volume 3, Polymer Matrix Composites, Material Usage, Design, and Analysis. MIL-HDBK-17-3F, 2002
.. [esdu83014] Failure criteria for an individual layer of a fibre reinforced composite laminate under in-plane loading. ESDU Data Item 83014. IHS ESDU, 1983
.. [Lekhn] Anisotropic plates
.. [aeros] Aerospatiale composite stress manual
.. [vdi2014] VDI 2014
 

## Module failurecriteria

Methods for strength analysis for PLANE STRESS in UD REINFORCED LAYERS!

Not for:
- 3D stress
- transverse shear
- cloth materials
- ... anything else 

Determine Strength/Stress ratio and Failure Index where there is a combination 
of mechanical and residual stresses:
Ref. Tsai, Theory of Composites Design, Ch. 9

Strength ratios are split in two parts: one for mechanical strains (Rm), the 
other one for the residual strains (Rr). Each part of the strain can act 
independently.

Quadratic Equation:
Fij si sj + Fi si = 1
with: si = sim + sir
Fij (Rm*sim + Rr*sir) (Rm*sjm + Rr*sjr) + Fi (Rm*sim + Rr*sir) = 1

Assume Rr = 1 (no variation in residual stresses) -> determine mechanical 
strength ratio. 
a*Rm**2 + b*Rm + c = 0
Coefficients of quadratic equation:
a = am, 
b = bsum = bm + bmix
c = -1 + ar + br

am ... quadratic term, mechanical stresses only
ar ... quadratic term, residual stresses only
bm ... linear term, mechanical stresses only
br ... linear term, residual stresses only
bmix = 2 * (F11*s1m*s1r + F12*s1m*s2r + F12*s2m*s1r + F22*s2m*s2r + F66*s6m*s6r)

Tsai-Hill choses strength properties depending of stress (tension or compression).
In case of combined mechanical and residual stresses, we don't know in advance 
which sign the final stresses will have. 
Solution: calculate all 4 combinations, select the one where signs of calculated 
failure stresses math with the criteria.
 
Alternative Solution: 
line segment: mechanical stress vector, from (s11r, s22r) to (s11r+s11m, s22r+s22m)
Point: tensile strength F22t
if point is left of line segment -> result will be in tensile (s11) region.
if point is right of line segment -> result will be in compressive (s11) region.
same for the other Points -> we can decide in advance, in which quadrant the solution will be.
Can use the computational geometry functions for that. 



### Failure stuff

We use the Tsai terminology here.

"Reserve Factor": is actually the "stretch factor" fs. Kollar uses term "stress ratio" for this.
(RF*s1/F1)**2 + (RF*s2/F2)**2 == 1
Tsai calls this "Strength/Stress Ratio" or "Strength Ratio", R: 
  ratio between ultimate (allowable) stress and the applied stress.

general solution: RF = (-b + sqrt(b**2 + 4*a) / (2*a)
where: a ... quadratic terms
       b ... linear terms

Failure Ratio: this is used here as sigma_applied/sigma_allowable -> "stress exposure", fe
(s1/(k*F1))**2 + (s2/(k*F2))**2 == 1
Tsai calls this "Failure Index" k: k = 1/R

need to distinguish between the different failure modes
e.g. for max stress: tension/compression, ...
return record arrays?
fmode, value, ...
could be extended by user: ply number, ...

### Implementation

Each individual function is a class!
+ can be customised via initialisation
  - set Tsai-Wu parameter
  - set behaviour: scale all stresses, scale mechanical stresses only, ...
  - set material types that it is valid for (like "UD").
+ __call__ method
+ other methods/attributes, like
  - is_fibre_failure
  - is_matrix_failure
  - is_tension
  - is_compression
  - is_plane_stress
  - the "theory" it belongs to
  - etc.
