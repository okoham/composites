# -*- coding: utf-8 -*-
"""
Created on 27.09.2014
@author: Oliver

This module provides methods for plain strength analysis of unnotched laminates.

A) First ply failure, regardless of mode (based on elastic stresses in 
   undegraded plies)
- works with all criteria
- one-shot analysis

B) First fibre failure, based on elastic stresses in degraded plies (all plies 
  degraded from the beginning)
  - works with criteria that distinguish between fibre and matrix failure 

C) Progressive Failure Analysis
- reduce stiffness as soon as matrix failure occurred
- Laminate failure: when
  i) first fibre failure occurs or
  ii) matrix failure has occurred in all layers
- works only with those criteria that can distinguish between matrix and fibre
  failure

"""

from __future__ import division, print_function
from collections import namedtuple
import numpy as np
import failurecriteria
import math
import clt


#################################

# a) FPF, based on elastic stresses, without matrix degradation
# b) First fibre fracture, degraded matrix properties, one-shot analysis
#    (like Airbus) -> is that netting theory???
# c) full progressive analysis until first fibre failure / or all matrix failure
# d) full progressive analysis until last ply failure 
    

failure_classes_a = {'MaxStress':   [failurecriteria.MaxStressTension1(),
                                     failurecriteria.MaxStressTension2(),
                                     failurecriteria.MaxStressCompression1(),
                                     failurecriteria.MaxStressCompression2(),
                                     failurecriteria.MaxStressShear()],
                     'MaxStrain':   [failurecriteria.MaxStrainTension1(),
                                     failurecriteria.MaxStrainTension2(),
                                     failurecriteria.MaxStrainCompression1(),
                                     failurecriteria.MaxStrainCompression2(),
                                     failurecriteria.MaxStrainShear()],
                     'Hoffmann':    [failurecriteria.Hoffmann()],
                     'TsaiHill':    [failurecriteria.TsaiHill()],
                     'TsaiWu/0.0':  [failurecriteria.TsaiWu(0.0)],
                     'TsaiWu/-0.5': [failurecriteria.TsaiWu(-0.5)],
                     'Airbus':      [failurecriteria.YamadaSunTension(),
                                     failurecriteria.YamadaSunCompression(),
                                     failurecriteria.ModPuckAirbus()],
                     'Hashin A':    [failurecriteria.HashinFibreTension(),
                                     failurecriteria.HashinFibreCompressionA(),
                                     failurecriteria.HashinMatrixTension(),
                                     failurecriteria.HashinMatrixCompression()],
                     'Hashin B':    [failurecriteria.HashinFibreTension(),
                                     failurecriteria.HashinFibreCompressionB(),
                                     failurecriteria.HashinMatrixTension(),
                                     failurecriteria.HashinMatrixCompression()],
                     'Puck':        [failurecriteria.PuckFibreTension(),
                                     failurecriteria.PuckFibreCompression(),
                                     failurecriteria.PuckMatrixTension(),
                                     failurecriteria.PuckMatrixCompression()],
                     'ModPuck':     [failurecriteria.PuckFibreTension(),
                                     failurecriteria.PuckFibreCompression(),
                                     failurecriteria.ModPuckMatrixFailure()],
                     'Puck C':      [failurecriteria.PuckFibreTension(),
                                     failurecriteria.PuckFibreCompression(),
                                     failurecriteria.PuckActionPlaneMF()],
                     }

# these include only such criteria that can distinguish between fibre and matrix 
# modes. Only the fibre failure classes are retained.
failure_classes_b = {'MaxStress':   [failurecriteria.MaxStressTension1(),
                                     failurecriteria.MaxStressCompression1()],
                     'MaxStrain':   [failurecriteria.MaxStrainTension1(),
                                     failurecriteria.MaxStrainCompression1()],
                     'Airbus':      [failurecriteria.YamadaSunTension(),
                                     failurecriteria.YamadaSunCompression()],
                     'Hashin A':    [failurecriteria.HashinFibreTension(),
                                     failurecriteria.HashinFibreCompressionA()],
                     'Hashin B':    [failurecriteria.HashinFibreTension(),
                                     failurecriteria.HashinFibreCompressionB()],
                     'Puck':        [failurecriteria.PuckFibreTension(),
                                     failurecriteria.PuckFibreCompression()],
                     'ModPuck':     [failurecriteria.PuckFibreTension(),
                                     failurecriteria.PuckFibreCompression()],
                     'Puck C':      [failurecriteria.PuckFibreTension(),
                                     failurecriteria.PuckFibreCompression()],
                     }

def layer_strength_analysis_a(mat, failure_criterion, s12m, s12r=np.zeros(3)):
    """Return strength ratios for a simple stress state."""
    try:
        fclasses = failure_classes_a[failure_criterion]
    except KeyError:
        raise    
    return [fclass.r(mat, s12m, s12r) for fclass in fclasses]



def layer_strength_enveloppe_2d(mat, comp1, comp2, criterion):
    # FIXME: move this into a "utility" module
    """Return a 2D strength enveloppe / interaction diagram. 2 components 
    are varied, the third one is zero.
    this works on 1-2-6 stresses
    this 
    """ 
    # FIXME: yields very high values (~inf) if laminate consists of one layer only.
    assert comp1 in ('s1', 's2', 's6')
    assert comp2 in ('s2', 's2', 's6')
    assert comp1 != comp2
    
    indices = {'s1': 0, 's2': 1, 's6': 2}

    first = indices[comp1]
    second = indices[comp2]
    
    all_res = []
    angles = np.radians(np.linspace(0, 360, 360))
    #lam = clt.MembraneLaminate([(0, mat)])
    #psa = PlainStrengthAnalysisA(lam, criterion)
    
    theta = 0
    
    for theta in angles:
        load = np.zeros(3) 
        load[[first, second]] = [math.cos(theta), math.sin(theta)] 
        load *= 100
        #res = psa.min_rf(load, 0)
        results = layer_strength_analysis_a(mat, criterion, load)
        results = [(r, fmode) for (r, fmode) in results if r is not None]
        rmin, fmode = min(results, key=lambda x: x[0])
        all_res.append((load*rmin, fmode))
    return all_res


def offaxis_layer_strength(mat, sign, criterion):
    # FIXME: move this into a "utility" module, or into the failure criteria module,
    # because it is related to the strength of a single layer (not lamminates). 
    """Return a 2D strength enveloppe / interaction diagram. 2 components 
    are varied, the third one is zero.
    this works on 1-2-6 stresses
    this 
    """ 
    signs = {'tension': 1, 'compression': -1}
    all_res = []
    angles = np.linspace(0, 90, 90) # in degrees
    scale = 100
    sx = signs[sign]*np.array([100, 0, 0]) 
    
    for theta in angles:
        ts = clt.tmat_sig(math.radians(theta))
        tsi = np.linalg.inv(ts)
        load = np.dot(tsi, sx)
        results = layer_strength_analysis_a(mat, criterion, load)
        results = [res for res in results if res[0] is not None]
        rmin, fmode = min(results, key=lambda x: x[0])
        all_res.append((theta, scale*rmin, fmode))
    return all_res


"""
StrengthResult is a "named tuple"  used to return strength analysis results.

Attributes:
- lid: int. Layer id, starting at zero
- angle: float. layer angle in degrees
- eta: float, 0 <= eta <= 1. position inside layer
- r: float. strength ratio
- fmode: string. description of the failure mode 
"""
StrengthResult = namedtuple('StrengthResult', ['lid', 'angle', 'eta', 'r', 'fmode'])


class PlainStrengthAnalysisA(object):
    """Return strength ratios for a given load.
    Kind = A: FPF, based on elastic stresses, without matrix degradation"""
    def __init__(self, lam, failure_criterion, zpositions=[0.5]):
        self._lam = lam
        try:
            self._fclasses = failure_classes_a[failure_criterion]
        except KeyError:
            raise
        if len(zpositions) > 0 and all(eta >= 0 for eta in zpositions) and all(eta <= 1 for eta in zpositions):
            self._zpositions = zpositions
        else:
            raise ValueError('0 <= zposistions <= 1')
        
    def all_rf(self, mload, dt):
        sol = self._lam.get_linear_response(mload, dt)
        rfs = []
        for i, layer in enumerate(self._lam.layers):
            mat = layer.material()
            angle = layer.angle()
            for eta in self._zpositions:
                sigma12m = sol.sigma_l_m(layer, relpos=eta)
                sigma12r = sol.sigma_l_r(layer, relpos=eta)
                for fclass in self._fclasses:
                    rf, fmode = fclass.r(mat, sigma12m, sigma12r)
                    rfs.append(StrengthResult(i, math.degrees(angle), eta, rf, fmode))
        return rfs     

    def min_rf(self, mload, dt):
        all_rfs = self.all_rf(mload, dt)
        all_rfs = filter(lambda x: x.r is not None, all_rfs)
        return min(all_rfs, key=lambda x: x.r)
    
    def minrf_in_layer(self, lid, mload, dt):
        all_rfs = self.all_rf(mload, dt)
        all_rfs = filter(lambda x: x.r is not None, all_rfs)
        layer_rfs = filter(lambda x: x.lid == lid, all_rfs)
        return min(layer_rfs, key=lambda x: x.r)        

        

def analyse_undamaged_layers(lam, mload, dt, failure_criterion,zpositions=[0.5]):
    """
    Return results for selected layers. This is all by default.
    provide layer ids to be excluded.
    no analysis foe layers that are already damaged!!!
    """
    fclasses = failure_classes_a[failure_criterion]
    sol = lam.get_linear_response(mload, dt)
    rfs = []
    for i, layer in enumerate(lam.layers):

        mat = layer.material()
        angle = layer.angle()
        for eta in zpositions:
            # TODO: calculate stresses only for the relevant layers
            sigma12m = sol.sigma_l_m(layer, relpos=eta)
            sigma12r = sol.sigma_l_r(layer, relpos=eta)

            if not layer.is_matrix_failed():
                for fclass in filter(lambda x: x.is_matrix_failure(), fclasses):
                    rf, fmode = fclass.r(mat, sigma12m, sigma12r)
                    rfs.append(StrengthResult(i, math.degrees(angle), eta, rf, fmode))

            if not layer.is_fibre_failed():
                for fclass in filter(lambda x: x.is_fibre_failure(), fclasses):
                    rf, fmode = fclass.r(mat, sigma12m, sigma12r)
                    rfs.append(StrengthResult(i, math.degrees(angle), eta, rf, fmode))
    return rfs   


def analyse_undamaged_layers_min(lam, mload, dt, failure_criterion,zpositions=[0.5]):
    rfs = analyse_undamaged_layers(lam, mload, dt, failure_criterion, zpositions)
    rfs = [x for x in rfs if x.r is not None]
    return min(rfs, key=lambda x: x.r)


def degrade_layer(lam, lid, fmode):
    """Degrade the layer lid.
    
    If fmode is MF: degrade matrix properties
    If fmode is FF: degrade fibre and matrix properties.
    """
    if 'FF' in fmode: 
        lam.layers[lid].set_fibre_failure(True)
        lam.layers[lid].set_matrix_failure(True)
    elif 'MF' in fmode:
        lam.layers[lid].set_matrix_failure(True)


def failure_analysis_b(lam, mload, dt, theory, zpositions=[0.5]):
    """Analyse fibre failure only. All matrix properties are degraded from the beginning.
    """
    assert theory in failure_classes_b
    for layer in lam.layers:
        layer.set_matrix_failure(True)
    res_min = analyse_undamaged_layers_min(lam, mload, dt, theory, zpositions=[0.5])
    # restore laminate
    for layer in lam.layers:
        layer.set_matrix_failure(False)
        layer.set_matrix_failure(False)
    return res_min    


# TODO: make a class out of this.
def failure_analysis_c(lam, mload, dt, theory, zpositions=[0.5], 
                       stop_on_last_matrix_failure=False,
                       full_output=False,
                       reset_laminate=True):
    """Return the strength ratio determined from progressive failure analysis.
    
    This is first fibre failure.
    """
    # TODO: analyse pure residual stresses at the beginning
    # TODO: make a copy of the laminate  
    # TODO: make "Stop on all matrix failures" optional.
    assert theory in failure_classes_b
    assert reset_laminate in (True, False)
    
    print(lam)

    print('* Residual Stresses')
    print('  to be added')
    
    load_level = 0.0
    history = {}

    i = 0
    # FIXME: find better termination criterion
    while True:
        i += 1
        #matrix_failures = [layer.is_matrix_failed() for layer in lam.layers]
        fibre_failures = [layer.is_fibre_failed() for layer in lam.layers]

        if any(fibre_failures):# or all(matrix_failures):
            print('-> Stop because of Fibre Failure\n\n')
            break

        res_min = analyse_undamaged_layers_min(lam, mload, dt, theory, zpositions)
        load_level = res_min.r 
        history[load_level] = [res_min]
        print('* Failure at load level: ', res_min.r, '\n  + ', res_min)
        degrade_layer(lam, res_min.lid, res_min.fmode)
        
        # inner loop: degrade all other plies that fail now at the given load level 
        NN  = 0
        while NN < 2*lam.num_layers():
            NN += 1
            try:
                res_min = analyse_undamaged_layers_min(lam, mload, dt, theory, zpositions)
            except ValueError:
                print('-> Stop inner loop: no further solution found - all layers damaged')
                # all are damaged
                break
            if res_min.r <= load_level:
                print('  - ', res_min)
                history[load_level].append(res_min)
                degrade_layer(lam, res_min.lid, res_min.fmode)
            else:
                break
            
        if i > 2*lam.num_layers(): # worst case: 1x matrix, then 1 x fibre failure per ply.
            print('-> Stop: too many iterations')
            break
        
    # restore laminate
    if reset_laminate:
        for layer in lam.layers:
            layer.set_matrix_failure(False)
            layer.set_matrix_failure(False)
        
    return load_level, history


def strength_polar(lam, dt, failure_criterion, load='tension', method='B'):
    """Return laminate strength, for uniaxial loading, loading direction varies 360 deg.
    Returns fluxes.
    membrane loading only.
    membrane laminates only
    select tensile or compressive load.
    """
    signs = {'tension': 1, 'compression': -1}
    assert isinstance(lam, clt.MembraneLaminate)
    assert load in signs
    # assume 1 N/mm
    sign = signs[load]
    n0 = sign * np.array([1, 0, 0])
    angles = np.arange(0, 360, 5) # 5° step
    
    all_res = []
    
    for theta in angles:
        theta_rad = np.radians(theta)
        ts = clt.tmat_sig(theta_rad)
        ntheta = np.dot(np.linalg.inv(ts), n0)
        strength = failure_analysis_b(lam, ntheta, dt, failure_criterion)
        all_res.append((theta, strength))
        
    return all_res
    

def strength_enveloppe_2d(lam, dt, comp1, comp2, failure_criterion):
    """Return a 2D strength enveloppe / interaction diagram. mechanical load 
    is given, 2 components 
    are varied, the third one is zero.
    """ 
    # FIXME: yields very high values (~inf) if laminate consists of one layer only.
    assert comp1 in ('Nx', 'Ny', 'Nxy')
    assert comp2 in ('Nx', 'Ny', 'Nxy')
    assert comp1 != comp2
    
    indices = {'Nx': 0, 'Ny': 1, 'Nxy': 2}

    first = indices[comp1]
    second = indices[comp2]
    
    all_res = []
    angles = np.radians(np.arange(0, 360, 5))
    
    for theta in angles:
        load = np.zeros(3) 
        load[[first, second]] = [math.cos(theta), math.sin(theta)] 
        strength = failure_analysis_b(lam, load, dt, failure_criterion)
        failure_load = load*strength.r
        all_res.append((failure_load, strength))
    return all_res



def strength_enveloppe_3d(lam, dt, failure_criterion):
    """Return a 3D strength enveloppe. 
    Returns fluxes
    """ 
    # FIXME: need a grid or a triangulation to plot this as a surface
    # spherical coordinates, phi = 0 ... 360, theta = 0 ... 180
    # x = r sin(theta) cos(phi) // y = r sin(theta) sin(phi) // z = r cos(theta)

    all_res = []
    r_phi = np.linspace(0, 2*np.pi, 36)
    r_theta = np.linspace(0, np.pi, 18)
    
    # see matplotlib/Examples/mplot3d/surface3d_demo2.py -> does not work here ...
#     nx_unit = np.outer(np.cos(r_phi), np.sin(r_theta))
#     ny_unit = np.outer(np.sin(r_phi), np.sin(r_theta))
#     nxy_unit = np.outer(np.ones(np.size(r_phi)), np.cos(r_theta))
#     
#     for (nx, ny, nxy) in zip(nx_unit.flat, ny_unit.flat, nxy_unit.flat):
#         load = np.array([nx, ny, nxy])
#         strength = failure_analysis_b(lam, load, dt, failure_criterion)
#         failure_load = load*strength.r
#         all_res.append((failure_load, strength))
        
    # this can now be plotted with Axes3D.plot_surface(x, y, z)
    
    for phi in r_phi:
        for theta in r_theta:
            load = np.array([np.sin(theta)*np.cos(phi), 
                             np.sin(theta)*np.sin(phi),
                             np.cos(theta)])
            strength = failure_analysis_b(lam, load, dt, failure_criterion)
            failure_load = load*strength.r
            all_res.append((failure_load, strength))

    return all_res


if __name__ == '__main__':
    pass