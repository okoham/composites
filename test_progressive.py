# -*- coding: utf-8 -*-
"""
Created on 28.09.2014
@author: Oliver
"""

from __future__ import division, print_function
import unittest
import numpy as np 
from clt import Laminate, MembraneLaminate
from material import TransverseIsotropicPlyMaterial
import plainstrength


class StrengthEnveloppe3D(unittest.TestCase):
    """Test the plain strength polar function."""
    
    def setUp(self):
        m = TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.127,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        sseq1 = [(a, m) for a in [-30,90,30,-30,30,30,-30 ,30,90,-30]]
        sseq2 = [(a, m) for a in [45,-45,0,90,90,0,-45,45]]
        self.lam1 = MembraneLaminate(sseq1)
        self.lam2 = MembraneLaminate(sseq2)
 
    @unittest.skip('save time')
    def test_lam1_hashinb(self):
        for ncrit, res in plainstrength.strength_enveloppe_3d(self.lam1, 0, 'Airbus'):
            print(ncrit, *res)
            
    @unittest.skip('save time')
    def test_lam2_hashinb(self):
        for ncrit, res in plainstrength.strength_enveloppe_3d(self.lam2, 0, 'Hashin B'):
            print(ncrit, *res)
       
    @unittest.skip('save time')    
    def test_lam2_hashinb_plot(self):
        # FIXME: this is not good yet ...
        from scipy.spatial import ConvexHull
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        r = plainstrength.strength_enveloppe_3d(self.lam2, 0, 'Hashin B')
        loads = np.array([x[0] for x in r])
        
        hull = ConvexHull(loads)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(loads[:,0], loads[:,1], loads[:,2])
        for simplex in hull.simplices:
            ax.plot_trisurf(loads[simplex,1], loads[simplex,2], loads[simplex,0])
        ax.set_xlabel(r'$N_{y,f}$')
        ax.set_ylabel(r'$N_{xy,f}$')
        ax.set_zlabel(r'$N_{x,f}$')      
        ax.set_aspect('equal')      
        plt.show()
        
            
class StrengthEnveloppe2D(unittest.TestCase):
    """Test the plain strength polar function."""
    
    def setUp(self):
        m = TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.127,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        #sseq = [(a, m) for a in [45,-45,0,90,0,90,0,-45,45]]
        sseq = [(a, m) for a in [-30,90,30,-30,30,30,-30 ,30,90,-30]]
        #self.lam = Laminate(sseq)
        self.mlam = MembraneLaminate(sseq)
        #self.load6 = np.array([-100,-50,34,-21,56,63])
        #self.load3 = np.array([100,0,0])
        #self.dtemp = 0    
        
    def test_Ny_Nx_airbus(self):
        for ncrit, res in plainstrength.strength_enveloppe_2d(self.mlam, 0, 'Ny', 'Nx', 'Airbus'):
            print(ncrit, *res)
                    
    def test_Nx_Ny_airbus(self):
        for ncrit, res in plainstrength.strength_enveloppe_2d(self.mlam, 0, 'Nx', 'Ny', 'Airbus'):
            print(ncrit, *res)
            
    def test_Nx_Ny_hashinb(self):
        for ncrit, res in plainstrength.strength_enveloppe_2d(self.mlam, 0, 'Nx', 'Ny', 'Hashin B'):
            print(ncrit, *res)            
            
    def test_Nx_Nxy_airbus(self):
        for ncrit, res in plainstrength.strength_enveloppe_2d(self.mlam, 0, 'Nx', 'Nxy', 'Airbus'):
            print(ncrit, *res)      
            
    def test_Ny_Nxy_airbus(self):
        for ncrit, res in plainstrength.strength_enveloppe_2d(self.mlam, 0, 'Ny', 'Nxy', 'Airbus'):
            print(ncrit, *res)   

class StrengthPolar(unittest.TestCase):
    """Test the plain strength polar function."""
    
    def setUp(self):
        m = TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.127,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        #sseq = [(a, m) for a in [45,-45,0,90,0,90,0,-45,45]]
        sseq = [(a, m) for a in [-30,90,30,-30,30,30,-30 ,30,90,-30]]
        #self.lam = Laminate(sseq)
        self.mlam = MembraneLaminate(sseq)
        #self.load6 = np.array([-100,-50,34,-21,56,63])
        self.load3 = np.array([100,0,0])
        self.dtemp = 0    
        
    def test_tension_airbus(self):
        for theta, res in plainstrength.strength_polar(self.mlam, 0, 'Airbus'):
            print(theta, *res)
            
    def test_compression_airbus(self):
        for theta, res in plainstrength.strength_polar(self.mlam, 0, 'Airbus', load='compression'):
            print(theta, *res)

    def test_tension_hashinb(self):
        for theta, res in plainstrength.strength_polar(self.mlam, 0, 'Hashin B'):
            print(theta, *res)
            
    def test_compression_hashinb(self):
        for theta, res in plainstrength.strength_polar(self.mlam, 0, 'Hashin B', load='compression'):
            print(theta, *res)
            
            
            
class TestFailureB(unittest.TestCase):
    
    
    def setUp(self):
        m = TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.127,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        #sseq = [(a, m) for a in [45,-45,0,90,0,90,0,-45,45]]
        sseq = [(a, m) for a in [-30,90,30,-30,30,30,-30 ,30,90,-30]]
        self.lam = Laminate(sseq)
        self.mlam = MembraneLaminate(sseq)
        
        self.load6 = np.array([-100,-50,34,-21,56,63])
        self.load3 = np.array([100,0,0])
        self.dtemp = -160
    

    def test_flex_hashina(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'Hashin A')
        print(r)

    def test_flex_hashinb(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'Hashin B')
        print(r)
        
    def test_flex_Puck(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'Puck')
        print(r)
                
    def test_flex_modpuck(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'ModPuck')
        print(r)

    def test_flex_maxstress(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'MaxStress')
        print(r)
                
    def test_flex_maxstrain(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'MaxStrain')
        print(r)
                        
    def test_flex_airbus(self):
        r = plainstrength.failure_analysis_b(self.lam, self.load6, self.dtemp, 'Airbus')
        print(r)

    def test_memb_hashina(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'Hashin A')
        print(r)

    def test_memb_hashinb(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'Hashin B')
        print(r)
        
    def test_memb_puck(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'Puck')
        print(r)
        
    def test_memb_modpuck(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'ModPuck')
        print(r)
        
    def test_memb_maxstress(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'MaxStress')
        print(r)
        
    def test_memb_maxstrain(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'MaxStrain')
        print(r)
        
    def test_memb_airbus(self):
        r = plainstrength.failure_analysis_b(self.mlam, self.load3, self.dtemp, 'Airbus')
        print(r)
   

class TestProgressiveFailure(unittest.TestCase):
    
    
    def setUp(self):
        m = TransverseIsotropicPlyMaterial(name='IMA_M21E_MG',
           E11=154000., E22=8500., G12=4200., nu12=0.35, t=0.127,
           a11t=0.15e-6, a22t=28.7e-6,
           F11t=2000., F11c=1000., F22t=50., F22c=200., F12s=100.)
        #sseq = [(a, m) for a in [45,-45,0,90,0,90,0,-45,45]]
        sseq = [(a, m) for a in [-30,90,30,-30,30,30,-30 ,30,90,-30]]
        self.lam = Laminate(sseq)
        self.mlam = MembraneLaminate(sseq)
        
        self.load6 = np.array([-100,-50,34,-21,56,63])
        self.load3 = np.array([-100,0,0])
        self.dtemp = -160
    
    def test_A(self):
        print(self.lam.A())
        
    def test_solution(self):
        sol = self.lam.get_linear_response([100,0,0,0,0,0], self.dtemp)
        print(sol.eps_kappa())
            
    def test_flex_hashina(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'Hashin A')

    def test_flex_hashinb(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'Hashin B')
        
    def test_flex_Puck(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'Puck')
                
    def test_flex_modpuck(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'ModPuck')

    def test_flex_puckC(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'Puck C')

    def test_flex_maxstress(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'MaxStress')
                
    def test_flex_maxstrain(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'MaxStrain')
                        
    def test_flex_airbus(self):
        plainstrength.failure_analysis_c(self.lam, self.load6, self.dtemp, 'Airbus')

    def test_memb_hashina(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'Hashin A')

    def test_memb_hashinb(self):
        r, hist = plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'Hashin B')
        print(r, hist)
        
    def test_memb_puck(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'Puck')
        
    def test_memb_modpuck(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'ModPuck')

    def test_memb_puckC(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'Puck C')        

    def test_memb_maxstress(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'MaxStress')
        
    def test_memb_maxstrain(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'MaxStrain')
        
    def test_memb_airbus(self):
        plainstrength.failure_analysis_c(self.mlam, self.load3, self.dtemp, 'Airbus')
        
        
     
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()