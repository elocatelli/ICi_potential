import numpy as np
from scipy.special import eval_legendre

from utils import *

def yukawa(r, kappa):
    return np.exp(-kappa*r)/r

def calc_dists_angles_2p(part1, part2, r, a):
    n1 = (part1[0] - part1[1:])/(a[:])
    n2 = (part2[0] - part2[1:])/(a[:])
    r0 = (part1[0] - part2[0])/r
	
    dp = np.linalg.norm(part1[0]-part2[1:], axis=1)

    r1 = (part1[0]-part2[1:])/(dp[:].reshape(dp.size,1))

    cth0 = np.dot(n1[0],r0); cth1 = np.dot(n1[0],r1[0]); cth2 = np.dot(n1[0],r1[1])

    if np.fabs(cth0) > 1.:
        temp = 1.*np.fabs(cth0)/cth0; cth0 = temp
    if np.fabs(cth1) > 1.:
        temp = 1.*np.fabs(cth1)/cth1; cth1 = temp
    if np.fabs(cth2) > 1.:
        temp = 1.*np.fabs(cth2)/cth2; cth2 = temp

    dd = [r, dp[0], dp[1]]; th_s = np.arccos([cth0, cth1, cth2])
    
    return dd, th_s  

def calc_dists_angles(part1, part2, r, a):
    
    r0 = (part2[0]- part1[0])
    ##dp = np.linalg.norm(part1[0]-part2[1:], axis=1)
    r1 = (part2[1:]-part1[0]) ##/(dp[:].reshape(dp.size,1))
    
    c0 = to_spherical(r0)
    dd = [c0[0]]; th_s = [c0[1]]; ph_s = [c0[2]] 
         
    for i in range(r1.shape[0]):
        temp = to_spherical(r1[i])
        dd.extend([temp[0]]); th_s.extend([temp[1]]); ph_s.extend([temp[2]])
    
    return dd, th_s, ph_s

class Mixin:

    def effective_potential_store(self, funcname, funcname_1P, Np=100):

        self.effective_potential = []
        for r in np.linspace(self.sigma,1.5*self.sigma,Np):

            self.effective_potential.extend([r])
            to_orient = self.generate_orientations()

            ##case 1 patch-patch case 2 equator-equator case 3 equator patch
            for _ori in to_orient:              
                VV = self.effpot_funcdict[funcname](r, 0., [0,1,0], _ori, funcname_1P)
                self.effective_potential.extend([VV[0,2]])

        l_to = len(to_orient)+1
        self.effective_potential = np.reshape(np.asarray(self.effective_potential),(int(len(self.effective_potential)/l_to),l_to))

    def effective_potential_plot_angles(self, funcname, funcname_1P, _myaxis, filename):

        ff=open(filename, "w"); ff.close();
        
        to_orient = self.generate_orientations()
        rotaxes = [_myaxis for i in range(len(to_orient))]

        ff = open(filename, "w")
        for th in np.linspace(0,2.*np.pi,100):
            r = self.sigma;  ff.write("%.5e " % (th))
            i=0
            for _ori in to_orient:
                VV = self.effpot_funcdict[funcname](r, th, rotaxes[i], _ori, funcname_1P)   
                ff.write("%.5e " % (VV[0,2])); i+=1
            ff.write("\n")

        ff.close()


    def effective_potential_2patches(self,r, th, rotaxis, topos, funcname):

        #funcname = '2patch_analytic' ##'yukawa'
        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        #l_bj_norm = self.bjerrum/self.real_size     ####da ridefinire fuori --- 200 e' il diametro reale in nm

        _part1_0 = self.generate_particle(topos[0])
        _part2_0 = self.generate_particle(topos[1])

        VV = np.zeros((1,3))
        
        for j in [0,1]:

            _part1 = np.copy(_part1_0[:,:3]) 
            if th == 0:
                _part2_1 = np.copy(_part2_0[:,:3])
            else:
                _part2_1 = rotate_part(_part2_0[:,:3], rotaxis, th)
            _part2 = move_part(_part2_1, [r,0,0])
            
            if j == 0:
                dd, thij = calc_dists_angles_2p(_part1, _part2, r, a)
            else:
                dd, thij = calc_dists_angles_2p(_part2, _part1, r, a)

            VV[0,j] = 0
            VV[0,j] += (Zc)*self.analytic_funcdict[funcname](dd[0], (thij[0]), 0.) 
            VV[0,j] += (Zp[0])*self.analytic_funcdict[funcname](dd[1], (thij[1]), 0. ) 
            VV[0,j] += (Zp[1])*self.analytic_funcdict[funcname](dd[2], (thij[2]), 0.) 

        ##removed a self.e_charge as prefactor from each term above
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1])*hc_c
        return VV
                
    def effective_potentials_yukawa(self, r, pvec, _id, funcname):

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        l_bj_norm = 1. ##self.bjerrum/self.real_size     

        _part1_0 = self.generate_particle(pvec[:3])
        _part2_0 = self.generate_particle(pvec[3:])

        VV = np.zeros((1,3))

        for j in [0,1]:
            _part1 = np.copy(_part1_0[:,:3]) #rotate_part(_part1_0[:,:3], [0,1,0], th[2*j])
            #_part2_1 = rotate_part(_part2_0[:,:3], [0,1,0], th[2*j+1])
            _part2 = move_part(_part2_0[:,:3], [r,0,0])

            if j == 0:
                dd, cthij = calc_dists_angles_2p(_part1, _part2, r, a)
            else:
                dd, cthij = calc_dists_angles_2p(_part2, _part1, r, a)

            #cthij = np.asarray(angles[_id][j])            
            
            V0 = Zc**2; V00 = 0.
            V1 = Zc*Zp[0]; V01 = 0.
            V2 = Zc*Zp[1]; V02 = 0.
            for l in range(0,self.lmax,2):
                V00 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[0]))
                V01 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[1]))
                V02 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[2]))

            V0 += V00*2.*Zc*Zp[0]; V0 *= yukawa(dd[0], self.kappa)*hc_c**2 #*l_bj_norm
            V1 += V01*2.*Zp[0]**2; V1 *= yukawa(dd[1], self.kappa)*hc_c**2 #*l_bj_norm
            V2 += V02*2.*Zp[1]**2; V2 *= yukawa(dd[2], self.kappa)*hc_c**2 #*l_bj_norm

            VV[0,j] = (V0+V1+V2)/self.eps

        VV[0,2] = 0.5*(VV[0,0]+VV[0,1])

        return VV

#######################################################################################################

    def effective_potential(self, r, th, rotaxis, topos, funcname):

        funcname == 'numeric'   ##use the general formula

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        #l_bj_norm = self.bjerrum/self.real_size     ####da ridefinire fuori --- 200 e' il diametro reale in nm

        _part1_0 = self.generate_particle(topos[0])
        _part2_0 = self.generate_particle(topos[1])

        VV = np.zeros((1,3))
        
        for j in [0,1]:

            _part1 = np.copy(_part1_0[:,:3]) 
            if th == 0:
                _part2_1 = np.copy(_part2_0[:,:3])
            else:
                _part2_1 = rotate_part(_part2_0[:,:3], rotaxis, th)
            _part2 = move_part(_part2_1, [r,0,0])
           
            if j == 0:
                dd, thij, phij = calc_dists_angles(_part1, _part2, r, a)
            else:
                dd, thij, phij = calc_dists_angles(_part2, _part1, r, a)

            self.topo = np.copy(topos[j]);  
            if j == 1:
                self.topo = self.topo_rotate(rotaxis, th)
            self.calc_patch_angles()

            VV[0,j] = 0
            
            VV[0,j] += (Zc)*self.analytic_funcdict[funcname](dd[0], (thij[0]),phij[0])
            print("core ", dd[0], (thij[0]),phij[0], self.analytic_funcdict[funcname](dd[0], (thij[0]),phij[0]), (Zc)*self.analytic_funcdict[funcname](dd[0], (thij[0]),phij[0])*hc_c )
            for i in range(self.npatch):
                VV[0,j] += (Zp[i])*self.analytic_funcdict[funcname](dd[i+1], (thij[i+1]), phij[i+1]) 
                print("patch ", self.sph_theta[i], self.sph_phi[i], dd[i+1], (thij[i+1]),phij[i+1], self.analytic_funcdict[funcname](dd[i+1], (thij[i+1]),phij[i+1]), (Zp[i])*self.analytic_funcdict[funcname](dd[i+1], (thij[i+1]), phij[i+1])*hc_c )
            
        self.restore_patches()        
        print("effective_pot ", VV[0,0]*hc_c, VV[0,1]*hc_c)
        ##removed a self.e_charge as prefactor from each term above
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1])*hc_c
        
        return VV



