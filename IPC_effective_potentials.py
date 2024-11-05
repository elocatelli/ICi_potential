import numpy as np
from scipy.special import eval_legendre, kv, iv, spherical_kn, spherical_in

from utils import *

def MinD(_in, L):
    return _in - np.rint(_in/L)*L

def PImg(_in, L):
    _idx = np.floor(_in/L)
    out = _in - _idx*L
    return out, _idx

def yukawa(r, kappa):
    return np.exp(-kappa*r)/r

def spherical_kn(n,z):
    return np.sqrt(2/(np.pi*z))*kv(n+0.5,z)

def spherical_in(n,z):
    return np.sqrt(np.pi/(2*z))*iv(n+0.5,z)

def calc_dists_angles_2p(part1, part2, r, a, box):
    n1 = MinD(part1[0] - part1[1:],box)/(a[:])
    n2 = MinD(part2[0] - part2[1:],box)/(a[:])
    r0 = MinD(part1[0] - part2[0],box)/r
	
    dp = np.linalg.norm(MinD(part1[0]-part2[1:],box), axis=1)

    r1 = MinD(part1[0]-part2[1:],box)/(dp[:].reshape(dp.size,1))

    cth0 = np.dot(n1[0],r0); cth1 = np.dot(n1[0],r1[0]); cth2 = np.dot(n1[0],r1[1])

    if np.fabs(cth0) > 1.:
        temp = 1.*np.fabs(cth0)/cth0; cth0 = temp
    if np.fabs(cth1) > 1.:
        temp = 1.*np.fabs(cth1)/cth1; cth1 = temp
    if np.fabs(cth2) > 1.:
        temp = 1.*np.fabs(cth2)/cth2; cth2 = temp

    dd = [r, dp[0], dp[1]]; th_s = np.arccos([cth0, cth1, cth2])
    
    return dd, th_s  

def calc_dists_angles(part1, part2, r, a, box):
    
    r0 = MinD(part2[0]- part1[0], box)
    ##dp = np.linalg.norm(part1[0]-part2[1:], axis=1)
    r1 = MinD(part2[1:]-part1[0], box) ##/(dp[:].reshape(dp.size,1))
    
    c0 = to_spherical(r0)
    dd = [c0[0]]; th_s = [c0[1]]; ph_s = [c0[2]] 
         
    for i in range(r1.shape[0]):
        temp = to_spherical(r1[i])
        dd.extend([temp[0]]); th_s.extend([temp[1]]); ph_s.extend([temp[2]])
    
    return dd, th_s, ph_s

class Mixin:

    def effective_Zc(self):
        kappa = self.kappa; sigma = self.sigma_core; Zc = self.colloid_charge;
        return Zc*np.exp(kappa*sigma)/(1.+kappa*sigma)

    def effective_Zp(self, r, th, phi, idp):
        
        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        kappa = self.kappa; sigma = self.sigma_core

        out0 = np.exp(kappa*sigma)/(1.+kappa*sigma)
        out1 = 0.

        for l in range(2,self.lmax,2):
            out1 = (2*l+1)*( kv(l+0.5,kappa*r)*np.exp(kappa*r)*np.sqrt(kappa*r)/((kappa*sigma)**1.5*kv(l+1.5,kappa*sigma)) )*( np.power(a[idp]/sigma,l)*eval_legendre(l,np.cos(th)) )
            out0 += out1

        out0 *= Zp[idp]
        #if self.real_units:
        #    out0 *= self.e_charge ##*self.permittivity/(self.real_size))

        return out0

    def do_effective_potential(self, funcname, funcname_1P, Np=100):

        num=Np
        self.effective_potential_store(funcname, funcname_1P, Np=num)
        np.savetxt(self.IPCdict['folder']+"/radial_effective_potential_"+funcname+".dat", self.effective_potential, fmt="%.5e")
        
        #self.effective_potential_plot_angles(funcname, funcname_1P, [1,0,0], myIPC.IPCdict['folder']+'/rotation_x_effective_potential.dat', Np=num)
        #self.effective_potential_plot_angles(funcname, funcname_1P, [0,1,0], myIPC.IPCdict['folder']+'/rotation_y_effective_potential.dat', Np=num)
        #self.effective_potential_plot_angles(funcname, funcname_1P, [0,0,1], myIPC.IPCdict['folder']+'/rotation_z_effective_potential.dat', Np=num)

        self.logfile_mf(funcname) 

    def effective_potential_store(self, funcname, funcname_1P, Np=100):

        self.effective_potential = []
        for r in np.linspace(self.sigma,3.*self.sigma,Np):

            self.effective_potential.extend([r])
            to_orient = self.generate_orientations()

            ##case 1 patch-patch case 2 equator-equator case 3 equator patch
            for _ori in to_orient:              
                VV = self.effpot_funcdict[funcname](r, 0., [0,1,0], _ori, funcname_1P)
                self.effective_potential.extend([VV[0,2]])

        l_to = len(to_orient)+1
        self.effective_potential = np.reshape(np.asarray(self.effective_potential),(int(len(self.effective_potential)/l_to),l_to))

    def effective_potential_plot_angles(self, funcname, funcname_1P, _myaxis, filename, Np=100):

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

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        hc_p = hc_c*self.kappa*a/(np.sinh(self.kappa*a))

        _part1_0 = self.generate_particle(topos[0])
        _part2_0 = self.generate_particle(topos[1])

        VV = np.zeros((1,3))
        
        for j in [0,1]:

            _part1 = np.copy(_part1_0[:,:3]) 
            if th == 0:
                _part2_1 = np.copy(_part2_0[:,:3])
            else:
                _part2_1 = rotate_patches(_part2_0[:,:3], rotaxis, th)
            _part2 = move_part(_part2_1, [r,0,0])
            
            if j == 0:
                dd, thij = calc_dists_angles_2p(_part1, _part2, r, a, self.box)
            else:
                dd, thij = calc_dists_angles_2p(_part2, _part1, r, a, self.box)
            
            VV[0,j] = 0
            VV[0,j] += (Zc)*hc_c*self.analytic_funcdict[funcname](dd[0], (thij[0]), 0.) 
            VV[0,j] += (Zp[0])*hc_p[0]*self.analytic_funcdict[funcname](dd[1], (thij[1]), 0. ) 
            VV[0,j] += (Zp[1])*hc_p[1]*self.analytic_funcdict[funcname](dd[2], (thij[2]), 0.) 

        ##removed a self.e_charge as prefactor from each term above
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1])

        if not self.real_units:
            VV[0,2] *= self.f_kBT

        return VV
                
    def effective_potentials_yukawa(self, r, th, rotaxis, topos, funcname):

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        #hc_p = np.exp(self.kappa*(self.sigma_core))/(1.+self.kappa*(self.sigma_core))
        l_bj_norm = self.bjerrum_real/self.real_size
        
        _part1_0 = self.generate_particle(topos[0])
        _part2_0 = self.generate_particle(topos[1])

        VV = np.zeros((1,3))

        for j in [0,1]:
            _part1 = np.copy(_part1_0[:,:3])
            if th == 0:
                _part2_1 = np.copy(_part2_0[:,:3])
            else:
                _part2_1 = rotate_patches(_part2_0[:,:3], rotaxis, th)
            _part2 = move_part(_part2_1, [r,0,0])

            if j == 0:
                dd, cthij = calc_dists_angles_2p(_part1, _part2, r, a, self.box)
            else:
                dd, cthij = calc_dists_angles_2p(_part2, _part1, r, a, self.box)
            
            self.topo = np.copy(topos[j]);
            if j == 1:
                self.topo = self.topo_rotate(rotaxis, th)
            self.calc_patch_angles()

            #cthij = np.asarray(angles[_id][j])     
            
            V0 = Zc**2; V00 = 0.;
            V1 = Zc*Zp[0]; V01 = 0.
            V2 = Zc*Zp[1]; V02 = 0.
            for l in range(0,self.lmax,2):
                V00 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[0]))
                V01 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[1]))
                V02 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[2]))

            V0 += V00*2.*Zc*Zp[0]; V0 *= yukawa(dd[0], self.kappa)*hc_c**2*l_bj_norm
            V1 += V01*2.*Zp[0]**2; V1 *= yukawa(dd[1], self.kappa)*hc_c**2*l_bj_norm
            V2 += V02*2.*Zp[1]**2; V2 *= yukawa(dd[2], self.kappa)*hc_c**2*l_bj_norm

            VV[0,j] = (V0+V1+V2) ##/self.eps 

        self.restore_patches()
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1])*(300*1.380649E-23/1.602E-19)
        
        return VV

#######################################################################################################

    def effective_potential(self, r, th, rotaxis, topos, funcname):

        funcname == 'numeric'   ##use the general formula

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge; a0 = 0.5*np.sum(a)
        Zoff = np.sum(Zp) + Zc
        kR = self.kappa*self.sigma_core; ka = self.kappa*a0
        #hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        #hc_p = hc_c*self.kappa*a/(np.sinh(self.kappa*a))
        
        if Zc == 0:
            hc_c = 0
            hc_p = (a0/self.sigma_core)/( kR**2*spherical_kn(2,kR)*spherical_in(1,ka) )
        else:
            hc_c = (Zoff/Zc)*np.exp(kR)/((1+kR)) -(np.sum(Zp)/Zc)*((a0/self.sigma_core)**2)*spherical_in(0,ka)/( (kR**2)*spherical_kn(3,kR)*spherical_in(2,ka) )
            hc_p = (a0/self.sigma_core)**2/( kR**2*spherical_kn(3,kR)*spherical_in(2,ka) )

        _part1_0 = self.generate_particle(topos[0])
        _part2_0 = self.generate_particle(topos[1])

        VV = np.zeros((1,3))
        
        for j in [0,1]:

            _part1 = np.copy(_part1_0[:,:3]) 
            if th == 0:
                _part2_1 = np.copy(_part2_0[:,:3])
            else:
                _part2_1 = rotate_patches(_part2_0[:,:3], rotaxis, th)
            _part2 = move_part(_part2_1, [r,0,0])
           
            if j == 0:
                dd, thij, phij = calc_dists_angles(_part1, _part2, r, a, self.box)
            else:
                dd, thij, phij = calc_dists_angles(_part2, _part1, r, a, self.box)

            self.topo = np.copy(topos[j]);  
            if j == 1:
                self.topo = self.topo_rotate(rotaxis, th)
            self.calc_patch_angles()

            VV[0,j] = 0
            
            VV[0,j] += Zc*hc_c*self.analytic_funcdict[funcname](dd[0], (thij[0]),phij[0])
            for i in range(self.npatch):
                #hc_p = self.effective_Zp(dd[i+1], (thij[i+1]), phij[i+1],i)
                VV[0,j] += (Zp[i]*hc_p)*self.analytic_funcdict[funcname](dd[i+1], (thij[i+1]), phij[i+1]) 
            
        self.restore_patches()        
        ##removed a self.e_charge as prefactor from each term above
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1]) 

        if not self.real_units:
            VV[0,2] *= self.f_kBT
        
        return VV


    def logfile_mf(self, funcname):

        PI = np.pi
        outfname = self.IPCdict['folder']+"/logfile.dat"
        f = open(outfname,'w')

        f.write("**************\nEffective Pair Potential\n****************\n\n")

        f.write("Radius colloid (nm) %.8f\t" % (self.real_size/2*1E9) )
        f.write("Charge asymmmetry (percentage) %s\n" % (" ".join(str(x) for x in self.ecc)) )
        f.write("Screening %.5f\t Debye length (nm) %.5f\t Interaction range (percentage) %.5f \n" % (self.emme, (1./self.kappa)*self.real_size*1E9, self.delta) )
        
        temp = self.colloid_charge*self.e_charge/(PI*self.real_size**2)
        f.write("Zc %.5f\t Surface charge of the colloid %.8e (C/m2)\t %.8e (qe/nm2)\n" % (-1*self.colloid_charge, temp, temp/0.16) )

        f.write("Zc*LambdaB/radius colloid (non-dim) %.5f\n\n\n" % (-2*self.colloid_charge*self.bjerrum_real/self.real_size) )
        f.write("Zp %s\t" % (" ".join(str(x) for x in self.patch_charge)) ) 
        f.write("Patch size (degrees)  %.8f\n" % (self.sp_zero))

        if len(self.effective_potential) == 0:
            if funcname == 'general':
                self.effective_potential_store('general', 'numeric', Np=1)
            elif funcname == 'yukawa':
                self.effective_potential_store('yukawa', 'yukawa', Np=1)
            else:
                print("keyword ", funcname, "not allowed\n"); exit(1)
        
        cfT=self.e_charge/(self.kB*self.temperature)
        if not self.real_units: 
            f.write("WARNING: output will be in units of kBT\n")
            cf = self.e_charge*self.permittivity/self.real_size
            f.write("conversion factors %.5e %.5e\n" % (cf, cfT) )
            
            f.write("EQUATORIAL-EQUATORIAL\nENERGY (eV) %.8f\t ENERGY (KbT) %.8f\n" % (self.effective_potential[0,2]/cfT, self.effective_potential[0,2]) )
            f.write("POLAR-POLAR\nENERGY (eV) %.8f\t ENERGY (KbT) %.8f\n" % (self.effective_potential[0,1]/cfT, self.effective_potential[0,1]) )
            f.write("EQUATORIAL-POLAR\nENERGY (eV) %.8f\t ENERGY (KbT) %.8f\n" % (self.effective_potential[0,3]/cfT, self.effective_potential[0,3]) )
        else:
            f.write("EQUATORIAL-EQUATORIAL\nENERGY (eV) %.8f\t ENERGY (KbT) %.8f\n" % (self.effective_potential[0,2], self.effective_potential[0,2]*cfT))         
            f.write("POLAR-POLAR\nENERGY (eV) %.8f\t ENERGY (KbT) %.8f\n" % (self.effective_potential[0,1], self.effective_potential[0,1]*cfT ))     
            f.write("EQUATORIAL-POLAR\nENERGY (eV) %.8f\t ENERGY (KbT) %.8f\n" % (self.effective_potential[0,3], self.effective_potential[0,3]*cfT ) )     
             

        f.close()

