import numpy as np
from scipy.special import eval_legendre, kv, iv, spherical_kn, spherical_in

from utils import *

def MinD(_in, L):
    return _in - np.rint(_in/L)*L

def PImg(_in, L):
    _idx = np.floor(_in/L)
    out = _in - _idx*L
    return out, _idx

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


class Mixin:

    def do_effective_potential(self, fname, Np=100):

        num=Np
        self.effective_potential_radial(Np=num)
        np.savetxt(self.ICidict['folder']+"/"+fname, self.effective_potential, fmt="%.5e")
       
        self.logfile_mf() 

    def effective_potential_radial(self, Np=100):

        self.effective_potential = []
        for r in np.linspace(self.sigma,3.*self.sigma,Np):

            self.effective_potential.extend([r])
            to_orient = self.generate_orientations()

            ##case 1 patch-patch case 2 equator-equator case 3 equator patch
            for _ori in to_orient:              
                VV = self.effpot_funcdict['2patch'](r, 0., [0,1,0], _ori)
                self.effective_potential.extend([VV[0,2]])

        l_to = len(to_orient)+1
        self.effective_potential = np.reshape(np.asarray(self.effective_potential),(int(len(self.effective_potential)/l_to),l_to))

    def effective_potential_plot_angles(self, _myaxis, filename, Np=100):

        ff=open(self.ICidict['folder']+'/'+filename, "w"); ff.close();
        
        to_orient = self.generate_orientations()
        rotaxes = [_myaxis for i in range(len(to_orient))]

        ff = open(filename, "w")
        for th in np.linspace(0,2.*np.pi,100):
            r = self.sigma;  ff.write("%.5e " % (th))
            i=0
            for _ori in to_orient:
                VV = self.effpot_funcdict['2patch'](r, th, rotaxes[i], _ori)   
                ff.write("%.5e " % (VV[0,2])); i+=1
            ff.write("\n")

        ff.close() 


    def effective_potential_2patches(self,r, th, rotaxis, topos):

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge; a0 = 0.5*np.sum(a)
        Zoff = np.sum(Zp) + Zc
        kR = self.kappa*self.sigma_core; ka = self.kappa*a0
        hc_p = np.empty_like(Zp)
        
        if np.prod(Zp) < 0:
            hc_c = 1./( kR**2*spherical_kn(1,kR)*spherical_in(0,ka) )
            hc_p = (a0/self.sigma_core)/( kR**2*spherical_kn(2,kR)*spherical_in(1,ka) )
        else:
            ll = 2
            hc_c = (Zoff/Zc)*np.exp(kR)/((1+kR)) -(np.sum(Zp)/Zc)*((a0/self.sigma_core)**ll)*spherical_in(0,ka)/( (kR**2)*spherical_kn(ll+1,kR)*spherical_in(ll,ka) )
            hc_p = (a0/self.sigma_core)**ll/( kR**2*spherical_kn(ll+1,kR)*spherical_in(ll,ka) )

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
            VV[0,j] += Zc*hc_c*self.analytic_funcdict['2patch'](dd[0], (thij[0]), 0.) 
            VV[0,j] += Zp[0]*hc_p*self.analytic_funcdict['2patch'](dd[1], (thij[1]), 0.) 
            VV[0,j] += Zp[1]*hc_p*self.analytic_funcdict['2patch'](dd[2], (thij[2]), 0.) 
        
        self.topo = np.copy(self.topo_backup)
        ##removed a self.e_charge as prefactor from each term above
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1])

        if not self.real_units:
            VV[0,2] *= self.f_kBT

        return VV
                

#######################################################################################################
    
    def compute_effective_potential(self, _part1, _part2, topos):

        funcname = '2patch'

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        Zoff = np.sum(Zp) + Zc
        a0 = 0.5*np.sum(a); kR = self.kappa*self.sigma_core; ka = self.kappa*a0
        hc_p = np.empty_like(Zp)
       
        if np.prod(Zp) < 0:
            hc_c = Zc/( kR**2*spherical_kn(1,kR)*spherical_in(0,ka) )
            hc_p = Zp*(a0/self.sigma_core)/( kR**2*spherical_kn(2,kR)*spherical_in(1,ka) )
        else:
            ll = 2
            hc_c = (Zoff)*np.exp(kR)/((1+kR)) -(np.sum(Zp))*((a0/self.sigma_core)**ll)*spherical_in(0,ka)/( (kR**2)*spherical_kn(ll+1,kR)*spherical_in(ll,ka) )
            hc_p = Zp*(a0/self.sigma_core)**ll/( kR**2*spherical_kn(ll+1,kR)*spherical_in(ll,ka) )
        
        l_bj_norm = self.bjerrum_real/self.real_size;    ####da ridefinire fuori --- 200 e' il diametro reale in nm
        r = np.linalg.norm(_part1[0,:3]-_part2[0,:3]); 

        VV = np.zeros((1,3))

        for j in [0,1]:

            if j == 0:
                dd, cthij = calc_dists_angles_2p(_part1, _part2, r, a, self.box)
            else:
                dd, cthij = calc_dists_angles_2p(_part2, _part1, r, a, self.box)

            self.topo = np.copy(topos[j]);

            ##self.calc_patch_angles()
            VV[0,j] = 0
            temp = (hc_c)*self.analytic_funcdict[funcname](dd[0], cthij[0], 0.)
            VV[0,j] += temp;  
            for i in range(self.npatch):
                temp = (hc_p[i])*self.analytic_funcdict[funcname](dd[i+1], cthij[i+1], 0.)
                VV[0,j] += temp;

        self.restore_patches()
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1]) 

        if not self.real_units:
            VV[0,2] *= self.f_kBT

        return VV

    def logfile_mf(self):

        funcname = '2patch'
        PI = np.pi
        outfname = self.ICidict['folder']+"/logfile.dat"
        f = open(outfname,'w')

        f.write("**************\nEffective Pair Potential\n****************\n\n")

        f.write("Radius colloid (nm) %.8f\t" % (self.real_size/2*1E9) )
        f.write("Charge asymmmetry (percentage) %s\n" % (" ".join(str(x) for x in self.ecc)) )
        f.write("Screening %.5f\t Debye length (nm) %.5f\n" % (self.emme, (1./self.kappa)*self.real_size*1E9) )

        temp = self.colloid_charge*self.e_charge/(PI*self.real_size**2)
        f.write("Zc %.5f\t Surface charge of the colloid %.8e (C/m2)\t %.8e (qe/nm2)\n" % (-1*self.colloid_charge, temp, temp/0.16) )

        f.write("Zc*LambdaB/radius colloid (non-dim) %.5f\n\n\n" % (-2*self.colloid_charge*self.bjerrum_real/self.real_size) )
        f.write("Zp %s\t" % (" ".join(str(x) for x in self.patch_charge)) )
        
        if self.sp_zero == -1:
            self.compute_potential_zero()
        
        if self.sp_zero < 180:
            f.write("Patch size (degrees)  %.8f\n" % (self.sp_zero))
        else:
            f.write("no root for sp potential, no patch size\n")

        if len(self.effective_potential) == 0:
            self.effective_potential_radial(Np=1)

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


