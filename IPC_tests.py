import numpy as np
from scipy.special import eval_legendre, kv, iv
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from utils import *

from IPC_coarse_graining import compute_dum_energy, compute_dum_energy_CG1
from IPC_effective_potentials import calc_dists_angles, calc_dists_angles_2p, yukawa, MinD, PImg

def spherical_kn(n,z):
    return np.sqrt(2/(np.pi*z))*kv(n+0.5,z)

def spherical_in(n,z):
    return np.sqrt(np.pi/(2*z))*iv(n+0.5,z)


def read_coordinates(_fname, ecc, patch_sigma, box):

    f = open(_fname, 'r')
    npatch = 1 
    i = 0
    coords = []
    for line in f:
        elems = line.strip("\n").split(" ")
        if i == 0:
            coords.extend(elems); coords.extend(["0.5"])
        elif i == 1:
            npatch = int(elems[0])
        else:
            coords.extend(elems); coords.extend(patch_sigma[i-2])
        
        i+= 1
        if i == npatch+2:
            i = 0

    f.close()
    coords = np.reshape(np.asarray(coords), (len(coords)//4,4)).astype(float)
    temp, trash = PImg(coords[::3,:3] + ecc[0]*coords[1::3,:3], box); coords[1::3,:3] = temp
    temp, trash = PImg(coords[::3,:3] + ecc[1]*coords[2::3,:3], box); coords[2::3,:3] = temp

    return coords

def get_topo(_input, box):

    out = []
    temp = MinD(_input[1] - _input[0], box)
    out.extend([temp/np.linalg.norm(temp)])
    if _input.shape[0] == 3:
        temp = MinD(_input[2] - _input[0], box)
        out.extend([temp/np.linalg.norm(temp)])

    return out


class Mixin:

    def compute_effective_potential(self, _part1, _part2, topos, funcname):

        ##funcname == 'numeric'   ##use the general formula

        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        Zoff = np.sum(Zp) + Zc
        a0 = 0.5*np.sum(a); kR = self.kappa*self.sigma_core; ka = self.kappa*a0
        #hc0 = (a0/self.sigma_core)**2/( kR*iv(2.5,ka)*kv(3.5,kR) )
        #hc_c = np.exp(kR)*(Zoff/(Zc*(1+kR)) - (np.sum(Zp)/Zc)*iv(0.5,ka)*kv(0.5,kR)*hc0)
        #hc_p = (a0/self.sigma_core)**0.5*hc0
        hc_p = np.empty_like(Zp)
       
        if np.prod(Zp) < 0:
            hc_c = Zc/( kR**2*spherical_kn(1,kR)*spherical_in(0,ka) )
            hc_p = Zp*(a/self.sigma_core)/( kR**2*spherical_kn(2,kR)*spherical_in(1,ka) )
        else:
            ll = 2
            hc_c = (Zoff)*np.exp(kR)/((1+kR)) -(np.sum(Zp))*((a0/self.sigma_core)**ll)*spherical_in(0,ka)/( (kR**2)*spherical_kn(ll+1,kR)*spherical_in(ll,ka) )
            hc_p = Zp*(a/self.sigma_core)**ll/( kR**2*spherical_kn(ll+1,kR)*spherical_in(ll,ka) )
        
        #hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        #hc_p = hc_c*(self.kappa*a/(np.sinh(self.kappa*a))) 
        
        l_bj_norm = self.bjerrum_real/self.real_size;    ####da ridefinire fuori --- 200 e' il diametro reale in nm
        r = np.linalg.norm(_part1[0,:3]-_part2[0,:3]); 

        VV = np.zeros((1,3))

        for j in [0,1]:

            if funcname == 'yukawa':
                if j == 0:
                    dd, cthij = calc_dists_angles_2p(_part1, _part2, r, a, self.box)
                else:
                    dd, cthij = calc_dists_angles_2p(_part2, _part1, r, a, self.box)
            else:
                if j == 0:
                    dd, thij, phij = calc_dists_angles(_part1, _part2, r, a, self.box)
                else:
                    dd, thij, phij = calc_dists_angles(_part2, _part1, r, a, self.box)

            self.topo = np.copy(topos[j]);

            if funcname == 'yukawa':
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

                VV[0,j] = (V0+V1+V2)*(300*1.380649E-23/1.602E-19) ##/self.eps

            else:
                self.calc_patch_angles()
                VV[0,j] = 0
                temp = (hc_c)*self.analytic_funcdict[funcname](dd[0], thij[0],phij[0])
                VV[0,j] += temp;  
                for i in range(self.npatch):
                    ##hc_p = self.effective_Zp(dd[i+1], (thij[i+1]), phij[i+1],i)
                    temp = (hc_p[i])*self.analytic_funcdict[funcname](dd[i+1], thij[i+1], phij[i+1])
                    VV[0,j] += temp;

        self.restore_patches()
        VV[0,2] = 0.5*(VV[0,0]+VV[0,1]) 

        if not self.real_units:
            VV[0,2] *= self.f_kBT

        return VV


    def compute_energy(self, _coords, funcname, mode):

        N = _coords.shape[0]//3
        a = self.ecc; Zc = self.colloid_charge; Zp = self.patch_charge
        hc_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
        l_bj_norm = self.bjerrum_real/self.real_size;

        Etot = 0
        for k in range(N-1):
            _part1 = _coords[3*k:3*(k+1)][:,:3]
            for w in range(k+1,N):
                _part2 = _coords[3*w:3*(w+1)][:,:3]
                
                r = np.linalg.norm(MinD(_part1[0]-_part2[0], self.box))
                
                if r < 1.:
                    print("error: overlap")
                if mode == 'ipc':
                    VV = np.zeros((1,3))
                
                    for j in [0,1]:
                        if funcname == 'yukawa':
                            if j == 0:
                                dd, cthij = calc_dists_angles_2p(_part1, _part2, r, a, self.box)
                            else:
                                dd, cthij = calc_dists_angles_2p(_part2, _part1, r, a, self.box)
                        else:
                            if j == 0:
                                dd, thij, phij = calc_dists_angles(_part1, _part2, r, a, self.box)
                                self.topo = get_topo(_part1,self.box)
                            else:
                                dd, thij, phij = calc_dists_angles(_part2, _part1, r, a, self.box)
                                self.topo = get_topo(_part2, self.box)
                    
                        self.calc_patch_angles()

                        if funcname == 'yukawa':
                            V0 = Zc**2; V00 = 0.;
                            V1 = Zc*Zp[0]; V01 = 0.
                            V2 = Zc*Zp[1]; V02 = 0.
                            for l in range(0,self.lmax,2):
                                 V00 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[0]))
                                 V01 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[1]))
                                 V02 += (2*l+1)*np.power(a[0]/self.sigma_core,l)*eval_legendre(l,np.cos(cthij[2]))

                            V0 += V00*2.*Zc*Zp[0]; V0 *= yukawa(dd[0], self.kappa)*hc_c#*#l_bj_norm
                            V1 += V01*2.*Zp[0]**2; V1 *= yukawa(dd[1], self.kappa)*hc_c#*#l_bj_norm
                            V2 += V02*2.*Zp[1]**2; V2 *= yukawa(dd[2], self.kappa)*hc_c#* #l_bj_norm

                            VV[0,j] = (V0+V1+V2)/self.eps2

                        else:
                            VV[0,j] = 0
                            VV[0,j] += (Zc)*self.analytic_funcdict[funcname](dd[0], (thij[0]),phij[0])
                            for i in range(self.npatch):
                                 VV[0,j] += (Zp[i])*self.analytic_funcdict[funcname](dd[i+1], (thij[i+1]), phij[i+1])
 
                    VV[0,2] = 0.5*(VV[0,0]+VV[0,1])#*hc_c
                    
                    Etot += VV[0,2]
                else:
                    sigma_core2 = self.sigma_core + self.delta/2.
                    if self.cgtype == 'cg1':
                        EE = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
                    else:
                        p = [self.sigma, self.delta]
                        for v in self.ecc:
                            p.extend([v])
                        EE = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)

                    Etot += np.dot(np.asarray(EE),self.u_cg)

        
        return Etot

    def rotate_colloid_around_axis(self, _myaxis, fname_mf, fname_cg):
        
        to_orient = self.generate_orientations()
        rotaxes = [_myaxis for i in range(len(to_orient))]

        ff = open(fname_mf, "w"); gg = open(fname_cg, "w")
        for th in np.linspace(0,2.*np.pi,100):
            ff.write("%.5e " % (th)); gg.write("%.5e " % (th));
            for _ori in to_orient:
                _part1 = self.generate_particle(_ori[0])[:,:3]
                _part2_0 = self.generate_particle(_ori[1])[:,:3]; _part2_0[:,:3] += np.asarray([1,0,0])
                _part2 = rotate_part(_part2_0, rotaxes[i], th)
                VV = self.compute_effective_potential(_part1, _part2, _ori, 'numeric')
                ff.write("%.5e " % (VV[0,2])); 
                ########################################################
                sigma_core2 = self.sigma_core + self.delta/2.
                if self.cgtype == 'cg1':
                    EE = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
                else:
                    p = [self.sigma, self.delta]
                    for v in self.ecc:
                        p.extend([v])
                    EE = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
                gg.write("%.5e " % (np.dot(np.asarray(EE),self.u_cg)) )

            ff.write("\n"); gg.write("\n")

        ff.close(); gg.close()

    def rotation_pathway(self, ipc_type, fname_mf, fname_cg):

        tempsave = self.max_eff_pot; self.max_eff_pot=1.;  
        _displ = self.path_displ;
        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]
        rotaxis = np.asarray([0,1,0])
        sigma_core2 = self.sigma_core + self.delta/2.

        ff = open(fname_mf, "w"); gg = open(fname_cg, "w")
        ## rotation 1: from EP to EE
        for th in np.linspace(0,np.pi/2.,self.path_N):
            ff.write("%.5e " % (th)); gg.write("%.5e " % (th));
            _part1 = self.generate_particle(EP[0])[:,:3]
            _part2_0 = self.generate_particle(EP[1])[:,:3]
            _part2 = rotate_patches(_part2_0, rotaxis, th); _part2[:,:3] += _displ*np.asarray([1,0,0])
            VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot));
            ########################################################
            if self.cgtype == 'cg1':
                CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
            gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )

        #ff.write("\n"); gg.write("\n")
        thend = np.pi/2.

        ## rotation 2: from EE to PP rotating colloid around y
        for th in np.linspace(0,np.pi/2.,self.path_N):
            ff.write("%.5e " % (th+thend)); gg.write("%.5e " % (th+thend));
            _part1 = self.generate_particle(EE[0])[:,:3]
            _part2_0 = self.generate_particle(EE[1])[:,:3]; _part2_0[:,:3] += _displ*np.asarray([1,0,0])
            _part2 = rotate_part(_part2_0, rotaxis, th)
            VV = self.compute_effective_potential(_part1, _part2, EE, ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot));
            ########################################################
            if self.cgtype == 'cg1':
                CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)

            gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )
       
        #ff.write("\n"); gg.write("\n")
        thend += np.pi/2.
        ## rotation 3: from PP to EP rotating patches around y (EE+shift z)
        for th in np.linspace(0,np.pi/2.,self.path_N):
            ff.write("%.5e " % (th+thend)); gg.write("%.5e " % (th+thend));
            _part1 = self.generate_particle(EE[0])[:,:3]
            _part2_0 = self.generate_particle(EE[1])[:,:3]; 
            _part2 = rotate_patches(_part2_0, rotaxis, th); _part2[:,:3] += _displ*np.asarray([0,0,1])
            VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot));
            ########################################################
            if self.cgtype == 'cg1':
                CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
            gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )

        #ff.write("\n"); gg.write("\n")
        thend += np.pi/2.
        ## rotation 4: from EP to EP rotating colloid around y (EP+shift z)
        for th in np.linspace(0,np.pi/2.,self.path_N):
            ff.write("%.5e " % (th+thend)); gg.write("%.5e " % (th+thend));
            _part1 = self.generate_particle(EP[0])[:,:3]
            _part2_0 = self.generate_particle(EP[1])[:,:3]; _part2_0[:,:3] += _displ*np.asarray([0,0,1])
            _part2 = rotate_part(_part2_0, rotaxis, -th)
            VV = self.compute_effective_potential(_part1, _part2, EP, ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot));
            ########################################################
            if self.cgtype == 'cg1':
                CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)

            gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )

        #ff.write("\n"); gg.write("\n")
        thend += np.pi/2.
        rotaxis = np.asarray([0,0,1])
        ## rotation 5: from EP to EP rotating colloid around z (EP+shift x)
        for th in np.linspace(0,np.pi/2.,self.path_N):
            ff.write("%.5e " % (th+thend)); gg.write("%.5e " % (th+thend));
            _part1 = self.generate_particle(EP[0])[:,:3]
            _part2_0 = self.generate_particle(EP[1])[:,:3]; _part2_0[:,:3] += _displ*np.asarray([1,0,0])
            _part2 = rotate_part(_part2_0, rotaxis, +th)
            VV = self.compute_effective_potential(_part1, _part2, EP, ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot));
            ########################################################
            if self.cgtype == 'cg1':
                CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
            gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )

        #ff.write("\n"); gg.write("\n")
        thend += np.pi/2.
        rotaxis = np.asarray([1,0,0])
        ## rotation 6: from perp-EP to PP rotating patches around x (EE+shift z)
        for th in np.linspace(0,np.pi/2.,self.path_N):
            ff.write("%.5e " % (th+thend)); gg.write("%.5e " % (th+thend));
            _part1 = self.generate_particle(EE[0])[:,:3]
            _part2_0 = self.generate_particle([[0,1,0],[0,-1,0]])[:,:3];
            _part2 = rotate_patches(_part2_0, rotaxis, th); _part2[:,:3] += _displ*np.asarray([1,0,0])
            VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot));
            ########################################################
            if self.cgtype == 'cg1':
                CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
            gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )

        ff.close(); gg.close()
        self.max_eff_pot = tempsave

    def rotate_2patches(self, _myaxis, fname_mf, fname_cg):

        to_orient = self.generate_orientations()
        rotaxes = [_myaxis for i in range(len(to_orient))]

        ff = open(fname_mf, "w"); gg = open(fname_cg, "w")
        for th in np.linspace(0,2.*np.pi,100):
            ff.write("%.5e " % (th)); gg.write("%.5e " % (th))
            for _ori in to_orient:
                _part1_0 = self.generate_particle(_ori[0])[:,:3]
                _part1 = rotate_patches(_part1_0, _myaxis, th)
                _part2_0 = self.generate_particle(_ori[1])[:,:3]
                _part2 = rotate_patches(_part2_0, _myaxis, th); _part2[:,:3] += np.asarray([1,0,0])
                VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], 'numeric')
                ff.write("%.5e " % (VV[0,2]))
                ########################################################
                sigma_core2 = self.sigma_core + self.delta/2.
                if self.cgtype == 'cg1':
                    EE = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
                else:
                    p = [self.sigma, self.delta]
                    for v in self.ecc:
                        p.extend([v])
                    EE = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
                gg.write("%.5e " % (np.dot(np.asarray(EE),self.u_cg)) )

            ff.write("\n"); gg.write("\n")

        ff.close(); gg.close()


    def find_pathway_extrema(self, fname_mf):

        data = np.loadtxt(fname_mf)

        ##chunks = data.shape[0]//self.path_N

        ### chunks to consider: 1, 3
        for i in [1,3]:
            temp = data[i*self.path_N:(i+1)*self.path_N]

            if i ==1:
                tmin = temp[np.where(temp[:,1] == np.amin(temp[:,1]))]
                self.path_min.extend([tmin[0,0]-i*np.pi/2.])
                #g = interp1d(temp[:,0], temp[:,1], kind='cubic')
                #try:
                #    res = minimize_scalar(g, bracket=[temp[0,0],temp[-1,0]], method='brent')
                #    self.path_min.extend([res.x-i*np.pi/2.])
                #except:
                #    print("no root for SP potential")
            if i == 3:
                self.path_max.extend([temp[self.path_N//2,0]-i*np.pi/2.])


    def potential_at_pathway_extrema(self, fname_mf, Np=100):

        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]
        rotaxis = np.asarray([0,1,0])

        ff = open(fname_mf, "w")

        for r in np.linspace(self.sigma,3*self.sigma, Np):

            ff.write("%.5e " % r)
            for th in self.path_min:
                _part1 = self.generate_particle(EE[0])[:,:3]
                _part2_0 = self.generate_particle(EE[1])[:,:3]; _part2_0[:,:3] += r*np.asarray([1,0,0])
                _part2 = rotate_part(_part2_0, rotaxis, th)
                VV = self.compute_effective_potential(_part1, _part2, EE, 'numeric')
                ff.write("%.5e " % (VV[0,2]));
                
            for th in self.path_max:
                _part1 = self.generate_particle(EP[0])[:,:3]
                _part2_0 = self.generate_particle(EP[1])[:,:3]; _part2_0[:,:3] += r*np.asarray([0,0,1])
                _part2 = rotate_part(_part2_0, rotaxis, -th)
                VV = self.compute_effective_potential(_part1, _part2, EP, 'numeric')
                ff.write("%.5e\n" % (VV[0,2]));


        ff.close()


'''
    def test_triangle_lattice(self, fname_mf, fname_cg):

        ##vectors
        tri_vec = [[1,0,0], [0.5, 0, 0.866025404], [-0.5, 0, 0.866025404], [-1, 0, 0], [-0.5, 0, -0.866025404], [0.5, 0, -0.866025404]]
        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]
        rotaxis = np.asarray([0,1,0])
        sigma_core2 = self.sigma_core + self.delta/2.

        ff = open(fname_mf, "w"); gg = open(fname_cg, "w")
        _part1 = self.generate_particle(EE[0])[:,:3]
        
        for th in np.linspace(0,2.*np.pi,100):
            ff.write("%.5e " % (th)); gg.write("%.5e " % (th))
            for tv in tri_vec:
                _part2_0 = self.generate_particle(EE[1])[:,:3]
                _part2 = rotate_patches(_part2_0, [0,1,0], th); _part2[:,:3] += np.array(tv)
                VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], 'numeric')
                ff.write("%.5e " % (VV[0,2]))
                ########################################################
                sigma_core2 = self.sigma_core + self.delta/2.
                EE = compute_dum_energy(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
                gg.write("%.5e " % (np.dot(np.asarray(EE),self.u_cg)) )

            ff.write("\n"); gg.write("\n")

        ff.close(); gg.close()


    def test_random_2P(self, r):

        topos = [[],[]]
        ## create a particle with random orientation
        a = random_on_sphere(); print("random vector, ", a , np.linalg.norm(a))
        _part1 = self.generate_particle([a,-a])[:,:3]
        topos[0].extend([a]); topos[0].extend([-a]); 
        
        ##create a particle at a random position with random orientation
        a = random_on_sphere(); #topos[1].extend([a]); topos[1].extend([-a])
        b = r*a;
        #a = a+0.01*random_on_sphere();
        topos[1].extend([a]); topos[1].extend([-a])
        _part2 = self.generate_particle([a,-a], e_c = b)[:,:3]
        print(_part1); print(_part2)
        print("topos", topos, "self.topo", self.topo)
        temp = self.compute_effective_potential(_part1, _part2, topos, 'yukawa')
        energy_ipc = temp[0,2]

        #############################################################################
        
        sigma_core2 = self.sigma_core + self.delta/2.
        if self.cgtype == 'cg1':
            EE = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
        else:
            p = [self.sigma, self.delta]
            for v in self.ecc:
                p.extend([v])
            EE = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
        energy_cg = np.dot(np.asarray(EE),self.u_cg)

        print(energy_ipc, energy_cg, energy_cg*self.max_eff_pot)

'''
