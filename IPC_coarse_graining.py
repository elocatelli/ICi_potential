import numpy as np

from IPC_effective_potentials import MinD
from utils import axis_angle_rotation, mybroadcast_to, move_part, rotate_patches

def gen_int(i1,i2, Np):
    ii = np.array([i1,i2]); ii = ii[ii.argsort()]; ##out = (ii[1]-ii[0])
    if ii[0] == 1 and ii[1] == 2:
        out = 1
    elif ii[0] == 1 and ii[1] == 3:
        out = 2
    elif ii[0] == 2 and ii[1] == 3:
        out = 3
    
    return out

############## CG 1: OVERLAP OF SPHERES ####################

def overlap_volume(d, R1, R2):

    R0 = np.minimum(R1,R2)
    rmin = np.fabs(R2 - R1)
    rmax = R2+R1

    out  = 0
    #print(d,R1, R2, R0, rmin, rmax)
    if d <= rmin:
        out = (4./3.*np.pi)*R0**3

    if d > rmin and d <= rmax:
        out = (np.pi/3.)*( 2*R1 + (R1**2 - R2**2 + d**2)/(2*d))*( R1 - (R1**2 - R2**2 + d**2)/(2*d))**2 
        out += (np.pi/3.)*( 2*R2 - (R1**2 - R2**2 - d**2)/(2*d))*( R2 + (R1**2 - R2**2 - d**2)/(2*d))**2  
 
    Vref = (4.*np.pi/3.)*0.5**3

    return out/Vref

def compute_dum_energy_CG1_0(_part1, _part2, Rc, Rp,box):

    Np1 = _part1.shape[0]
    Np2 = _part2.shape[0]
    out = np.zeros(3)  ###only in the case EE PP EP are sufficient
    R10 = 0; R20 = 0
    k = -1
    cc = MinD(_part1[0,:] - _part2[0,:],box); cc /= np.linalg.norm(cc)
    p1 = MinD(_part1[1,:] - _part1[2,:],box); p1 /= np.linalg.norm(p1)
    p2 = MinD(_part2[1,:] - _part2[2,:],box); p2 /= np.linalg.norm(p2)
    th_cp1 = np.dot(cc,p1); th_cp2 = np.dot(cc,p2); #th_pp = np.dot(p1,p2)

    for ll in range(Np1):
        for nn in range(Np2):

            if ll == 0 and nn == 0:
                R10 = Rc; R20 = Rc; k = 0; _myth = 1.
            elif ll == 0 and nn > 0:
                R10 = Rc; R20 = Rp[nn-1]; k = 2; _myth = (th_cp2)**4
            elif nn == 0 and ll > 0:
                R10 = Rp[ll-1]; R20 = Rc; k = 2; _myth = (th_cp1)**4
            else:
                R10 = Rp[ll-1]; R20 = Rp[nn-1]; k = 1; 
                pp = MinD(_part1[ll,:] - _part2[nn,:],box); pp /= np.linalg.norm(pp)
                _myth = np.maximum((np.dot(p1,pp))**4, (np.dot(p2,pp))**4 )

            d = np.linalg.norm(MinD(_part1[ll,:] - _part2[nn,:],box));

            vol = overlap_volume(d, R10, R20)
            #print(ll,nn, R10, R20, d, vol, _myth, cc)
            out[k] += vol*1. #_myth

    return out

def compute_dum_energy_CG1_2(_part1, _part2, Rc, Rp, box):

    Np1 = _part1.shape[0]
    Np2 = _part2.shape[0]
    if Np1 == 3:
        out = np.zeros(6)
    else:
        out = np.zeros(10)
    R10 = 0; R20 = 0
    k = -1

    for ll in range(Np1):
        for nn in range(Np2):

            if ll == 0 and nn == 0:       ##equator-equator
                R10 = Rc; R20 = Rc; k = 0
            elif ll == 0 and nn > 0:
                R10 = Rc; R20 = Rp[nn-1]; k = nn
            elif nn == 0 and ll > 0:
                R10 = Rp[ll-1]; R20 = Rc; k = ll
            else:
                R10 = Rp[ll-1]; R20 = Rp[nn-1]; 
                if ll == nn: 
                    k = Np1-1+ll
                else:
                    k = 2*(Np1-1)+gen_int(ll,nn,Np1-1)
            
            d = np.linalg.norm(MinD(_part1[ll,:] - _part2[nn,:],box));
            
            vol = overlap_volume(d, R10, R20)
            out[k] += vol

    return out

def compute_dum_energy_CG1(_part1, _part2, Rc, Rp, samepatch, box):

    if samepatch:
        out = compute_dum_energy_CG1_0(_part1, _part2, Rc, Rp, box)
    else:
        out = compute_dum_energy_CG1_2(_part1, _part2, Rc, Rp, box)

    return out


############## CG 2: DISTANCE EXPONENTIAL ####################

def CG_factors(d, i, j, params):
    
    ## params[0] = sigma params[1] = delta params[2...] = ecc
    out  = 0
    
    if i == 0 and j == 0:  ## core-core
        return np.exp(-(d-params[0])/params[1])
    elif i == 0 and j != 0:
        return np.exp(-(d-params[0]+params[j+1])/params[1])
    elif i != 0 and j == 0:
        return np.exp(-(d-params[0]+params[i+1])/params[1])
    else:
        return np.exp(-(d-params[0]+(params[i+1]+params[j+1]))/params[1])
    

def compute_dum_energy_0(_part1, _part2, params, box):

    Np1 = _part1.shape[0]
    Np2 = _part2.shape[0]
    out = np.zeros(3)  ###only in the case EE PP EP are sufficient
    
    cc = MinD(_part1[0,:] - _part2[0,:],box); cc /= np.linalg.norm(cc)
    if Np1 == 2:
        p1 = MinD(_part1[1,:] - _part1[0,:],box); p1 /= np.linalg.norm(p1)
    else:
        p1 = MinD(_part1[1,:] - _part1[2,:],box); p1 /= np.linalg.norm(p1)
    if Np2 == 2:
        p2 = MinD(_part2[1,:] - _part2[0,:],box); p2 /= np.linalg.norm(p2)
    else:
        p2 = MinD(_part2[1,:] - _part2[2,:],box); p2 /= np.linalg.norm(p2)
    th_cp1 = np.dot(cc,p1); th_cp2 = np.dot(cc,p2); #th_pp = np.dot(p1,p2)

    for ll in range(Np1):
        for nn in range(Np2):

            if ll == 0 and nn == 0:
                k = 0;
            elif ll == 0 and nn > 0:
                k = 2; 
            elif nn == 0 and ll > 0:
                k = 2; 
            else:
                k = 1;

            d = np.linalg.norm(MinD(_part1[ll,:] - _part2[nn,:],box));

            omega = CG_factors(d, ll, nn, params)
            out[k] += omega

    return out


def compute_dum_energy(_part1, _part2, params, samepatch, box):
    
    if samepatch:
        out = compute_dum_energy_0(_part1, _part2, params, box)
    else:
        print("TBD"); exit(1)
    #    out = compute_dum_energy_2(_part1, _part2, Rc, Rp, box)

    return out

class Mixin:

    def do_coarse_graining_max(self,funcname, funcname_1P):

        if self.npatch == 1:
            self.coarse_graining_max_1p(funcname, funcname_1P)
        else:
            self.coarse_graining_max(funcname, funcname_1P)

        self.print_coarse_graining_dist()
        #self.print_coarse_graining_omega('x')
        #self.print_coarse_graining_omega('y')
        #self.print_coarse_graining_omega('z')

        self.logfile_cg(funcname)

    def coarse_graining_max_1p(self, funcname, funcname_1P):

        sigma_core2 = self.sigma_core + self.delta/2.
        
        self.effective_potential_store(funcname, funcname_1P, Np=1)
        ##works only if patches are opposite
        ##three orientations: EE PP EP
        ##eff_pot has EE, PP, EP

        ##MAX cg: match the potential at contact
        MAX_eff_pot = self.effective_potential[0,1:] ##self.effective_potential[np.where(self.effective_potential[:,0] == self.sigma)]
        MAX_eff_pot = np.asarray([MAX_eff_pot[0]/np.abs(MAX_eff_pot[4]), MAX_eff_pot[2]/np.abs(MAX_eff_pot[4]), -1]) #MAX_eff_pot/np.amin(MAX_eff_pot)
        
        to_orient = self.generate_orientations()
        _mat = []
        i = 0
        for ff in to_orient:

            if i == 1 or i == 3 or i == 5:
                i +=1
                continue
            
            i+=1 
            _part1_0 = self.generate_particle(ff[0])
            _part2_0 = self.generate_particle(ff[1])

            _part1 = _part1_0[:,:3]; _part2_1 = _part2_0[:,:3]
            _part2 = move_part(_part2_1, [1.,0,0])

            if self.cgtype == 'cg1':
                _mat.extend(compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, True, self.box))
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                _mat.extend(compute_dum_energy(_part1, _part2, p, True, self.box))
            
        _mat = np.reshape(np.asarray(_mat),(3,3))
        self.u_cg = np.linalg.solve(_mat, MAX_eff_pot)
        np.savetxt(self.IPCdict['folder']+"/coefficients_cg_"+funcname+".dat", self.u_cg, fmt='%.8e') 

    def coarse_graining_max(self, funcname, funcname_1P):

        sigma_core2 = self.sigma_core + self.delta/2.
        ##works only if patches are opposite
        ##three orientations: EE PP EP
        ##eff_pot has EE, PP, EP
        print("performing coarse graining", self.cgtype) 
        self.effective_potential_store(funcname, funcname_1P, Np=1)
        ##MAX cg: match the potential at contact
        MAX_eff_pot = self.effective_potential[0,1:] ##self.effective_potential[np.where(self.effective_potential[:,0] == self.sigma)]
        self.max_eff_pot = np.abs(np.amin(MAX_eff_pot))
        MAX_eff_pot = MAX_eff_pot/self.max_eff_pot
        to_orient = self.generate_orientations()
        _mat = []; k = 0
        for ff in to_orient:
            if self.samepatch and k > 2:
                continue

            _part1_0 = self.generate_particle(ff[0])
            _part2_0 = self.generate_particle(ff[1])

            _part1 = _part1_0[:,:3]; _part2_1 = _part2_0[:,:3]
            
            _part2 = move_part(_part2_1, [1.,0,0])

            if self.cgtype == 'cg1':
                _mat.extend(compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box))
            else:
                p = [self.sigma, self.delta]
                for v in self.ecc:
                    p.extend([v])
                _mat.extend(compute_dum_energy(_part1, _part2, p, self.samepatch, self.box))
            k += 1
        
        _mat = np.reshape(np.asarray(_mat),(k,k))
        self.u_cg = np.linalg.solve(_mat, MAX_eff_pot[:k]); 
        np.savetxt(self.IPCdict['folder']+"/coefficients_cg_"+funcname+".dat", self.u_cg, fmt='%.18e')

    def print_coarse_graining_dist(self, Np=100):
        
        self.cg_potential = []
        sigma_core2 = self.sigma_core + self.delta/2.

        to_orient = self.generate_orientations()
       
        for r in np.linspace(self.sigma, 1.5*self.sigma, Np):
            self.cg_potential.extend([r])

            for ff in to_orient:
    
                _part1_0 = self.generate_particle(ff[0])
                _part2_0 = self.generate_particle(ff[1])

                _part1 = _part1_0[:,:3]; _part2_1 = _part2_0[:,:3]
                _part2 = move_part(_part2_1, [r,0,0]) 

                if self.cgtype == 'cg1':
                    EE = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
                else:
                    p = [self.sigma, self.delta]
                    for v in self.ecc:
                        p.extend([v])
                    EE = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
                self.cg_potential.extend([np.dot(np.asarray(EE),self.u_cg)])

        ##print
        self.cg_potential = np.reshape(np.asarray(self.cg_potential), (Np,len(to_orient)+1))

        np.savetxt(self.IPCdict['folder']+"/radial_cg_potential.dat", self.cg_potential, fmt='%.8e')


    def print_coarse_graining_omega(self, _axis, Np=100):
       
        if _axis == 'x':
            rotaxis = [1,0,0]
        elif _axis == 'y':
            rotaxis = [0,1,0]
        elif _axis == 'z':
            rotaxis = [0,0,1]
        else:
            print("cg potential: such axis ",_axis, " is not allowed"); exit(1)

        self.cg_potential = []
        sigma_core2 = self.sigma_core + self.delta/2.

        to_orient = self.generate_orientations()
       
        for th in np.linspace(0., 2.*np.pi, Np):
            self.cg_potential.extend([th])

            for ff in to_orient:
    
                _part1_0 = self.generate_particle(ff[0])
                _part2_0 = self.generate_particle(ff[1])

                _part1 = np.copy(_part1_0[:,:3])
                
                _part2_1 = rotate_patches(_part2_0[:,:3], rotaxis, th)

                _part2 = move_part(_part2_1, [1.,0,0]) 

                if self.cgtype == 'cg1':
                    EE = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
                else:
                    p = [self.sigma, self.delta]
                    for v in self.ecc:
                        p.extend([v])
                    EE = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
                self.cg_potential.extend([np.dot(np.asarray(EE),self.u_cg)])

        self.cg_potential = np.reshape(np.asarray(self.cg_potential), (Np,len(to_orient)+1))

        np.savetxt(self.IPCdict['folder']+"/rotation_"+_axis+"_cg_potential.dat", self.cg_potential, fmt='%.8e')


    def logfile_cg(self, funcname):
 
        outfname = self.IPCdict['folder']+"/logfile.dat"
        f = open(outfname,'a')

        f.write("**************\nOverlap of spheres model\n****************\n\n")

        f.write("Particles interaction range %.5f\t Patch Interaction Range %.5f\n" % (self.delta, self.delta) )
        f.write("Patch extension %s\t" %  (" ".join(str(x) for x in self.patch_sigma)) )
        f.write("Patch gamma %s\n" % (" ".join(str(x) for x in self.gamma) ) )

        if len(self.u_cg) == 0:
            if funcname == 'general':
                self.coarse_graining_max('general', 'numeric')
            elif funcname == 'yukawa':
                self.coarse_graining_max('yukawa', 'yukawa')
            else:
                print("keyword ", funcname, "not allowed\n"); exit(1)
        
        f.write("Normalization: %.8f\n" % (self.max_eff_pot))
        f.write("Get epsilon values:\n Epsilon BB  %.8f\t Epsilon PP %.8f\t Epsilon BP %.8f\n" % (self.u_cg[0] , self.u_cg[1] , self.u_cg[2]) )
        
        f.write("EQUATORIAL-EQUATORIAL\nENERGY (eV) %.8f\n" % (self.cg_potential[0,2]*self.max_eff_pot) )         
        f.write("POLAR-POLAR\nENERGY (eV) %.8f\n" % (self.cg_potential[0,1]*self.max_eff_pot) )
        f.write("EQUATORIAL-POLAR\nENERGY (eV) %.8f\n" % (self.cg_potential[0,3]*self.max_eff_pot) )
        ##f.write("DLVO bare colloid (eV)  " ,  , " DLVO bare colloid (KbT)  ", , "\n")     

  
        f.close()


