import numpy as np
from utils import axis_angle_rotation, mybroadcast_to, move_part, rotate_part

def gen_int(i1,i2, Np):
    ii = np.array([i1,i2]); ii = ii[ii.argsort()]; ##out = (ii[1]-ii[0])
    if ii[0] == 1 and ii[1] == 2:
        out = 1
    elif ii[0] == 1 and ii[1] == 3:
        out = 2
    elif ii[0] == 2 and ii[1] == 3:
        out = 3
    
    return out

def overlap_volume(d, R1, R2):

    R0 = np.minimum(R1,R2)
    rmin = np.fabs(R2 - R1)
    rmax = R2+R1

    out  = 0
    #print(d,R1, R2, R0, rmin, rmax)
    if d <= rmin:
        out = (4./3.*np.pi)*R0**3

    if d > rmin and d <= rmax:
        print("d %.5f R1 %.5f R2 %.5f R0 %.5f rmin %.5f rmax %.5f\n" % (d, R1, R2, R0, rmin, rmax))
        out = (np.pi/3.)*( 2*R1 + (R1**2 - R2**2 + d**2)/(2*d))*( R1 - (R1**2 - R2**2 + d**2)/(2*d))**2; print("out_1 ", out)
        out += (np.pi/3.)*( 2*R2 - (R1**2 - R2**2 - d**2)/(2*d))*( R2 + (R1**2 - R2**2 - d**2)/(2*d))**2; print("out_2 ", out) 
 
    Vref = (4.*np.pi/3.)*0.5**3

    return out/Vref

def compute_dum_energy_0(_part1, _part2, Rc, Rp):

    Np1 = _part1.shape[0]
    Np2 = _part2.shape[0]
    out = np.zeros(3)  ###only in the case EE PP EP are sufficient
    R10 = 0; R20 = 0
    k = -1

    for ll in range(Np1):
        for nn in range(Np2):

            if ll == 0 and nn == 0:
                R10 = Rc; R20 = Rc; k = 0
            elif ll == 0 and nn > 0:
                R10 = Rc; R20 = Rp[nn-1]; k = 2
            elif nn == 0 and ll > 0:
                R10 = Rp[ll-1]; R20 = Rc; k = 2
            else:
                R10 = Rp[ll-1]; R20 = Rp[nn-1]; k = 1

            d = np.linalg.norm(_part1[ll,:] - _part2[nn,:]);

            vol = overlap_volume(d, R10, R20)
            print(ll, nn, d, vol)
            out[k] += vol

    return out

def compute_dum_energy_2(_part1, _part2, Rc, Rp):

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
            
            d = np.linalg.norm(_part1[ll,:] - _part2[nn,:]);
            
            vol = overlap_volume(d, R10, R20)
            out[k] += vol

    return out

def compute_dum_energy(_part1, _part2, Rc, Rp, samepatch):
    
    if samepatch:
        out = compute_dum_energy_0(_part1, _part2, Rc, Rp)
    else:
        out = compute_dum_energy_2(_part1, _part2, Rc, Rp)

    return out

class Mixin:

    def do_coarse_graining_max(self):

        if self.npatch == 1:
            self.coarse_graining_max_1p()
        else:
            self.coarse_graining_max()

        #self.print_coarse_graining_dist()
        #self.print_coarse_graining_omega('y')
        #self.print_coarse_graining_omega('z')

    def coarse_graining_max_1p(self):

        sigma_core2 = self.sigma_core + self.delta/2.

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

            ##EE
            _mat.extend(compute_dum_energy(_part1, _part2, sigma_core2, self.patch_sigma, True))
            
        _mat = np.reshape(np.asarray(_mat),(3,3))
        self.u_cg = np.linalg.solve(_mat, MAX_eff_pot)
        

    def coarse_graining_max(self):

        sigma_core2 = self.sigma_core + self.delta/2.
        print("sigma_core2 %.5f delta %.5f" % (sigma_core2, self.delta))
        ##works only if patches are opposite
        ##three orientations: EE PP EP
        ##eff_pot has EE, PP, EP

        ##MAX cg: match the potential at contact
        MAX_eff_pot = self.effective_potential[0,1:] ##self.effective_potential[np.where(self.effective_potential[:,0] == self.sigma)]
        MAX_eff_pot = MAX_eff_pot/np.abs(np.amin(MAX_eff_pot))
 
        to_orient = self.generate_orientations()
        _mat = []; k = 0
        for ff in to_orient:
            if self.samepatch and k > 2:
                continue

            _part1_0 = self.generate_particle(ff[0])
            _part2_0 = self.generate_particle(ff[1])

            _part1 = _part1_0[:,:3]; _part2_1 = _part2_0[:,:3]
            
            _part2 = move_part(_part2_1, [1.,0,0])

            _mat.extend(compute_dum_energy(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch))
            print("----------------------------------------------------------------------")
            k += 1
        
        _mat = np.reshape(np.asarray(_mat),(k,k))
        self.u_cg = np.linalg.solve(_mat, MAX_eff_pot[:k]); print(self.u_cg)
        np.savetxt(self.IPCdict['folder']+"/coefficients_cg.dat", self.u_cg, fmt='%.8e')

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

                EE = compute_dum_energy(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch)
                self.cg_potential.extend([np.dot(np.asarray(EE),self.u_cg)])

        ##print
        self.cg_potential = np.reshape(np.asarray(self.cg_potential), (Np,len(to_orient)+1))

        np.savetxt(self.IPCdict['folder']+"/potential_cg.dat", self.cg_potential, fmt='%.8e')


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

                _part1 = _part1_0[:,:3]; 
                _part2_1 = rotate_part(_part2_0[:,:3], rotaxis, th)

                _part2 = move_part(_part2_1, [1.,0,0]) 

                EE = compute_dum_energy(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch)
                self.cg_potential.extend([np.dot(np.asarray(EE),self.u_cg)])

        ##print
        self.cg_potential = np.reshape(np.asarray(self.cg_potential), (Np,len(to_orient)+1))

        np.savetxt(self.IPCdict['folder']+"/potential_cg_omega_"+_axis+".dat", self.cg_potential, fmt='%.8e')

