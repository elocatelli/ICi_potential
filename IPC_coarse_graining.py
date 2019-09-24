import numpy as np
from utils import axis_angle_rotation, mybroadcast_to 

def overlap_volume(d, R1, R2):

        R0 = np.minimum(R1,R2)
        rmin = np.fabs(R2 - R1)
        rmax = R2+R1

        out  = 0

        if d <= rmin:
                out = (4./3.*np.pi)*R0**3

        if d > rmin and d <= rmax:
                out = (np.pi/3.)*( 2*R1 + (R1**2 - R2**2 + d**2)/(2*d))*( R1 - (R1**2 - R2**2 + d**2)/(2*d))**2
                out += (np.pi/3.)*( 2*R2 - (R1**2 - R2**2 - d**2)/(2*d))*( R2 + (R1**2 - R2**2 - d**2)/(2*d))**2

        Vref = 1. #(4.*np.pi/3.)*0.5**3

        return out/Vref

def compute_dum_energy(_part1, _part2, Rc, Rp):

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
                    
                ##if ll == 0 and nn == 0 and d < 1.:
                ###    print "something is wrong cores are too close"; exit(1) 
                vol = overlap_volume(d, R10, R20)
                out[k] += vol             

        return out

def move_part(_part, _mov):
    _mov = np.asarray(_mov)
    return  _part + mybroadcast_to(_mov, _part.shape)

def rotate_part(_part, _axis, angle):
    _mat = axis_angle_rotation(_axis, angle)
    _out = np.empty_like(_part);

    for i in range(_part.shape[0]):
        _out[i,:3] = np.dot(_mat,_part[i,:3])

    return _out


class Mixin:
    
    def generate_particle(self, box):

        out = np.empty((self.npatch+1,4),dtype = float)

        # tetra  patch_vec = [[0.942809042, 0, -0.333333], [-0.471404521, 0.816496581, -0.33333 ], [-0.471404521, -0.816496581, -0.333333], [0,0,1]]
        #cross  patch_vec = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        #belt'   th = 2.*np.pi/(Npatch); patch_vec = []  for i in range(Npatch):     patch_vec.append([np.cos(i*th),np.sin(i*th),0])

        out[0,:] = np.asarray([box/2., box/2., box/2., self.sigma_core])
        for k in range(self.npatch):
            out[k+1,:3] = out[0,:3] + np.asarray(self.topo[k])*self.ecc[k]
            out[k+1,3] = self.sigma_patch[k]

        return out

    def coarse_graining_twopatches_max(self):
    
        sigma_core2 = self.sigma_core + self.delta/2.
    
        ##works only if patches are opposite
        ##three orientations: EE PP EP
        ##eff_pot has EE, PP, EP

        ##MAX cg: match the potential at contact
        MAX_eff_pot = self.effective_potential[np.where(self.effective_potential[:,0] == self.sigma)]
        MAX_eff_pot = MAX_eff_pot[0,1:].ravel()

        _part1_0 = self.generate_particle(0.)
        _part2_0 = np.copy(_part1_0) ##generate_particle(Npatch, box, sigma_core, sigma_patch, ecc, patch_vec)

        _part1_0 = _part1_0[:,:3]; _part2_0 = _part2_0[:,:3]

        ##EE
        _part1 = np.copy(_part1_0)
        _part2 = move_part(_part1, [self.sigma,0,0])
        EE = compute_dum_energy(_part1, _part2, sigma_core2, self.sigma_patch)

        ##PP
        _part1 = rotate_part(_part1_0, [0,1,0], np.pi/2.)
        _part2_1 = rotate_part(_part2_0, [0,1,0], -np.pi/2.)
        _part2 = move_part(_part2_1, [self.sigma,0,0])
        PP = compute_dum_energy(_part1, _part2, sigma_core2, self.sigma_patch)

        #EP
        _part1 = np.copy(_part1_0)
        _part2_1 = rotate_part(_part2_0, [0,1,0], -np.pi/2.)
        _part2 = move_part(_part2_1, [self.sigma,0,0])
        EP = compute_dum_energy(_part1, _part2, sigma_core2, self.sigma_patch)

        _mat = [EE,PP,EP]; _mat = np.reshape(np.asarray(_mat),(3,3))

        print _mat
        print MAX_eff_pot
        u_cg = np.linalg.solve(_mat, MAX_eff_pot)
        print u_cg

        ##EE
        _part1 = np.copy(_part1_0)
        VEE = []

        for r in np.linspace(self.sigma, 1.5*self.sigma, 100):
            _part2 = move_part(_part1, [r,0,0])
            EE = compute_dum_energy(_part1, _part2, sigma_core2, self.sigma_patch)
            VEE.extend([r,np.dot(np.asarray(EE),u_cg)])

        VEE = np.reshape(np.asarray(VEE),(len(VEE)/2,2))
    
        ##PP
        VPP = []
        _part1 = rotate_part(_part1_0, [0,1,0], np.pi/2.)
        _part2_1 = rotate_part(_part2_0, [0,1,0], -np.pi/2.)

        for r in np.linspace(self.sigma, 1.5*self.sigma, 100):
            _part2 = move_part(_part2_1, [r,0,0])
            PP = compute_dum_energy(_part1, _part2, sigma_core2, self.sigma_patch)
            VPP.extend([r,np.dot(np.asarray(PP),u_cg)])

        VPP = np.reshape(np.asarray(VPP),(len(VPP)/2,2))

        #EP
        VEP = []
        _part1 = np.copy(_part1_0)
        _part2_1 = rotate_part(_part2_0, [0,1,0], -np.pi/2.)

        for r in np.linspace(self.sigma, 1.5*self.sigma, 100):
            _part2 = move_part(_part2_1, [r,0,0])
            EP = compute_dum_energy(_part1, _part2, sigma_core2, self.sigma_patch)
            VEP.extend([r,np.dot(np.asarray(EP),u_cg)])

        VEP = np.reshape(np.asarray(VEP),(len(VEP)/2,2))

        ##print
        self.cg_potential = np.empty((VPP.shape[0],4),dtype = float)

        self.cg_potential[:,0] = VEE[:,0]; self.cg_potential[:,1] = VEE[:,1] 
        self.cg_potential[:,2] = VPP[:,1]; self.cg_potential[:,3] = VEP[:,1]

        return u_cg


