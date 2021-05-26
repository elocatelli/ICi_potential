import numpy as np
from utils import *
import os
from scipy.optimize import fsolve
from math import sin, cos

def doubleint_equations(p, sp, a):
    x,y,th = p
    return (x**2+y**2-0.25, (x-a[0])**2+y**2-sp[0]**2, (x-a[1]*cos(th))**2+(y-a[1]*sin(th))**2-sp[1]**2)

class Mixin:
	
    def calc_patch_angles(self):
        self.sph_theta = []; self.sph_phi = []
        for el in self.topo:
            temp = to_spherical(el) 
            self.sph_theta.extend([temp[1]]); self.sph_phi.extend([temp[2]])

    def restore_patches(self):
        self.topo = np.copy(self.topo_backup)
        self.calc_patch_angles()
    
    def check_doubleint(self):
        if self.npatch == 2:
            self.check_doubleint_2P()

    def check_doubleint_2P(self):
        
        my_th = np.arccos(np.dot(self.topo[0], self.topo[1]))
        if (np.pi-my_th) < 1E-2:
            print("patches are polar:angle between patches: ", my_th*180/np.pi)
            self.default_topo = True
            return

        x,y,th = fsolve(doubleint_equations, [0.5,0.5,1.57], args=(self.patch_sigma,self.ecc))

        if th > np.pi:
            th = 2.*np.pi-th

        if my_th < th:
            f = open("RES/effective_potential_general.dat", "w")
            f.write("an angle of %.5f is forbidden! minimum angle to have two patches is %.5f\n" % (my_th*180/np.pi, th*180/np.pi))
            f.close()
            print(my_th, "forbidden! minimum angle to have two patches is", th*180./np.pi); exit(1)

        th = 2.*np.arccos((0.5-self.patch_sigma)/(self.ecc))
        if my_th < np.amax(th):
            self.doubleint = True


    def set_charge_topology(self):

        if not os.path.isfile(self.IPCdict['topofile']):

            print("topology is not defined")
            self.default_topo = True
            if self.npatch == 1:
                self.topo = [[1., 0., 0.]]
            elif self.npatch == 2:
                print("using the default for %d patches: polar patches" % self.npatch)
                self.topo = [[0., 0., 1], [0., 0., -1]]
            elif self.npatch == 3:
                print("using the default for %d patches: triangle patches" % self.npatch)
                #self.topo = [[0., 0., 1], [0.86602, 0., -0.5], [-0.86602, 0., -0.5]]
                self.topo = [[1.0, 0.0, 0.0], [-0.5, 0.8660254037844387, 0.0], [-0.50, -0.8660254037844384, 0.0]]  
            else:
                print("no default topology for more than 3 patches")
                exit(1)
            
        else:
            f = open(self.IPCdict['topofile'], 'r')

            for lines in f:
                lines = lines.strip("\n");
                self.topo.append([float(x) for x in (lines.strip(" ").split(','))])

            f.close()

            for el in self.topo:
                if np.fabs(np.linalg.norm(np.asarray(el)) - 1.) > 1E-5:
                    print("vector ", el, "is not a unit vector, err:", np.linalg.norm(np.asarray(el)) - 1.)
                    exit(1)

            if len(self.topo) != self.npatch:
                print("topology ", self.topo, "is wrong, too many or too less unit vectors given"); exit(1)
        
            self.check_doubleint()

            ### for 3 patches, check that they lay in a plane
            if self.npatch == 3:
                cth = np.dot(self.topo[0], np.cross(self.topo[1],self.topo[2]))
                if np.abs(cth) > 1E-5:
                    print("vectors are not coplanar")
                else:
                    print("vectors are coplanar")            

        if self.npatch == 1:
            self.samepatch = True
        self.calc_patch_angles(); ##print("sph_theta ", self.sph_theta, " sph_phi ", self.sph_phi)
        self.topo = np.asarray(self.topo); self.topo_backup = np.copy(self.topo)
           

    def generate_particle(self, topo, e_c = [0.,0.,0.]): #topo = self.topo):
         
        out = np.empty((self.npatch+1,4),dtype = float)

        if len(e_c) != 3:
            print("error in position vector"); exit(1)

        # tetra  patch_vec = [[0.942809042, 0, -0.333333], [-0.471404521, 0.816496581, -0.33333 ], [-0.471404521, -0.816496581, -0.333333], [0,0,1]]
        # cross  patch_vec = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        # belt'   th = 2.*np.pi/(Npatch); patch_vec = []  for i in range(Npatch): patch_vec.append([np.cos(i*th),np.sin(i*th),0])

        out[0,:] = np.asarray([e_c[0], e_c[1], e_c[2], self.sigma_core])
        for k in range(self.npatch):
            out[k+1,:3] = out[0,:3] + np.asarray(topo[k])*self.ecc[k]
            out[k+1,3] = self.patch_sigma[k]
        
        return out
        

    def topo_rotate(self, _axis, _angle):
        _mat = axis_angle_rotation(_axis,_angle)
        out = []
        for i in range(self.npatch):
            out.append(np.dot(_mat,self.topo[i]))
        return out

