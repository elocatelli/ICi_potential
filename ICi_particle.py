import numpy as np
from utils import *
import os
from scipy.optimize import fsolve
from math import sin, cos

class Mixin:
	
    def restore_patches(self):
        self.topo = np.copy(self.topo_backup)
    
    def set_charge_topology(self):

        if not os.path.isfile(self.ICidict['topofile']):
            print("topology is not defined")
            self.default_topo = True
            if self.npatch == 2:
                print("using the default for %d patches: polar patches" % self.npatch)
                self.topo = [[0., 0., 1], [0., 0., -1]]
            else:
                print("no default topology for n!=2  patches")
                exit(1)
            
        else:
            ## MODIFY THIS TO A CHECK FOR POLAR PATCHES
            f = open(self.ICidict['topofile'], 'r')
            print("Using topology from file ", self.ICidict['topofile'])
            for lines in f:
                lines = lines.strip("\n");
                self.topo.append([float(x) for x in (lines.strip(" ").split(','))])
            print("Topology: ", self.topo)
            f.close()

            for el in self.topo:
                if np.fabs(np.linalg.norm(np.asarray(el)) - 1.) > 1E-5:
                    print("vector ", el, "is not a unit vector, err:", np.linalg.norm(np.asarray(el)) - 1.); exit(1)

            if len(self.topo) != self.npatch:
                print("topology ", self.topo, "is wrong, too many or too less unit vectors given"); exit(1)
            
            ## check for polar arrangement
            a = np.asarray(self.topo[0]); b = np.asarray(self.topo[1])
            if np.dot(a,b) > 1E-5:
                print("only polar arrangements are allowed!"); exit(1)
            else:
                self.default_topo = True


        self.topo = np.asarray(self.topo); self.topo_backup = np.copy(self.topo)
           

    def generate_particle(self, topo, e_c = [0.,0.,0.]):
         
        out = np.empty((self.npatch+1,4),dtype = float)

        if len(e_c) != 3:
            print("error in position vector"); exit(1)

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

