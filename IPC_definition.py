import numpy as np
import string
import os 

try:
    _mod = 'matplotlib'; import matplotlib.pyplot as plt
    _plt = True
except Exception as e:
    print(e); _plt = False

from utils import safecopy, myconvert, to_spherical
import IPC_particle
import IPC_orientations
import IPC_SP_potentials
import IPC_effective_potentials
import IPC_coarse_graining

class IPC(IPC_particle.Mixin, IPC_orientations.Mixin, IPC_SP_potentials.Mixin, IPC_effective_potentials.Mixin, IPC_coarse_graining.Mixin):

    def __init__(self) :

        self.IPCdict = {'folder': 'IPC_RESULTS', 'toread': 'params.dat', 'nr_patches' : 2, 'same_patch' : True, 'colloid_charge' : 0., 'patch_charge' :  np.zeros(2), 'sigma' : 1., 'sigma_core' : 0.5, 'patch_sigma' : np.zeros(2), 'screening_constant' : 2., 'debye_constant' : 1., 'kappa' : 0., 'gamma' : np.zeros(2), 'cosgamma' : np.zeros(2), 'delta' : 0., 'eccentricity' : np.zeros(2), 'bjerrum' : 0.7, 'epsilon' : 80., 'real_size' : 200., 'topofile' : 'topology.dat', 'topo' : [], 'lmax' : 30, 'plt' : _plt }


        self.analytic_funcdict = {'yukawa' : self.potential_outside_2sympatches_yukawa, '2patch_analytic' : self.potential_outside_2sympatches, 'numeric' : self.potential_outside_dd }

        self.effpot_funcdict = {'yukawa' : self.effective_potentials_yukawa, 'general' : self.effective_potential, '2patch': self.effective_potential_2patches}
        self.cg_funcdict = {}

        self.npatch = 2                 ##nr patches
        self.samepatch = True                   ## patches are all the same or different
        self.colloid_charge = 0.        ## central colloid charge
        self.patch_charge = np.empty(1)         ## array for patch charges
        self.sigma = 1
        self.sigma_core = 0.5                   ## sigma_core remember! sigma_core/patch is the radius not the diameter
        self.patch_sigma = np.empty(1)      ## array of patch_sigma
        self.emme = 0                           ## screening constant
        self.enne = 0                   ## debye constant
        self.kappa = 0.                         ## (inverse) debye length
        self.gamma = np.empty(1)        ## array of angle aperture patch
        self.cosgamma = np.empty(1)             ## cosine of gamma
        self.delta = 0.                 ## interaction range
        self.ecc = np.empty(1)                  ## array of eccentricity

        self.bjerrum_real = 0.7             ##nm
        self.eps = 80.
        self.eps1 = 4.
        self.eps2 = 80.

        self.real_size = 200E-9                ## m

        self.e_charge = 1.602E-19           ## C
        self.permittivity = 8.987E9 #*1E18    ## N*m^2/C^2
        self.temperature = 300              ## K
        self.kB = 1.38E-23                  ## J/K

        self.bjerrum = self.e_charge**2*self.permittivity/(self.eps*self.kB*self.temperature)
        print("check bjerrum: %.5e vs %.5e\n" % (self.bjerrum_real, self.bjerrum))

        self.topo = []                      ##unit vectors defining the position of the charges/patches
        self.topo_backup = []
        self.sph_theta = []                ##theta and phi values corresponding to the unit vectors given in topo
        self.sph_phi = []
        self.default_topo = False
        self.doubleint = False

        self.lmax = 30                      ###highest lmax = 80
            
        self.is_wall = False                    ##is there a wall?

        self.plot = _plt                    ##

        self.plist_part = ['nr_patches', 'same_patch', 'colloid_charge', 'patch_charge', 'patch_sigma', 'screening_constant', 'debye_constant', 'kappa', 'gamma', 'delta', 'eccentricity']
        self.plist_wall = ['wall_charge']
        self.plist_misc = ['plot']

        #################################

        self.DH_potential = []
        self.effective_potential = []
        self.cg_potential = []
        self.u_cg = []

    def set_params(self):

        ##create the folder 
        os.makedirs(self.IPCdict['folder'], exist_ok=True)

        f = open(self.IPCdict['toread'], 'r')
        plist = list()

        nr_params = 1
        for line in f:

            try:
                name, var = line.partition("=")[::2]
                #print("name", name); print("content", var, type(var))
            except Exception as e:
                print(e); exit(1)
            
            name = name.strip()
            var = var.strip("\n"); var = var.strip(" ") 
            #myconvert(self.IPCdict[name], var)

            #elems = string.split(line.strip("\n"), "=")
            if name[0] == '#':
                continue
            if name[0] == ' ':
                continue

            if name == 'nr_patches':
                self.npatch = int(var)
                self.patch_charge = np.empty((self.npatch,1)); self.patch_sigma = np.empty((self.npatch,1))
                self.ecc = np.empty((self.npatch,1)); self.gamma = np.empty((self.npatch,1))
            elif  name == 'same_patch':
                if var == 'False':
                    self.samepatch = False
                elif var == 'True':
                    self.samepatch = True
                else:
                    raise ValueError
            elif name == 'patch_charge':
                temp = np.asarray(var.split(','), dtype = float)
                safecopy(temp, self.patch_charge)
            elif name == 'colloid_charge':
                self.colloid_charge = float(var)
            #elif elems[0] == 'sigma_core':
            #    self.sigma_core = float(elems[1])
            elif name == 'patch_sigma':
                temp = np.asarray(var.split(', '), dtype = float)
                safecopy(temp, self.patch_sigma)
            elif name == 'debye_constant':
                self.enne = float(var)
            elif name == 'screening_constant':
                self.emme = float(var)
            elif name == 'kappa':
                self.kappa = float(var)
            elif name == 'eccentricity':
                temp = np.asarray(var.split(', '), dtype = float)
                safecopy(temp, self.ecc)
            elif name == 'gamma':
                temp = np.asarray(var.split(', '), dtype = float)
                safecopy(temp, self.gamma)
                self.cosgamma = np.cos(self.gamma)
            elif name == 'delta':
                self.delta = float(var)
            elif name == 'eps1':
                self.eps1 = float(var)
            elif name == 'lmax':
                self.lmax = int(var)
            elif name == 'real_size':
                self.real_size = float(var)
            elif name == 'plot':
                self.plot = var
            else :
                print("entry:", name, "unkown")
                exit(1)

            if name in self.plist_part and name not in plist:
                plist.extend([name])

        f.close()


        ##check for errors
        if 'nr_patches' not in plist:
            print("number of patch not found, defaulting to 2!")
        if 'same_patch' not in plist:
            print("same_patch not specified, defaulting to True")
        if 'colloid_charge' not in plist:
            print('colloid charge required')
            exit(1)
        if 'patch_charge' not in plist:
            print('patch_charge required')
            exit(1)

        if self.lmax > 90:
             print("lmax too large, sum over l in analytic solution will not converge"); exit(1)

        plist2 = plist[:]
        try:
            plist2.remove('nr_patches');
        except Exception as e:
            print(e); print("nr patches was not specified")
        try:
            plist2.remove('same_patch')
        except Exception as e:
            print(e); print("same_patch was not specified")

        plist2.remove('colloid_charge');
        plist2.remove('patch_charge');

        if len(plist2) < 3:
            print("Error: too few parameters", plist2); exit(1)

        if not ('debye_constant' in plist2 or 'screening_constant' in plist2 or 'kappa' in plist2) or not ('patch_sigma' in plist2 or 'gamma' in plist2 or 'eccentricity' in plist2):
            print("at least one between: debye_constant, screening_constant and kappa and one between: patch_sigma, gamma, eccentricity are needed")
            exit(1)

        if len(plist2) == 3:

            if 'patch_sigma' in plist2 and 'debye_constant' in plist2 and 'screening_constant' in plist2 :
                self.kappa = self.emme/self.sigma_core
                self.delta = self.enne/self.kappa
                self.ecc = self.delta/2. + self.sigma_core - self.patch_sigma
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            elif 'patch_sigma' in plist2 and 'debye_constant' in plist2 and 'kappa' in plist2 :
                print("not supported"); exit(1)

            elif 'patch_sigma' in plist2 and 'screening_constant' in plist2 and 'gamma' in plist2 :
                print("not supported"); exit(1)

            elif 'patch_sigma' in plist2 and 'screening_constant' in plist2 and 'delta' in plist2 :
                self.kappa = self.emme/self.sigma_core
                self.enne = self.delta/self.kappa
                self.ecc = self.delta/2. + self.sigma_core - self.patch_sigma
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            elif 'patch_sigma' in plist2 and 'screening_constant' in plist2 and 'eccentricity' in plist2 :
                self.kappa = self.emme/self.sigma_core
                self.delta = 2*(self.ecc - self.sigma_core + self.patch_sigma)
                self.enne = self.delta*self.kappa
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            elif 'eccentricity' in plist2 and 'screening_constant' in plist2 and 'debye_constant' in plist2:
                self.kappa = self.emme/self.sigma_core
                self.delta = self.enne/self.kappa;    print("DELTA %.3f" % self.delta)
                self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc;
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)
 
            elif 'gamma' in plist2 and 'screening_constant' in plist2 and 'debye_constant' in plist2:
                self.kappa = self.emme/self.sigma_core
                self.delta = self.enne/self.kappa; 
                self.cosgamma = np.cos(self.gamma)
                self.ecc = (self.delta**2+self.sigma_core*self.delta)/(self.delta+2.*self.sigma_core-2*self.sigma_core*self.cosgamma)
                self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc;

            else:
                print("not supported"); exit(1)

            #self.ecc = self.delta/2. + self.sigma_core - self.patch_sigma
            #self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*np.dot(self.patch_sigma,self.ecc))
            #self.gamma = np.arccos(self.cosgamma)

            #self.ecc = self.delta*(self.delta/4. + self.sigma_core)/(self.delta + 2.*self.sigma_core*(1.-self.cosgamma))
            #self.patch_sigma = ((2.*self.sigma_core**2 + self.sigma_core*self.delta)*(1. - self.cosgamma) + self.delta**2/4.)/( self.delta + 2.*self.sigma_core*(1.-self.cosgamma) )

            #self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc

        if len(plist2) > 3:
            print("too many inputs; not supported"); exit(1)

        #####
        if not np.all(self.patch_charge == self.patch_charge[0]) or not np.all(self.patch_sigma == self.patch_sigma[0]) or not np.all(self.ecc == self.ecc[0]) or not np.all(self.gamma == self.gamma[0]):
            print("are we sure?", self.patch_charge, self.patch_sigma, self.ecc, self.gamma)
            self.samepatch = False
        if np.all(self.patch_charge == self.patch_charge[0]) and np.all(self.patch_sigma == self.patch_sigma[0]) and np.all(self.ecc == self.ecc[0]) and np.all(self.gamma == self.gamma[0]):
            self.samepatch = True; print("ecc", self.ecc)

        ###if self.IPCdict['folder'] == 'default_folder':
        ###    self.IPCdict['folder'] = "IPC_Np"+str(self.npatch)+"/ecc_"+"_".join(self.ecc.astype(str))+"/Zc_"+str(self.



    def check_params(self):

        outfname = self.IPCdict['folder']+"/parameters_check.dat"
        f = open(outfname,'w')
        f.write("nr_patches = %d\n" % self.npatch)
        f.write("same_patch = %s\n" % str(self.samepatch))
        f.write("patch_charge = %s\n" % (" ".join(str(x) for x in self.patch_charge)) )
        f.write("colloid_charge = %.3f\n" % self.colloid_charge)
        f.write("patch_sigma = %s\n" % (" ".join(str(x) for x in self.patch_sigma) ) )
        f.write("debye_constant = %.3f\n" % self.enne )
        f.write("screening_constant = %.3f\n" % self.emme )
        f.write("kappa = %.3f\n" % self.kappa)
        f.write("eccentricity = %s\n" % (" ".join(str(x) for x in self.ecc) ) )
        f.write("gamma = %s\n" % (" ".join(str(x*180/np.pi) for x in self.gamma) ) )
        f.write("cosgamma = %s\n" % (" ".join(str(x) for x in self.cosgamma) ) )
        f.write("delta = %.3f\n" % self.delta)
        f.write("lmax = %d\n" % self.lmax)

        f.close()

  
