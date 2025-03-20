import numpy as np
import string
import os 

try:
    _mod = 'numpy'; import numpy as np
    _mod = 'scipy'; import scipy as sp ##improve import
except ImportError:
    print("necessary module ", _mod," missing"); exit(1)

from utils import safecopy, myconvert
import ICi_particle
import ICi_orientations
import ICi_sp_potentials
import ICi_effective_potentials
import ICi_pathway

class ICi(ICi_particle.Mixin, ICi_orientations.Mixin, ICi_sp_potentials.Mixin, ICi_effective_potentials.Mixin, ICi_pathway.Mixin):

    def __init__(self) :

        self.ICidict = {'folder': 'ICi_RESULTS', 'toread': 'params.dat', 'topofile' : 'topology.dat'}
 
        self.analytic_funcdict = {'2patch' : self.potential_outside_2sympatches }

        self.effpot_funcdict = {'2patch': self.effective_potential_2patches}

        self.npatch = 2                     ##nr patches
        self.samepatch = True               ## patches are all the same or different
        self.colloid_charge = 0.            ## central colloid charge
        self.patch_charge = np.empty(1)     ## array for patch charges
        self.sigma = 1
        self.sigma_core = 0.5               ## sigma_core remember! sigma_core/patch is the radius not the diameter
        self.patch_sigma = np.empty(1)      ## array of patch_sigma
        self.emme = 0                       ## screening constant

        self.kappa = 0.                     ## (inverse) debye length
        self.ecc = np.empty(1)              ## array of eccentricity

        self.bjerrum = 0.7                  ##nm
        self.eps = 80.

        self.real_size = 200E-9             ## m real diameter of the colloid
        self.salt = 1                       ## in mM or mol/m^3
        self.zeta_pot = -20                 ## zeta potential in mV

        self.imbalance = 0
        self.real_units = False 
        self.e_charge = 1.602E-19           ## C
        self.permittivity = 8.987E9         ## N*m^2/C^2
        self.temperature = 293.15           ## K
        self.kB = 1.380649E-23              ## J/K
        self.avogadro = 6.02214E23

        self.bjerrum_real = self.e_charge**2*self.permittivity/(self.eps*self.kB*self.temperature)
        self.f_kBT = 1.

        self.topo = []                      ##unit vectors defining the position of the charges/patches
        self.topo_backup = []
        self.default_topo = False

        self.lmax = 30                      ###highest lmax = 80
        
        self.sp_zero = -1.
        self.box = np.asarray([100.,100.,100.])
        self.max_eff_pot = 0.                  

        self.path_N = 100
        self.path_max = []
        self.path_min = []
        self.path_displ = 1.
        self.thend = 0
        self.fpath_mf = 'dummy.dat'

        self.plist_part = ['nr_patches', 'same_patch', 'salt', 'colloid_charge', 'patch_charge', 'patch_sigma', 'screening_constant', 'debye_constant', 'kappa', 'gamma', 'delta', 'eccentricity']
        
        #################################

        self.sp_potential = []
        self.effective_potential = []

    def set_params(self):

        ##create the folder 
        os.makedirs(self.ICidict['folder'], exist_ok=True)

        f = open(self.ICidict['toread'], 'r')
        plist = list()

        nr_params = 1
        for line in f:

            try:
                name, var = line.partition("=")[::2]
            except Exception as e:
                print(e); exit(1)
            
            name = name.strip()
            var = var.strip("\n"); var = var.strip(" ") 

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
            elif name == 'screening_constant':
                self.emme = float(var)
            elif name == 'kappa':
                self.kappa = float(var)
            elif name == 'eccentricity':
                temp = np.asarray(var.split(', '), dtype = float)
                safecopy(temp, self.ecc)
            elif name == 'lmax':
                self.lmax = int(var)
            elif name == 'real_size':
                self.real_size = float(var)
            elif name == 'energy_units':
                if var == 'False':
                    self.real_units = False
                elif var == 'True':
                    self.real_units = True
                else:
                    raise ValueError
            elif name == 'pathway_distance':
                self.path_displ = float(var)
            else :
                print("entry:", name, "unkown")
                exit(1)
            
            if name in self.plist_part and name not in plist:
                plist.extend([name])

        f.close()

        self.f_kBT=self.e_charge/(self.kB*self.temperature)   ##self.bjerrum_real*self.eps/self.real_size

        ##check for errors
        if 'nr_patches' not in plist:
            print("number of patch not found, defaulting to 2!")
        if 'same_patch' not in plist:
            print("same_patch not specified, defaulting to True")
        
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

        if 'colloid_charge' in plist:
            plist2.remove('colloid_charge');
        if 'patch_charge' in plist:
            plist2.remove('patch_charge');

        if len(plist2) < 2:
            print("Error: too few parameters", plist2); exit(1)

        if not ('screening_constant' in plist2 or 'kappa' in plist2 or 'salt' in plist2) or not ('eccentricity' in plist2):
            print("screening_constant and kappa and eccentricity are needed")
            exit(1)

        if len(plist2) == 2:

            if 'screening_constant' in plist2 :
                self.kappa = self.emme/self.sigma_core
            elif 'kappa' in plist2 :
                self.emme = self.kappa*self.sigma_core

            else:
                print("not supported"); exit(1)

        if 'colloid_charge' not in plist:
            print('colloid charge not present but mandatory'); exit(1)
        if 'patch_charge' not in plist: 
            if self.samepatch == True:
                print('patch charge not present but but mandatory'); exit(1)
            else:
                print('patch_charge required if patches are different'); exit(1)

        if len(plist2) > 3:
            print("too many inputs; not supported"); exit(1)

        #####
        if self.npatch != 2:
            print("only polar patches are admitted in this version of the code"); exit(1)

        if not np.all(self.patch_charge == self.patch_charge[0]) or not np.all(self.ecc == self.ecc[0]):
            print("check:", self.patch_charge, self.ecc)
            self.samepatch = False
        if np.all(self.patch_charge == self.patch_charge[0]) and np.all(self.patch_sigma == self.patch_sigma[0]) and np.all(self.ecc == self.ecc[0]):
            self.samepatch = True


    def check_params(self):
        
        outfname = self.ICidict['folder']+"/parameters_check.dat"
        f = open(outfname,'w')
        f.write("nr_patches = %d\n" % self.npatch)
        f.write("same_patch = %s\n" % str(self.samepatch))
        f.write("patch_charge = %s\n" % (" ".join(str(x) for x in self.patch_charge)) )
        f.write("colloid_charge = %.3f\n" % self.colloid_charge)
        f.write("patch_sigma = %s\n" % (" ".join(str(x) for x in self.patch_sigma) ) )
        f.write("screening_constant = %.3f\n" % self.emme )
        f.write("kappa = %.3f\n" % self.kappa)
        f.write("eccentricity = %s\n" % (" ".join(str(x) for x in self.ecc) ) )
        f.write("lmax = %d\n" % self.lmax)
        f.write("real colloid size = %.5e\n" % self.real_size)

        f.close()

