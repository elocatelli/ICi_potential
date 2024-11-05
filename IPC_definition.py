import numpy as np
import string
import os 

try:
    _mod = 'numpy'; import numpy as np
    _mod = 'scipy'; import scipy as sp ##improve import
except ImportError:
    print("necessary module ", _mod," missing"); exit(1)

_plt = False
try:
    _mod = 'matplotlib'; import matplotlib.pyplot as plt
    _plt = True
except ImportError as e:
    print(e); _plt = False

def calc_kappa(_salt, LB, NA, real_size):

    # kappa in units of sigma

    # For a 1:1 electrolyte, where each ion is singly-charged,
    # the ionic strength is equal to the concentration (wiki)

    return np.sqrt(8*np.pi*NA*LB*_salt)*real_size

def calc_charge(LB, real_size, emme, psi, Np, ch_imb):

   sigma_c = real_size/2.
   Z = sigma_c*(1+emme)/LB*psi 
   C=10
   if np.abs(Z)> C*sigma_c/LB:
       print('CHARGE RENORMALIZATION REGIME REACHED!', Z, ' C = Z LB/ sigma_c', np.abs(Z)*LB/sigma_c )
       temp = np.sign(Z)*C*sigma_c/LB; Z = temp
   
   temp = np.fabs(Z)*ch_imb+Z; Z = temp

   return Z/Np


from utils import safecopy, myconvert, to_spherical
import IPC_particle
import IPC_orientations
import IPC_SP_potentials
import IPC_effective_potentials
import IPC_coarse_graining
import IPC_tests
import IPC_pathway

class IPC(IPC_particle.Mixin, IPC_orientations.Mixin, IPC_SP_potentials.Mixin, IPC_effective_potentials.Mixin, IPC_coarse_graining.Mixin, IPC_tests.Mixin, IPC_pathway.Mixin):

    def __init__(self) :

        self.IPCdict = {'folder': 'IPC_RESULTS', 'toread': 'params.dat', 'nr_patches' : 2, 'same_patch' : True, 'colloid_charge' : 0., 'patch_charge' :  np.zeros(2), 'sigma' : 1., 'sigma_core' : 0.5, 'patch_sigma' : np.zeros(2), 'screening_constant' : 2., 'debye_constant' : 1., 'kappa' : 0., 'gamma' : np.zeros(2), 'cosgamma' : np.zeros(2), 'delta' : 0., 'eccentricity' : np.zeros(2), 'bjerrum' : 0.7, 'epsilon' : 80., 'real_size' : 200., 'topofile' : 'topology.dat', 'topo' : [], 'lmax' : 30, 'plt' : _plt }

        self.systype = 'general'
        self.cgtype = 'cg2'
        self.analytic_funcdict = {'yukawa' : self.potential_outside_2sympatches_yukawa, '2patch' : self.potential_outside_2sympatches, 'numeric' : self.potential_outside_dd }

        self.effpot_funcdict = {'yukawa' : self.effective_potentials_yukawa, 'general' : self.effective_potential, '2patch': self.effective_potential_2patches}
        self.cg_funcdict = {}

        self.npatch = 2                    ##nr patches
        self.samepatch = True              ## patches are all the same or different
        self.colloid_charge = 0.           ## central colloid charge
        self.patch_charge = np.empty(1)    ## array for patch charges
        self.sigma = 1
        self.sigma_core = 0.5              ## sigma_core remember! sigma_core/patch is the radius not the diameter
        self.patch_sigma = np.empty(1)     ## array of patch_sigma
        self.emme = 0                      ## screening constant
        self.enne = 0                      ## debye constant
        self.kappa = 0.                    ## (inverse) debye length
        self.gamma = np.empty(1)           ## array of angle aperture patch
        self.cosgamma = np.empty(1)        ## cosine of gamma
        self.delta = 0.                    ## interaction range
        self.ecc = np.empty(1)             ## array of eccentricity

        self.bjerrum = 0.7                 ##nm
        self.eps = 80.
        self.eps1 = 4.
        self.eps2 = 80.

        self.real_size = 200E-9                ## m real diameter of the colloid
        self.salt = 1                           ## in mM or mol/m^3
        self.zeta_pot = -20                      ## zeta potential in mV

        self.imbalance = 0
        self.real_units = False 
        self.e_charge = 1.602E-19           ## C
        self.permittivity = 8.987E9 #*1E18    ## N*m^2/C^2
        self.temperature = 293.15              ## K
        self.kB = 1.380649E-23                  ## J/K
        self.avogadro = 6.02214E23

        self.bjerrum_real = self.e_charge**2*self.permittivity/(self.eps*self.kB*self.temperature)
        self.f_kBT = 1.
        #print("check bjerrum: %.5e vs %.5e\n" % (self.bjerrum_real, self.bjerrum))

        self.topo = []                      ##unit vectors defining the position of the charges/patches
        self.topo_backup = []
        self.sph_theta = []                ##theta and phi values corresponding to the unit vectors given in topo
        self.sph_phi = []
        self.default_topo = False
        self.doubleint = False

        self.lmax = 30                      ###highest lmax = 80
        
        self.sp_zero = -1.
        self.box = np.asarray([100.,100.,100.])
        self.max_eff_pot = 0.                  

        self.path_N = 100
        self.path_max = []
        self.path_min = []
        self.path_displ = 1.
        self.thend = 0
        self.fpath_mf = 'dummy.dat'; self.fpath_cg = 'dummy.dat';
        self.plot = _plt                    ##

        self.plist_part = ['nr_patches', 'same_patch', 'salt', 'colloid_charge', 'patch_charge', 'patch_sigma', 'screening_constant', 'debye_constant', 'kappa', 'gamma', 'delta', 'eccentricity']
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
            except Exception as e:
                print(e); exit(1)
            
            name = name.strip()
            var = var.strip("\n"); var = var.strip(" ") 

            if name[0] == '#':
                continue
            if name[0] == ' ':
                continue

            ##myconvert(self.IPCdict[name], var)
            
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
            elif name == 'salt':
                self.salt = float(var)
            elif name == 'zeta_pot':
                self.zeta_pot = float(var)
            elif name == 'charge_imbalance':
                self.imbalance = float(var)
            elif name == 'energy_units':
                if var == 'False':
                    self.real_units = False
                elif var == 'True':
                    self.real_units = True
                else:
                    raise ValueError
            elif name == 'system_type':
                self.systype = str(var)
            elif name == 'cg_type':
                self.cgtype = str(var)
            elif name == 'pathway_distance':
                self.path_displ = float(var)
            elif name == 'plot':
                self.plot = var
            else :
                print("entry:", name, "unkown")
                exit(1)
            
            if name in self.plist_part and name not in plist:
                plist.extend([name])

        f.close()

        self.f_kBT=self.e_charge/(self.kB*self.temperature)   ##self.bjerrum_real*self.eps/self.real_size

        print("System studied:", self.systype)
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

        if len(plist2) < 3:
            print("Error: too few parameters", plist2); exit(1)

        if not ('debye_constant' in plist2 or 'screening_constant' in plist2 or 'kappa' in plist2 or 'salt' in plist2) or not ('patch_sigma' in plist2 or 'gamma' in plist2 or 'eccentricity' in plist2):
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
                self.delta = self.enne/self.kappa;    
                self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc;
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)
 
            elif 'gamma' in plist2 and 'screening_constant' in plist2 and 'debye_constant' in plist2:
                self.kappa = self.emme/self.sigma_core
                self.delta = self.enne/self.kappa; 
                self.cosgamma = np.cos(self.gamma)
                self.ecc = (self.delta**2+self.sigma_core*self.delta)/(self.delta+2.*self.sigma_core-2*self.sigma_core*self.cosgamma)
                self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc;
            
            elif 'eccentricity' in plist2 and 'salt' in plist2 and 'debye_constant' in plist2:
                self.kappa = calc_kappa(self.salt, self.bjerrum_real, self.avogadro, self.real_size)
                self.emme = self.kappa*self.sigma_core
                self.delta = self.enne/self.kappa
                self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc;
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            else:
                print("not supported"); exit(1)

        if 'colloid_charge' not in plist:
            print('colloid charge not present!! computing colloidal charge...')
            psi0 = 1E-3*self.zeta_pot*self.e_charge/(self.kB*self.temperature)
            self.colloid_charge = calc_charge(self.bjerrum_real, self.real_size, self.emme, psi0, 1, 0)
        if 'patch_charge' not in plist: 
            if self.samepatch == True:
                print('patch charge not present!! computing patch charge...')
                psi0 = 1E-3*self.zeta_pot*self.e_charge/(self.kB*self.temperature)
                temp = calc_charge(self.bjerrum_real, self.real_size, self.emme, -1.*psi0, self.npatch, self.imbalance)
                safecopy(temp, self.patch_charge)
            else:
                print('patch_charge required if patches are different')
                exit(1)

            #self.ecc = self.delta/2. + self.sigma_core - self.patch_sigma
            #self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.patch_sigma**2)/(2.*np.dot(self.patch_sigma,self.ecc))
            #self.gamma = np.arccos(self.cosgamma)

            #self.ecc = self.delta*(self.delta/4. + self.sigma_core)/(self.delta + 2.*self.sigma_core*(1.-self.cosgamma))
            #self.patch_sigma = ((2.*self.sigma_core**2 + self.sigma_core*self.delta)*(1. - self.cosgamma) + self.delta**2/4.)/( self.delta + 2.*self.sigma_core*(1.-self.cosgamma) )

            #self.patch_sigma = self.delta/2. + self.sigma_core - self.ecc

        if len(plist2) > 3:
            print("too many inputs; not supported"); exit(1)

        #####
        if not np.all(self.patch_charge == self.patch_charge[0]) or not np.all(self.patch_sigma == self.patch_sigma[0]) or not np.all(self.ecc == self.ecc[0]):
            print("check:", self.patch_charge, self.patch_sigma, self.ecc, self.gamma)
            self.samepatch = False
        if np.all(self.patch_charge == self.patch_charge[0]) and np.all(self.patch_sigma == self.patch_sigma[0]) and np.all(self.ecc == self.ecc[0]):
            self.samepatch = True



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
        f.write("delta = %.8f\n" % self.delta)
        f.write("lmax = %d\n" % self.lmax)
        f.write("real colloid size = %.5e\n" % self.real_size)

        f.close()

