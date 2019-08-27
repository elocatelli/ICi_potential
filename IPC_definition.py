import numpy as np
import string
import os.path

try:
    _mod = 'matplotlib'; import matplotlib.pyplot as plt
    _plt = True
except Exception as e:
    print e; _plt = False

from utils import safecopy
import IPC_DH_potentials
import IPC_effective_potentials
import IPC_coarse_graining

class IPC(IPC_DH_potentials.Mixin, IPC_effective_potentials.Mixin, IPC_coarse_graining.Mixin):

    def __init__(self) :
        self.toread = 'dum'
        self.npatch = 2 		    ##nr patches
        self.samepatch = True		    ## patches are all the same or different
        self.colloid_charge = 0.	    ## central colloid charge
        self.patch_charge = np.empty(1)	    ## array for patch charges
        self.sigma = 1
        self.sigma_core = 0.5 		    ## sigma_core remember! sigma_core/patch is the radius not the diameter
        self.sigma_patch = np.empty(1)      ## array of sigma_patch 
        self.emme = 0			    ## screening constant	
        self.enne = 0     		    ## debye constant
        self.kappa = 0.     		    ## (inverse) debye length 
        self.gamma = np.empty(1)	    ## array of angle aperture patch
        self.cosgamma = np.empty(1)		    ## cosine of gamma
        self.delta = 0. 		    ## interaction range
        self.ecc = np.empty(1)	     	    ## array of eccentricity
       
        self.eps = 80.
       
        self.topofile = 'dum'
        self.topo = []                      ##unit vectors defining the position of the charges/patches

        self.lmax = 80

        self.is_wall = False	            ##is there a wall? 
               
        self.plot = _plt			##
        
        self.plist_part = ['nr_patches', 'same_patch', 'colloid_charge', 'patch_charge', 'sigma_patch', 'screening_constant', 'debye_constant', 'kappa', 'gamma', 'delta', 'eccentricity']  ## list of parameters for the particle
        self.plist_wall = ['wall_charge'] 
        self.plist_misc = ['plot']

        #################################
    
        self.DH_potential = []
        self.effective_potential = []
        self.cg_potential = []


    def set_params(self, infname):

        self.toread = infname
        f = open(self.toread, 'r') 
        plist = list()

        nr_params = 1
        for line in f:

            elems = string.split(line.strip("\n"), "=")
            if len(elems) != 2:
                print "entry: ", line, "is too short or too long"
                exit(1)
            
            for i in range(len(elems)):
                elems[i] = elems[i].strip()
          
            print elems

            if elems[0] == 'nr_patches':
                self.npatch = int(elems[1])
                self.patch_charge = np.empty((self.npatch,1)); self.sigma_patch = np.empty((self.npatch,1))
                self.ecc = np.empty((self.npatch,1)); self.gamma = np.empty((self.npatch,1))
            elif elems[0] == 'same_patch':
                if elems[1] == 'False':
                    self.samepatch = False
                elif elems[1] == 'True':
                    self.samepatch = True
                else:
                    raise ValueError
            elif elems[0] == 'patch_charge':
                temp = np.asarray(string.split(elems[1], ', '), dtype = float)
                safecopy(temp, self.patch_charge)
            elif elems[0] == 'colloid_charge':
                self.colloid_charge = float(elems[1])
            #elif elems[0] == 'sigma_core':
            #    self.sigma_core = float(elems[1])
            elif elems[0] == 'sigma_patch':
                temp = np.asarray(string.split(elems[1], ', '), dtype = float)
                safecopy(temp, self.sigma_patch)
            elif elems[0] == 'debye_constant':
                self.enne = float(elems[1]) 
            elif elems[0] == 'screening_constant':
                self.emme = float(elems[1])    
            elif elems[0] == 'kappa':
                self.kappa = float(elems[1])
            elif elems[0] == 'eccentricity':
                temp = np.asarray(string.split(elems[1], ', '), dtype = float)
                safecopy(temp, self.ecc)
            elif elems[0] == 'gamma':
                temp = np.asarray(string.strip(elems[1], ', '), dtype = float)
                safecopy(temp, self.gamma)
                self.cosgamma = np.cos(self.gamma)
            elif elems[0] == 'delta':
                self.delta = float(elems[1])
            elif elems[0] == 'plot':
                self.plot = elems[1]
            else :
                print "entry:", elems[0], "unkown" 
                exit(1)
                
            if elems[0] in self.plist_part and elems[0] not in plist:
	        print elems[0]
                plist.extend([elems[0]])

        f.close()

        print plist

        ##check for errors
        if 'nr_patches' not in plist:
	    print "number of patch not found, defaulting to 2!"
        if 'same_patch' not in plist:
	    print "same_patch not specified, defaulting to True"
        if 'colloid_charge' not in plist:
	    print 'colloid charge required'
	    exit(1)
        if 'patch_charge' not in plist:
	    print 'patch_charge required'
	    exit(1)
      
	plist2 = plist[:]
        try:
            plist2.remove('nr_patches'); 
        except Exception as e:
            print e; print "nr patches was not specified"
        try:
            plist2.remove('same_patch')
        except Exception as e:
            print e; print "same_patch was not specified"
        
        plist2.remove('colloid_charge'); 
        plist2.remove('patch_charge'); 
        	
	if len(plist2) < 3:
            print "Error: too few parameters ", plist2; exit(1)
	
        if not ('debye_constant' in plist2 or 'screening_constant' in plist2 or 'kappa' in plist2) or not ('sigma_patch' in plist2 or 'gamma' in plist2 or 'eccentricity' in plist2):
            print "at least one between: debye_constant, screening_constant and kappa and one between: sigma_patch, gamma, eccentricity are needed"
            exit(1)

        if len(plist2) == 3:

            if 'sigma_patch' in plist2 and 'debye_constant' in plist2 and 'screening_constant' in plist2 :
                self.kappa = self.emme/self.sigma
                self.delta = self.enne/self.kappa
                self.ecc = self.delta/2. + self.sigma_core - self.sigma_patch
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.sigma_patch**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            elif 'sigma_patch' in plist2 and 'debye_constant' in plist2 and 'kappa' in plist2 :
                print "not supported"; exit(1)

            elif 'sigma_patch' in plist2 and 'screening_constant' in plist2 and 'gamma' in plist2 :
                print "not supported"; exit(1)

            elif 'sigma_patch' in plist2 and 'screening_constant' in plist2 and 'delta' in plist2 :
                self.kappa = self.emme/self.sigma
                self.enne = self.delta/self.kappa
                self.ecc = self.delta/2. + self.sigma_core - self.sigma_patch
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.sigma_patch**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            elif 'sigma_patch' in plist2 and 'screening_constant' in plist2 and 'eccentricity' in plist2 :
                self.kappa = self.emme/self.sigma
                self.delta = 2*(self.ecc - self.sigma_core + self.sigma_patch) 
                self.enne = self.delta*self.kappa
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.sigma_patch**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma)

            elif 'eccentricity' in plist2 and 'screening_constant' in plist2 and 'debye_constant' in plist2:
                print "should be here", self.ecc, self.emme, self.enne
                self.kappa = self.emme/self.sigma
                self.delta = self.enne/self.kappa
                self.sigma_patch = self.delta/2. + self.sigma_core - self.ecc; print self.delta, self.sigma_patch
                self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.sigma_patch**2)/(2.*self.sigma_core*self.ecc)
                self.gamma = np.arccos(self.cosgamma); print self.gamma

            else:
                print "not supported"; exit(1)

                #self.ecc = self.delta/2. + self.sigma_core - self.sigma_patch
                #self.cosgamma = (self.sigma_core**2 + self.ecc**2 - self.sigma_patch**2)/(2.*np.dot(self.sigma_patch,self.ecc))
                #self.gamma = np.arccos(self.cosgamma)

                #self.ecc = self.delta*(self.delta/4. + self.sigma_core)/(self.delta + 2.*self.sigma_core*(1.-self.cosgamma))
		#self.sigma_patch = ((2.*self.sigma_core**2 + self.sigma_core*self.delta)*(1. - self.cosgamma) + self.delta**2/4.)/( self.delta + 2.*self.sigma_core*(1.-self.cosgamma) )

                #self.sigma_patch = self.delta/2. + self.sigma_core - self.ecc

        if len(plist2) > 3:
                print "too many inputs; not supported"; exit(1)

    def set_charge_topology(self, infname):
        
        self.topofile = infname

        if not os.path.isfile(self.topofile):
            
            print "topology is not defined"

            if self.npatch == 2:
                print "using the default for %d patches: polar patches" % self.npatch
                self.topo = [[0., 0., 1], [0., 0., -1.]]
            if self.npatch == 3:
                print "using the default for %d patches: triangle patches" % self.npatch
                self.topo = [[0., 0., 1], [0.86602, 0., -0.5], [-0.86602, 0., -0.5]]
        else:
            f = open(self.topofile, 'r')
        
            for lines in f:
                self.topo.append([float(x) for x in string.split(lines.strip("\n"),' ')])
            
            f.close()

        for el in self.topo:
            if np.fabs(np.linalg.norm(np.asarray(el)) - 1.) > 1E-5:
                print "vector ", el, "is not a unit vector, err:", np.linalg.norm(np.asarray(el)) - 1. 
                exit()

        print self.topo

        if len(self.topo) != self.npatch:
            print "topology is wrong, too many or too less unit vectors given"
            

    def check_params(self, outfname):
		
        f = open(outfname,'w')
	f.write("nr_patches = %d\n" % self.npatch) 
        f.write("same_patch = %s\n" % str(self.samepatch))
        f.write("patch_charge = %s\n" % (" ".join(str(x) for x in self.patch_charge)) )
        f.write("colloid_charge = %.3f\n" % self.colloid_charge)
        f.write("sigma_patch = %s\n" % (" ".join(str(x) for x in self.sigma_patch) ) )
        f.write("debye_constant = %.3f\n" % self.enne )
        f.write("screening_constant = %.3f\n" % self.emme )
        f.write("kappa = %.3f\n" % self.kappa)
        f.write("eccentricity = %s\n" % (" ".join(str(x) for x in self.ecc) ) )
        f.write("gamma = %s\n" % (" ".join(str(x*180/np.pi) for x in self.gamma) ) )
        f.write("delta = %.3f\n" % self.delta)

        f.close()

