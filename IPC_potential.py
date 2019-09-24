#!/usr/bin/env python

import sys
import string

try:
    _mod = 'numpy'; import numpy as np
    _mod = 'scipy'; import scipy as sp ##improve import
except:
    print "necessary module ", _mod," missing"; exit(1)

try:
    _mod = 'matplotlib'; import matplotlib.pyplot as plt
    _plt = True
except Exception as e:
    print e; _plt = False
    #print "module ", _mod," not found; not using"; no_plt = True

from IPC_definition import IPC 
from print_functions import *


def main():    
        
    
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        print "no help yet"

    myIPC = IPC(); #attrs = vars(myIPC); print ', '.join("%s: %s" % item for item in attrs.items())
    myIPC.set_params('params.dat')
    myIPC.set_charge_topology('topology.dat')
    myIPC.check_params('parameters_check.dat')

    test = myIPC.print_potential('analytic')
    np.savetxt("RES/test_potential.dat", test, fmt="%.5e")

    test_y = myIPC.print_potential('yukawa')
    np.savetxt("RES/test_potential_y.dat", test_y, fmt="%.5e")

    test = myIPC.print_potential('numerical'); 
    np.savetxt("RES/test_potential_general.dat", test, fmt="%.5e")

    myIPC.effective_potentials_yukawa()
    ##myIPC.effective_potential_2patches()
    np.savetxt("RES/effective_pot.dat", myIPC.effective_potential, fmt="%.5e")

    #test = effective_potentials_yukawa(myIPC.colloid_charge, myIPC.patch_charge, myIPC.kappa, myIPC.sigma_core, myIPC.ecc, myIPC.eps)
    #np.savetxt("RES/effective_pot_y.dat", test, fmt="%.5f")

    coeff = myIPC.coarse_graining_twopatches_max()
    np.savetxt("RES/cg_potentials.dat", myIPC.cg_potential, fmt="%.5e")

    _testp = myIPC.generate_particle(20.)

    print_lorenzo(_testp, myIPC.npatch, myIPC.samepatch, myIPC.delta, 20., "ipc4.dat") 

    

if __name__=='__main__':
    main()

