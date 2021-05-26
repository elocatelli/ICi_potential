#!/usr/bin/env python

import sys

try:
    _mod = 'numpy'; import numpy as np
    _mod = 'scipy'; import scipy as sp ##improve import
except:
    print("necessary module ", _mod," missing"); exit(1)

try:
    _mod = 'matplotlib'; import matplotlib.pyplot as plt
    _plt = True
except Exception as e:
    print(e); _plt = False
    #print "module ", _mod," not found; not using"; no_plt = True

from IPC_definition import IPC
from print_functions import *


def main():

    myIPC = IPC(); #attrs = vars(myIPC); print ', '.join("%s: %s" % item for item in attrs.items())
    
    i = 1
    for _opts in ('folder', 'toread', 'topofile'):
        try:
            temp = sys.argv[i]
            if temp == 'help':
                print("usage is: python3 "+sys.argv[0]+" (optional parameters) folder_name parameter_file topology_file")
                exit(1)
            else:
                myIPC.IPCdict[_opts] = temp  
            i += 1
        except Exception as e:
            break

    myIPC.set_params()
    myIPC.set_charge_topology()
    myIPC.check_params()

    #myIPC.print_potential('numeric')
    
    #myIPC.effective_potential_store('general', 'numeric', Np=1)
    #np.savetxt(myIPC.IPCdict['folder']+"/effective_potential_general.dat", myIPC.effective_potential, fmt="%.5e")
    #print("effective potential done")
    #myIPC.effective_potential_plot_angles('general', 'numeric', [1,0,0], myIPC.IPCdict['folder']+'/eff_pot_omega_x.dat')
    myIPC.effective_potential_plot_angles('general', 'numeric', [0,1,0], myIPC.IPCdict['folder']+'/eff_pot_omega_y.dat')
    myIPC.effective_potential_plot_angles('general', 'numeric', [0,0,1], myIPC.IPCdict['folder']+'/eff_pot_omega_z.dat')
    #myIPC.effective_potential_plot_angles('2patch', '2patch_analytic', [0,0,1], 'RES/eff_2patch_omega_z.dat')
    
    myIPC.do_coarse_graining_max()

    #if myIPC.npatch == 2 and myIPC.default_topo:

        #myIPC.effective_potential_store('2patch', 'yukawa')
        #np.savetxt("RES/effective_pot_yukawa.dat", myIPC.effective_potential, fmt="%.5e")

        #myIPC.effective_potential_store('2patch', '2patch_analytic', Np=1)
        #np.savetxt(myIPC.IPCdict['folder']+"/effective_pot_analytic.dat", myIPC.effective_potential, fmt="%.5e")

        #myIPC.effective_potential_store('yukawa', 'bob')
        #myIPC.effective_potential[:,1:] /= np.fabs(myIPC.effective_potential[0,3])
        #np.savetxt("RES/effective_pot_yukawa_2.dat", myIPC.effective_potential, fmt="%.5e")

        #myIPC.effective_potential_plot_angles('2patch', '2patch_analytic', 'RES/eff_pot_omega_analytic.dat')
        #myIPC.effective_potential_plot_angles('2patch', 'yukawa', 'RES/eff_pot_omega_yukawa.dat')

        #coeff = myIPC.coarse_graining_twopatches_max()
        #np.savetxt("RES/cg_potentials.dat", myIPC.cg_potential, fmt="%.5e")

    ##_testp = myIPC.generate_particle([10.,10.,10.])
    ##print_lorenzo(_testp, myIPC.npatch, myIPC.samepatch, myIPC.delta, 20., "ipc4.dat")


if __name__=='__main__':
    main()
