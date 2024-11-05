#!/usr/bin/env python
import time
import sys

try:
    _mod = 'numpy'; import numpy as np
    _mod = 'scipy'; import scipy as sp ##improve import
except:
    print("necessary module ", _mod," missing"); exit(1)

_plt = False
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

    sstart = time.time()
   
    if myIPC.systype == 'general':
        #myIPC.print_potential('numeric', nph=100)
        myIPC.do_effective_potential('general', 'numeric', Np=100)
        myIPC.pathway_pp_ep_pp(myIPC.IPCdict['folder']+'/ipc_pp_ep_pp.dat', myIPC.IPCdict['folder']+'/test_pathway_cg.dat')
        myIPC.pathway_ee_ep_ee(myIPC.IPCdict['folder']+'/ipc_ee_ep_ee.dat', myIPC.IPCdict['folder']+'/test_pathway_cg.dat')
        ##myIPC.do_coarse_graining_max('general', 'numeric')
        #myIPC.rotation_pathway('numeric', myIPC.IPCdict['folder']+'/rotation_pathway_ipc.dat', myIPC.IPCdict['folder']+'/rotation_pathway_cg.dat')
        #myIPC.pathway(myIPC.IPCdict['folder']+'/test_pathway_ipc.dat', myIPC.IPCdict['folder']+'/test_pathway_cg.dat')
        #myIPC.print_ep_pp_sep()
        #myIPC.print_pp_janus()
        #myIPC.find_pathway_extrema(myIPC.IPCdict['folder']+'/rotation_pathway_ipc.dat')
        #myIPC.potential_at_pathway_extrema(myIPC.IPCdict['folder']+'/radial_pathway_extrema.dat')
    elif myIPC.systype == 'yukawa':
        myIPC.print_potential('yukawa', nph=100)
        myIPC.do_effective_potential('yukawa', 'yukawa', Np=1) 
        myIPC.do_coarse_graining_max('yukawa', 'yukawa')
        myIPC.rotation_pathway('yukawa', myIPC.IPCdict['folder']+'/rotation_pathway_ipc_yukawa.dat', myIPC.IPCdict['folder']+'/rotation_pathway_cg_yukawa.dat')
    else:
        print(myIPC.systype, "not supported"); exit(1)


    #sstart = time.time()
    #myIPC.effective_potential_plot_angles('general', 'numeric', [1,0,0], myIPC.IPCdict['folder']+'/eff_pot_omega_x.dat', Np=100)
    eend = time.time(); print("done, elapsed time ", eend-sstart)
    '''
    data1 = np.loadtxt(myIPC.IPCdict['folder']+'/rotation_pathway_ipc.dat')
    data2 = np.loadtxt(myIPC.IPCdict['folder']+'/rotation_pathway_cg.dat')

    ff = open(myIPC.IPCdict['folder']+'/pathway_difference.dat', "w")
    ff.write("%.5f %.5f %.8e %.8e\n" % (myIPC.enne, myIPC.delta, np.sum(np.fabs(data1[:,1]-data2[:,1])), np.trapz(np.fabs(data1[:,1]-data2[:,1]),data1[:,0]) ) )
    ff.close()
    '''
    '''
    sstart = time.time()
    myIPC.effective_potential_plot_angles('general', 'numeric', [0,1,0], myIPC.IPCdict['folder']+'/eff_pot_omega_y.dat', Np=100)
    eend = time.time(); print("potential y done", eend-sstart)
    sstart = time.time()
    myIPC.effective_potential_plot_angles('general', 'numeric', [0,0,1], myIPC.IPCdict['folder']+'/eff_pot_omega_z.dat', Np=100)
    eend = time.time(); print("potential z done", eend-sstart)
    #myIPC.effective_potential_plot_angles('2patch', '2patch_analytic', [0,0,1], 'RES/eff_2patch_omega_z.dat')
    '''

    #myIPC.rotate_colloid_around_axis([0,1,0], myIPC.IPCdict['folder']+'/eff_pot_rotate_colloid_y.dat', myIPC.IPCdict['folder']+'potential_cg_rotate_colloid_y.dat')
    #myIPC.rotate_colloid_around_axis([0,0,1], myIPC.IPCdict['folder']+'/eff_pot_rotate_colloid_z.dat', myIPC.IPCdict['folder']+'potential_cg_rotate_colloid_z.dat')
    #myIPC.rotate_2patches([0,0,1], myIPC.IPCdict['folder']+'/eff_pot_rotate_2patches_z.dat', myIPC.IPCdict['folder']+'potential_cg_rotate_2patches_z.dat')
    
    #myIPC.do_coarse_graining_max('yukawa', 'yukawa')
    #myIPC.test_random_2P(1.0)

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
