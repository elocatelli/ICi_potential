#!/usr/bin/env python
import time
import sys

try:
    _mod = 'numpy'; import numpy as np
    _mod = 'scipy'; import scipy as sp ##improve import
except:
    print("necessary module ", _mod," missing"); exit(1)

from ICi_definition import ICi

def main():

    myICi = ICi(); 
    #attrs = vars(myICi); print ', '.join("%s: %s" % item for item in attrs.items())
    
    i = 1
    for _opts in ('folder', 'toread', 'topofile'):
        try:
            temp = sys.argv[i]
            if temp == 'help':
                print("usage is: python3 "+sys.argv[0]+" (optional parameters) folder_name parameter_file topology_file")
                exit(1)
            else:
                myICi.ICidict[_opts] = temp  
            i += 1
        except Exception as e:
            break

    myICi.set_params()
    myICi.set_charge_topology()
    myICi.check_params()

    sstart = time.time()
  
    myICi.print_potential_surface('2patch', nph=100)
    myICi.do_effective_potential('2patch', '2patch', Np=1)
    myICi.pathway(myICi.ICidict['folder']+'/test_pathway_ipc.dat')

    #myICi.pathway_pp_ep_pp(myICi.ICidict['folder']+'/ipc_pp_ep_pp.dat', myICi.ICidict['folder']+'/test_pathway_cg.dat')
    #myICi.pathway_ee_ep_ee(myICi.ICidict['folder']+'/ipc_ee_ep_ee.dat', myICi.ICidict['folder']+'/test_pathway_cg.dat')
    
    #myICi.print_ep_pp_sep()
    #myICi.print_pp_janus()

    eend = time.time(); print("done, elapsed time ", eend-sstart)


if __name__=='__main__':
    main()
