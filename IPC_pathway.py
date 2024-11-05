import numpy as np
from utils import *

from IPC_coarse_graining import compute_dum_energy, compute_dum_energy_CG1
from IPC_effective_potentials import calc_dists_angles, calc_dists_angles_2p, yukawa, MinD, PImg

def get_topo(_input, box):

    out = []
    temp = MinD(_input[1] - _input[0], box)
    out.extend([temp/np.linalg.norm(temp)])
    if _input.shape[0] == 3:
        temp = MinD(_input[2] - _input[0], box)
        out.extend([temp/np.linalg.norm(temp)])

    return out

def neg(_in):
    return [-1*x for x in _in]

def skip_a_line(fname):
    ff = open(fname, "a"); ff.write("\n"); ff.close(); 

class Mixin:

    def pathway_reset(self,fname_mf, fname_cg):

        self.thend = 0
        
        self.fpath_mf = fname_mf; self.fname_cg = fname_cg
        ff = open(self.fpath_mf, "w"); ff.close() 
        gg = open(self.fpath_cg, "w"); gg.close()

    def get_pathway_point(self,  instructions, th):

        sigma_core2 = self.sigma_core + self.delta/2.
        ipc_type = 'numeric'
        start_o = instructions['orientation'];
        shift_v = instructions['shift_vector']; 
        _displ = instructions['pathway_distance']
        rotaxis_p = instructions['rotation_axis_patches']; th_p = instructions['rotation_angle_patches']
        rotaxis_c = instructions['rotation_axis_colloid']; th_c = instructions['rotation_angle_colloid']

        if th_p == 0 and th_c != 0:
            th_2 = th; th_1 = th_p 
        elif th_p != 0 and th_c == 0:
            th_2 = th_c; th_1 = th
        else:
            print("double rotation or no rotation not allowed\n"); exit(1)
        self.max_eff_pot = 1
        _part1 = self.generate_particle(start_o[0])[:,:3]
        _part2_0 = self.generate_particle(start_o[1])[:,:3]
        _part2_1 = rotate_patches(_part2_0, rotaxis_p, th_1); _part2_1[:,:3] += _displ*shift_v
        _part2 = rotate_part(_part2_1, rotaxis_c, th_2)
        VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], ipc_type)
        
        return VV[0,2]/self.max_eff_pot

    def print_ep_pp_sep(self):
        
        xx = np.asarray([1,0,0]); yy = np.asarray([0,1,0]); zz = np.asarray([0,0,1])

        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]
        
        E_EP = self.get_pathway_point(self.build_instructions(EP,xx, -yy, 1, yy, 0), 0.)
        E_PP = self.get_pathway_point(self.build_instructions(PP,xx, yy, 0, -yy, 1), 0.)
        E_SEP = self.get_pathway_point(self.build_instructions(EP,zz, yy, 0, yy, 1), np.pi/4.)
        
        f = open(self.IPCdict['folder']+"/energies_ep_pp_sep.dat", 'w')
        f.write("%.8e %.8e %.8e\n" % (E_EP, E_PP, E_SEP))
        f.close()

    def print_pp_janus(self):

        xx = np.asarray([1,0,0]); yy = np.asarray([0,1,0]); zz = np.asarray([0,0,1])

        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[3]; EE2 = to_orient[4]; EP = to_orient[5]; EP2 = to_orient[6]

        E_PP = self.get_pathway_point(self.build_instructions(EE,zz, yy, 1, yy, 0), 0.)
        E_PP2 = self.get_pathway_point(self.build_instructions(EE2,zz, yy, 1, yy, 0), 0.)

        f = open(self.IPCdict['folder']+"/energies_pp_janus.dat", 'w')
        f.write("%.8e %.8e\n" % (E_PP, E_PP2))
        f.close()


    def pathway_block(self, instructions, rth=np.pi/2.):

        sigma_core2 = self.sigma_core + self.delta/2.
        ipc_type = 'numeric' 
        start_o = instructions['orientation']; 
        shift_v = instructions['shift_vector']; print("start", start_o, shift_v)
        _displ = instructions['pathway_distance']
        rotaxis_p = instructions['rotation_axis_patches']; th_p = instructions['rotation_angle_patches']
        rotaxis_c = instructions['rotation_axis_colloid']; th_c = instructions['rotation_angle_colloid']

        ff = open(self.fpath_mf, "a"); gg = open(self.fpath_cg, "a")
        for th in np.linspace(0, rth, self.path_N):
            if th_p == 0 and th_c != 0:
                th_2 = th; th_1 = th_p
            elif th_p != 0 and th_c == 0:
                th_2 = th_c; th_1 = th
            else:
                print("double rotation or no rotation not allowed\n"); exit(1)
            self.max_eff_pot = 1
            ff.write("%.5e " % (th+self.thend)); gg.write("%.5e " % (th+self.thend));
            _part1 = self.generate_particle(start_o[0])[:,:3]
            _part2_0 = self.generate_particle(start_o[1])[:,:3]
            _part2_1 = rotate_patches(_part2_0, rotaxis_p, th_1); _part2_1[:,:3] += _displ*shift_v
            _part2 = rotate_part(_part2_1, rotaxis_c, th_2)
            VV = self.compute_effective_potential(_part1, _part2, [get_topo(_part1,self.box),get_topo(_part2,self.box)], ipc_type)
            ff.write("%.5e\n" % (VV[0,2]/self.max_eff_pot))
            ##ff.write("%.5e\n" % (self.kappa*self.path_displ*np.exp(self.kappa*self.path_displ)*VV[0,2]/self.max_eff_pot))
            ########################################################
            #if self.cgtype == 'cg1':
            #    CG = compute_dum_energy_CG1(_part1, _part2, sigma_core2, self.patch_sigma, self.samepatch, self.box)
            #else:
            #    p = [self.sigma, self.delta]
            #    for v in self.ecc:
            #        p.extend([v])
            #    CG = compute_dum_energy(_part1, _part2, p, self.samepatch, self.box)
            #gg.write("%.5e\n" % (np.dot(np.asarray(CG),self.u_cg)) )
      
        ##ff.write("\n")
        print("finish", [get_topo(_part1,self.box),get_topo(_part2,self.box)], _part2[0] )
        self.thend += rth
        ff.close(); gg.close();


    def build_instructions(self, start_o, shift_v, rotaxis_p, th_p, rotaxis_c, th_c):
        out = {}
        out['pathway_distance'] = self.path_displ; out['shift_vector'] = shift_v 
        out['orientation'] = start_o
        out['rotation_axis_patches'] = rotaxis_p; out['rotation_angle_patches'] = th_p
        out['rotation_axis_colloid'] = rotaxis_c; out['rotation_angle_colloid'] = th_c
        return out

    def pathway(self, fname_mf, fname_cg):

        if self.npatch == 1:
            self.pathway_janus(fname_mf, fname_cg)
        elif self.npatch == 2:
            self.pathway_polar(fname_mf, fname_cg)
        else:
            print("NOT ALLOWED"); exit(1)

    def pathway_ee_ep_ee(self, fname_mf, fname_cg):

        self.pathway_reset(fname_mf, fname_cg)
        xx = np.asarray([1,0,0]); yy = np.asarray([0,1,0]); zz = np.asarray([0,0,1])

        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]
        
        ## rotation 1: from EE to EP
        self.pathway_block(self.build_instructions(EE,xx, -yy, 1, yy, 0))
        ## rotation 2: from EP to EE
        self.pathway_block(self.build_instructions(EP,xx, yy, 1, yy, 0))
                
    def pathway_pp_ep_pp(self, fname_mf, fname_cg):

        self.pathway_reset(fname_mf, fname_cg)
        xx = np.asarray([1,0,0]); yy = np.asarray([0,1,0]); zz = np.asarray([0,0,1])

        to_orient = self.generate_orientations()
        PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]

        ## rotation 1: from PP to EP
        self.pathway_block(self.build_instructions(PP,xx, -yy, 1, yy, 0))
        ## rotation 2: from EP to PP
        self.pathway_block(self.build_instructions([PP[0],EE[1]],xx, yy, 1, yy, 0))


    def pathway_polar(self, fname_mf, fname_cg):

        self.pathway_reset(fname_mf, fname_cg)
        xx = np.asarray([1,0,0]); yy = np.asarray([0,1,0]); zz = np.asarray([0,0,1])

        to_orient = self.generate_orientations()
        if self.samepatch:
            PP = to_orient[0]; EE = to_orient[1]; EP = to_orient[2]
            ## rotation 1: from EP to EE
            self.pathway_block(self.build_instructions(EP,xx, yy, 1, yy, 0))
            ## rotation 2: from EE to PP rotating colloid around y
            self.pathway_block(self.build_instructions(EE,xx, yy, 0, yy, 1))
            ## rotation 3: from PP to EP rotating patches around y (EE+shift z)
            self.pathway_block(self.build_instructions(EE,zz, yy, 1, yy, 0))
            ## rotation 4: from EP to EP rotating colloid around y (EP+shift z)
            self.pathway_block(self.build_instructions(EP,zz, yy, 0, yy, 1))
            ## rotation 5: from EP to EP rotating colloid around z (EP+shift x)
            self.pathway_block(self.build_instructions(EP,xx, zz, 0, zz, 1))
            ## rotation 6: from perp-EP to PP rotating patches around x (EE+shift z)
            self.pathway_block(self.build_instructions([EE[0], [[0,1,0],[0,-1,0]]], xx, xx, 1, xx, 0))
        else:
            PP = to_orient[0]; EE = to_orient[3]; EE2 = to_orient[4]; EP = to_orient[5]; EP2 = to_orient[6]
            ## rotation 1: from EP to EE
            print("1"); self.pathway_block(self.build_instructions(EP,xx, -yy, 1, yy, 0))
            ## rotation 2: from EE to PP rotating colloid around y
            print("2"); self.pathway_block(self.build_instructions(EE,xx, yy, 0, -yy, 1))
            ## rotation 3: from PP to EP rotating patches around y (EE+shift z)
            print("3"); self.pathway_block(self.build_instructions(EE,zz, yy, 1, yy, 0))
            ## rotation 4: from EP to EP rotating colloid around y (EP+shift z)
            print("4"); self.pathway_block(self.build_instructions(EP,zz, yy, 0, yy, 1))
            ## rotation 5: from EP to EP rotating colloid around z (EP+shift x)
            print("5"); self.pathway_block(self.build_instructions(EP,xx, zz, 0, zz, 1))
            ## rotation 6: from perp-EP to PP rotating patches around x (EE+shift z)
            print("6"); #self.pathway_block(self.build_instructions([EE[0], [[0,1,0],[0,-1,0]]], xx, xx, 1, xx, 0))
            self.pathway_block(self.build_instructions(EP, yy, -yy, 1, xx, 0))
            ## rotation 7: from EP to EP
            print("7"); self.pathway_block(self.build_instructions(EE, yy, xx, 1, yy, 0))
            ## ROUND 2
            skip_a_line(self.fpath_mf); skip_a_line(self.fpath_cg)
            self.thend = 0
            ## rotation 8: from EP to EE2
            print("1"); self.pathway_block(self.build_instructions(EP,xx, yy, 1, yy, 0))
            ## rotation 9: from EE to PP rotating colloid around y
            print("2"); self.pathway_block(self.build_instructions(EE2,xx, yy, 0, -yy, 1))
            ## rotation 10: from PP to EP rotating patches around y (EE+shift z)
            print("3"); self.pathway_block(self.build_instructions(EE2,zz, yy, 1, yy, 0))
            ## rotation 11: from EP to EP rotating colloid around y (EP+shift z)
            print("4"); self.pathway_block(self.build_instructions(EP2,zz, -yy, 0, yy, 1))
            ## rotation 12: from EP to EP rotating colloid around z (EP+shift x)
            print("5"); self.pathway_block(self.build_instructions(EP2,xx, zz, 0, zz, 1))
            ## rotation 13: from perp-EP to PP rotating patches around x (EE+shift z)
            #self.pathway_block(self.build_instructions([EE[0], [[0,1,0],[0,-1,0]]], xx, -xx, 1, xx, 0))
            print("6"); self.pathway_block(self.build_instructions(EP2, yy, -yy, 1, xx, 0))
            ## rotation 14: from PP to EP
            print("7"); self.pathway_block(self.build_instructions(EE2, yy, xx, 1, yy, 0))
           
        

    def pathway_janus(self, fname_mf, fname_cg):

        self.pathway_reset(fname_mf, fname_cg) 
        to_orient = self.generate_orientations()
        PP = to_orient[0]; EPB = to_orient[1]; EE = to_orient[2];
        EEB1 = to_orient[3]; EP = to_orient[4]; EEB2 = to_orient[5]
        ##PP -> <-     EPB ^ <-      EE <- ->       EEB1 ^ ^       EP <- <-        EEB2 ^ ->
        xx = np.asarray([1,0,0]); yy = np.asarray([0,1,0]); zz = np.asarray([0,0,1])

        #print("step 1")
        #self.pathway_block(self.build_instructions(EPB,xx, yy, 0, -yy, 1), rth=2*np.pi)
        #print("step 2")
        #self.pathway_block(self.build_instructions(EPB,xx, yy, 0, -zz, 1), rth=2*np.pi)
        #print("step 3")
        #self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]], zz, yy, 0, -yy, 1), rth=2*np.pi)
        #print("step 4")
        #self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]], zz, yy, 0, -xx, 1), rth=2*np.pi)

        ## rotation 1: from BB to BE
        print("step 1")
        self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]], -zz, -yy, 1, yy, 0))
        ## rotation 2: from BE to BP 
        print("step 2")
        self.pathway_block(self.build_instructions(EEB2, -zz, -yy, 1, yy, 0))
        ## rotation 3: from BP to sEE rotating patches around y (shift z)
        print("step 3")
        self.pathway_block(self.build_instructions(EEB1, -zz, yy, 0, -yy, 1))
        ## rotation 4: from sEE to tEE
        print("step 4")
        self.pathway_block(self.build_instructions(EEB1,xx, xx, 1, xx, 0))
        ## rotation 5: from tEE to EP
        print("step 5")
        self.pathway_block(self.build_instructions([EEB1[0],[[0,-1,0]]],xx, -zz, 1, zz, 0))
        ## rotation 6: from EP to BE2
        print("step 6")
        self.pathway_block(self.build_instructions(EPB,xx, yy, 0, -yy, 1))
        ## rotation 7: from BE2 ro PP
        print("step 7")
        self.pathway_block(self.build_instructions(EPB,zz, -yy, 1, yy, 0))
        ## rotation 8 from PP to aEE
        print("step 8")
        self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]],zz, yy, 0, -yy, 1))
        ## rotation 9 from aEE to BB
        print("step 9")
        self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]],-xx, yy, 0, -yy, 1))
        
        '''
        ## rotation 6: from EP to aEE
        print("step 6")
        self.pathway_block(self.build_instructions(EPB, xx, -yy, 1, xx, 0))
        ## rotation 7: from aEE to PP
        print("step 7")
        self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]], xx, yy, 0, -yy, 1))
        ## rotation 8: from PP2 to aEE
        print("step 8")
        self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]], zz, yy, 0, -yy, 1))
        ## rotation 9: from aEE to BB
        print("step 9")
        self.pathway_block(self.build_instructions([EEB1[0],[neg(EEB1[1][0])]], -xx, yy, 0, -yy, 1))
        '''
