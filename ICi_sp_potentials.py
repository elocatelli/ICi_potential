
import numpy as np
from scipy.special import eval_legendre, lpmv, kn, kv, iv, kvp 
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, bisect

sq2 = np.sqrt(2)

class Mixin:

    def pot_wrapper(self, th, r, phi, name):
        return self.analytic_funcdict[name](r, th, phi)[0]

    def get_theta0(self, name):
        return bisect(self.pot_wrapper, 0, 1.83, args=(1,0., name))

    def h_a(self, a, target, name):
        self.ecc = np.asarray([a,a])
        return self.get_theta0(name)-target

    def find_theta0(self, name, target):
        
        if self.samepatch == False or self.npatch != 2: 
            print("only polar ICis with equal patches are admissible"); exit(1)
        _store = self.ecc
        
        sol = root_scalar(self.h_a, bracket=[0.005,0.45], method='brentq', args=(target, name))
        mysol = sol.root
        np.savetxt(self.ICidict['folder']+'/new_abar.dat', np.asarray([mysol]), fmt='%.8f')


    def print_potential_surface(self, name, nth = 50, nph = 100):

        if self.npatch == 2 and self.default_topo:
             
            coord = [[ix,iy] for ix in [self.sigma_core] for iy in np.linspace(0,np.pi,nph) ]

            out = []
         
            for r, th in coord:
                out0 = self.analytic_funcdict[name](r, th, 0.)
                out.extend([((th)*180./np.pi), out0[0]])

            out = np.reshape((np.asarray(out)),(len(out)//2,2))
            np.savetxt(self.ICidict['folder']+"/single_particle_potential_outside.dat", out, fmt="%.12e")
            
            if nph > 10:
                g = interp1d(out[:,0], out[:,1], kind='cubic')
                try:
                    sol = root_scalar(g, bracket=[0,105.], method='brentq')
                    self.sp_zero = sol.root
                except:
                    print("no root for SP potential")
        else:
			
            print("special solutions not available for this topology"); exit(1)
				
    
    def print_potential_radial(self, name):
        coord = [[ix,iy,iz] for ix in np.linspace(0.5, 1.5, 100) for iy in [0] for iz in [0] ]
            
        out = []

        for r, th, ph in coord:
            out0 = self.analytic_funcdict[name](r, th, 0.)       
            out.extend([r, float(out0)])

        out = np.reshape(np.asarray(out).astype('float64'),(int(len(out)/2),2))
        
        np.savetxt(self.ICidict['folder']+"/potential_outside_radial.dat", out, fmt="%.5e")



    def potential_outside_2sympatches(self, r, th, phi):

        sigma = self.sigma_core; Zp = self.patch_charge; Zc = self.colloid_charge
        a = self.ecc; kappa = self.kappa;
        
        out0 = ((Zc+Zp[0]+Zp[1])/self.eps)*(np.exp(self.kappa*sigma)/(1.+self.kappa*sigma))*(np.exp(-self.kappa*r)/r)
        out01 = 0.

        if self.samepatch==True:
            _vals = range(2,self.lmax,2)
        else:
            _vals = range(1,self.lmax)

        for l in _vals:
            out1 = ( Zp[0]*np.power(a[0]/sigma,l)*eval_legendre(l,np.cos(th)) + Zp[1]*np.power(a[1]/sigma,l)*eval_legendre(l,np.cos(np.pi-th))  )
            out01 += ((2*l+1.)*(kv(l+0.5,kappa*r)/kv(l+1.5,kappa*sigma)))*out1
        out01 *= (1.)/(self.kappa*sigma*np.sqrt(sigma*r)*self.eps)
        out0 += out01
        
        #if self.real_units:
        out0 *= (self.e_charge*self.permittivity/(self.real_size))

        return out0

    def potential_inside_2sympatches(self, r, th, phi):

        sigma = self.sigma_core; Zp = self.patch_charge; Zc = self.colloid_charge
        a = self.ecc; kappa = self.kappa;
        
        out0 = -(Zc+np.sum(Zp))*(kappa*sigma) + (Zc+np.sum(Zp))
        out1 = 0.

        for l in range(2,self.lmax,2):
            out1 =  ( 2*Zp[0]*np.power(a[0]/sigma,l)*eval_legendre(l,np.cos(th)) ) 
            out0 += ( (kv(l+0.5,kappa*sigma)/kv(l+1.5,kappa*sigma)) + 1 )*out1
        out0 *= (1.)/(sigma*self.eps)

        if self.real_units:
            out0 *= (self.e_charge*self.permittivity/(self.real_size))
        
        return out0
