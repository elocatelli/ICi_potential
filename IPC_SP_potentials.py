### compute analytical solution as in chapter

import numpy as np
from scipy.special import eval_legendre, lpmv, kn, kv, iv, kvp, sph_harm
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, bisect
from math import factorial

_plt = False
try:
    _mod = 'matplotlib'; import matplotlib.pyplot as plt
    _plt = True
    from print_functions import print_potential_colormap
except Exception as e:
    print(e); _plt = False
    print("module ", _mod," not found; not using")

sq2 = np.sqrt(2)

def real_sph_harm(th, ph, l, m):

    out = 0.

    if m < 0:
        m1 = np.abs(m); f1 = factorial(l-m1); f2 = factorial(l+m1)
        pl = lpmv(m1,l,np.cos(th))
        out = sq2*np.power(-1,m1)*np.sqrt((2*l+1)*f1/(4*np.pi*f2))*pl*np.sin(m1*ph)
    elif m == 0:
        out = np.sqrt((2*l+1)/(4*np.pi))*lpmv(0,l,np.cos(th))
    else:
        f1 = factorial(l-m); f2 = factorial(l+m)
        pl = lpmv(m,l,np.cos(th))
        out = sq2*np.power(-1,m)*np.sqrt((2*l+1)*f1/(4*np.pi*f2))*pl*np.cos(m*ph)

    return out

class Mixin:

    def pot_wrapper(self, th, r, phi, name):
        return self.analytic_funcdict[name](r, th, phi)[0]

    def get_theta0(self, name):
        return bisect(self.pot_wrapper, 0, 1.83, args=(1,0., name))

    def h_a(self, a, target, name):
        self.ecc = np.asarray([a,a])
        return self.get_theta0(name)-target

    def find_theta0(self, name, target):
        
        if self.samepatch == False: 
            print("only polar IPCs with equal patches are admissible"); exit(1)
        _store = self.ecc
        
        sol = root_scalar(self.h_a, bracket=[0.005,0.45], method='brentq', args=(target, name))
        mysol = sol.root
        np.savetxt(self.IPCdict['folder']+'/new_a.dat', np.asarray([mysol]), fmt='%.8f')


    def print_potential(self, name, nth = 50, nph = 100):

        if self.npatch == 2 and self.default_topo:
             
            coord = [[ix,iy] for ix in [self.sigma_core] for iy in np.linspace(0,np.pi,nph) ]

            out = []
         
            for r, th in coord:
                out0 = self.analytic_funcdict[name](r, th, 0.)
                out.extend([((th)*180./np.pi), out0[0]])

            out = np.reshape((np.asarray(out)),(len(out)//2,2))
            np.savetxt(self.IPCdict['folder']+"/single_particle_potential_outside.dat", out, fmt="%.12e")

            g = interp1d(out[:,0], out[:,1], kind='cubic')
            try:
                sol = root_scalar(g, bracket=[0,105.], method='brentq')
                self.sp_zero = sol.root
            except:
                print("no root for SP potential")

        else:
			
            if name == '2patch_analytic' or name == 'yukawa':
                print("special solutions not available for this topology"); exit(1)
				
            coord = [[ix,iy,iz] for ix in [self.sigma_core] for iy in np.linspace(0,np.pi,nth) for iz in np.linspace(0,2.*np.pi,nph) ] ##np.linspace(0,2.*np.pi,100) ]
            
            out = []

            for r, th, ph in coord:
                ##out0 = self.analytic_funcdict[name](r, th, ph)
                out0 = self.potential_outside_dd(r, th, ph)               
                out.extend([th, ph, float(out0)])

            out = np.reshape(np.asarray(out).astype('float64'),(int(len(out)/3),3))
        
            #if self.plot: 
            #    print_potential_colormap(out)
            np.savetxt(self.IPCdict['folder']+"/potential_outside.dat", out, fmt="%.5e")
            
            '''
            coord = [[ix,iy,iz] for ix in np.linspace(0.2*self.sigma_core, self.sigma_core,1000) for iy in [0.] for iz in [0.] ] 
            out = []

            for r, th, ph in coord:
                out0 = self.potential_inside_dd(r, th, ph)
                out.extend([r, float(out0)])

            out = np.reshape(np.asarray(out).astype('float64'),(int(len(out)/2),2))
        
            np.savetxt(self.IPCdict['folder']+"/potential_inside.dat", out, fmt="%.5e")
            '''
    
    def print_potential_radial(self, name):
        coord = [[ix,iy,iz] for ix in np.linspace(0.5, 1.5, 100) for iy in [0] for iz in [0] ]
            
        out = []

        for r, th, ph in coord:
            out0 = self.potential_outside_dd(r, th, ph)       
            out.extend([r, float(out0)])

        out = np.reshape(np.asarray(out).astype('float64'),(int(len(out)/2),2))
        
        np.savetxt(self.IPCdict['folder']+"/potential_outside_radial.dat", out, fmt="%.5e")


    def print_potential_lookup(self, name):
        
        Nr = 50; Nth = 100; Nph = 100;
        coord = [[ix,iy,iz] for ix in np.linspace(0.5, 1.5, Nr) for iy in np.linspace(0,np.pi,Nth) for iz in np.linspace(0,2.*np.pi,Nph) ] 

        _f = open(self.IPCdict['folder']+"/potential_lookup.dat", "w")
        _f.write("%d %d %d %d\n" % (Nr*Nth*Nph, Nr, Nth, Nph) )
        i = -1; dx = 0.01
        for r, th, ph in coord:
           i += 1 
           if float(i)/(Nr*Nth*Nph) > dx:
               print(dx, " percent done"); dx += 0.01
           out0 = self.potential_outside_dd(r, th, ph)               
           _f.write("%.5e %.5e %.5e %.5e\n" % (r, th, ph, float(out0)) )
            
        _f.close()
 

    def potential_outside_2sympatches(self, r, th, phi):

        sigma = self.sigma_core; Zp = self.patch_charge; Zc = self.colloid_charge
        a = self.ecc; kappa = self.kappa;
        
        out0 = ((Zc+Zp[0]+Zp[1]))*(np.exp(self.kappa*sigma)/(1.+self.kappa*sigma))*(np.exp(-self.kappa*r)/r)
        out1 = 0.

        #for l in range(2,self.lmax,2):
        #    out1 += (np.power(a[0]/sigma,l)*((2*l+1.)*(kv(l+0.5,kappa*r)/kv(l+1.5,kappa*sigma)))*eval_legendre(l,np.cos(th)))
        #out0 += (2.*Zp[0])/(self.kappa*sigma*np.sqrt(sigma*r))*out1
        for l in range(2,self.lmax,2):
            out1 = (Zp[0]*np.power(a[0]/sigma,l)*eval_legendre(l,np.cos(th))+Zp[1]*np.power(a[1]/sigma,l)*eval_legendre(l,np.cos(np.pi-th))  )
            out0 += ((2*l+1.)*(kv(l+0.5,kappa*r)/kv(l+1.5,kappa*sigma)))*out1
        out0 *= (1.)/(self.kappa*sigma*np.sqrt(sigma*r)*self.eps)
        #out0 *= (self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB)
        ##if self.real_units:
        
        out0 *= (self.e_charge*self.permittivity/(self.real_size))

        return out0

    def potential_inside_2sympatches(self, r, th, phi):

        sigma = self.sigma_core; Zp = self.patch_charge; Zc = self.colloid_charge
        a = self.ecc; kappa = self.kappa;
        
        out0 = -(Zc+np.sum(Zp))*(kappa*sigma) + (Zc+np.sum(Zp))
        out1 = 0.

        for l in range(2,self.lmax,2):
            #out1 = (Zp[0]*np.power(a[0]/sigma,l)*eval_legendre(l,np.cos(th))+Zp[1]*np.power(a[1]/sigma,l)*eval_legendre(l,np.cos(np.pi-th))  )
            out1 =  ( 2*Zp[0]*np.power(a[0]/sigma,l)*eval_legendre(l,np.cos(th)) ) 
            out0 += ( (kv(l+0.5,kappa*sigma)/kv(l+1.5,kappa*sigma)) + 1 )*out1
        out0 *= (1.)/(sigma*self.eps)

        if self.real_units:
            out0 *= (self.e_charge*self.permittivity/(self.real_size))
        
        return out0

    def potential_outside_2sympatches_yukawa(self, r, th, ph):

        sigmac = self.sigma_core; Zp = np.ravel(self.patch_charge); Zc = self.colloid_charge
        a = np.ravel(self.ecc); kappa = self.kappa; eps = self.eps; lmax = self.lmax

        el = 1.
        out0 = (Zc+2*Zp[0])
        out1 = 0.

        for l in range(2,lmax,2):
            out1 += np.power(a[0]/sigmac,l)*(2*l+1)*eval_legendre(l,np.cos(th))

        out0 += 2.*Zp[0]*out1
        out0 *= (np.exp(kappa*sigmac)/(1.+kappa*sigmac))*(np.exp(-kappa*r)/(r))
        out0 /= self.eps
        #out0 *= (self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB) 
        if self.real_units:
            out0 *= (self.e_charge*self.permittivity/(self.real_size))

        return [out0]

    def potential_inside(self, r, th, ph):

        Zc = self.colloid_charge; Zp = self.patch_charge; sigma = self.sigma_core
        sqsigma = np.sqrt(sigma)

        out = 0. + 0.*1j
        pref = 4.*np.pi
       
        out += -((Zc+np.sum(Zp))/sigma)*(self.kappa*sigma/(1.+self.kappa*sigma))
        out += (Zc+np.sum(Zp))/r

        for l in range(1, self.lmax):
            
            temp_l = 0.
            bessel_l = -(kv(l-0.5, self.kappa*sigma)/(kv(l+1.5, self.kappa*sigma)))*np.power(r/sigma,l)
            ##bessel_l = (((2*l+1)/(self.kappa*sigma))*(kv(l+0.5, self.kappa*sigma)/kv(l+1.5, self.kappa*sigma)) - 1)*np.power(r/sigma,l)

            for m in range(-l,l+1):

                temp_m = 0.
                temp_mp = 0.

                for k in range(self.npatch):
                    sph = np.conjugate(sph_harm(m,l,self.sph_phi[k],self.sph_theta[k]))     
                    temp_m += Zp[k]*np.power(self.ecc[k]/sigma,l)*sph
                    if r > self.ecc[k]:
                        temp_mp += Zp[k]*np.power(self.ecc[k]/r,l)*sph/r
                    else:
                        temp_mp += Zp[k]*np.power(r/self.ecc[k],l)*sph/self.ecc[k]

                sph = sph_harm(m, l, ph, th)
                temp_m *= sph; temp_mp *= sph
                temp_l += temp_m*bessel_l/sigma + temp_mp
                
            out += temp_l*pref/((2*l+1))
        
        out /= self.eps ##*(self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB)
        
        return np.real(out)
       
    def potential_outside(self, r, th, ph):

        Zc = self.colloid_charge; Zp = self.patch_charge; sigma = self.sigma_core
        sqsigma = np.sqrt(sigma)

        out = 0. + 0.*1j
        pref = 4.*np.pi
       
        out += ((Zc+np.sum(Zp)))*(np.exp(self.kappa*sigma)/(1.+self.kappa*sigma))*(np.exp(-self.kappa*r)/r) 
       
        for l in range(1, self.lmax):
            
            temp_l = 0.
            p0 = kv(l+0.5, self.kappa*r)/np.sqrt(r*sigma) 
            bessel_l = kv(l+1.5, self.kappa*sigma)

            for m in range(-l,l+1):

                temp_m = 0.
                
                for k in range(self.npatch):
                    sph = np.conjugate(sph_harm(m,l,self.sph_phi[k],self.sph_theta[k]))     
                    temp_m += Zp[k]*np.power(self.ecc[k]/sigma,l)*sph

                temp_m *= sph_harm(m, l, ph, th)
                temp_l += temp_m
                
            out += temp_l*(p0/bessel_l)*pref*(1./(self.kappa*sigma))
        
        out /= self.eps ##*(self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB)
        
        return np.real(out)


    def potential_inside_dd(self, r, th, ph):

        Zc = self.colloid_charge; Zp = self.patch_charge; sigma = self.sigma_core
        sqsigma = np.sqrt(sigma); _kappa = self.kappa
        eps1 = self.eps1; eps2 = self.eps2

        eps12 = self.eps2/self.eps1
        out = 0. + 0.*1j
        pref = 4.*np.pi
       
        out += -((Zc+np.sum(Zp))/(eps1*sigma))*(_kappa*sigma/(1.+_kappa*sigma))
        out += (Zc+np.sum(Zp))/(r*eps1)

        for l in range(1, self.lmax):
            
            temp_l = 0.
            bessel_num = np.power(r/sigma,l)*(-eps12*(_kappa*sigma)*kv(l+1.5, _kappa*sigma) + 
                          + eps12*l*kv(l+0.5, _kappa*sigma) + (l+1)*kv(l+0.5, _kappa*sigma));
            bessel_den = eps2*_kappa*sigma*kv(l+1.5, _kappa*sigma)+(eps1-eps2)*l*kv(l+0.5, _kappa*sigma)

            for m in range(-l,l+1):

                temp_m = 0.
                temp_mp = 0.

                for k in range(self.npatch):
                    sph = np.conjugate(sph_harm(m,l,self.sph_phi[k],self.sph_theta[k]))     
                    temp_m += Zp[k]*np.power(self.ecc[k]/sigma,l)*sph
                    if r > self.ecc[k]:
                        temp_mp += Zp[k]*np.power(self.ecc[k]/r,l)*sph/r
                    else:
                        temp_mp += Zp[k]*np.power(r/self.ecc[k],l)*sph/self.ecc[k]

                sph = sph_harm(m, l, ph, th)
                temp_m *= sph; temp_mp *= sph                
                temp_l += temp_m*(bessel_num/bessel_den)/sigma + temp_mp/eps1
                
            out += temp_l*(pref/(2*l+1))
        
        if self.real_units:
            out *= (self.e_charge*self.permittivity/(self.real_size))

        return np.real(out)

    def potential_outside_dd(self, r, th, ph):

        Zc = self.colloid_charge; Zp = self.patch_charge; sigma = self.sigma_core
        sqsigma = np.sqrt(sigma)
        
        out = 0. + 0.*1j
        pref = 4.*np.pi
       
        out += ((Zc+np.sum(Zp))/self.eps2)*(np.exp(self.kappa*sigma)/(1.+self.kappa*sigma))*(np.exp(-self.kappa*r)/r) 
       
        for l in range(1, self.lmax):
            
            temp_l = 0.
            p0 = kv(l+0.5, self.kappa*r)/np.sqrt(r*sigma) 
            bessel_l = self.eps2*self.kappa*sigma*kv(l+1.5, self.kappa*sigma)+(self.eps1-self.eps2)*l*kv(l+0.5, self.kappa*sigma)
            for m in range(-l,l+1):

                temp_m = 0.
                
                for k in range(self.npatch):
                    sph = np.conjugate(sph_harm(m,l,self.sph_phi[k],self.sph_theta[k])) 
                    temp_m += Zp[k]*np.power(self.ecc[k]/sigma,l)*sph

                temp_m *= sph_harm(m, l, ph, th)
                temp_l += temp_m
                
            out += temp_l*(p0/bessel_l)*pref
        
        ##if self.real_units:
        out *= (self.e_charge*self.permittivity/(self.real_size))

        return np.real(out)


