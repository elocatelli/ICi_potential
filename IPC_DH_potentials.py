### compute analytical solution as in chapter

import numpy as np
from scipy.special import eval_legendre, lpmv, kn, kv, kvp, sph_harm
from math import factorial

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

    def print_potential(self, name):

        if self.npatch == 2:
            th0 = 0.
            if name == 'numeric':
                th0 -= np.arccos(np.dot(np.asarray([0.,0.,1.]),self.topo[0]))
            coord = [[ix,iy] for ix in [self.sigma_core] for iy in np.linspace(th0,th0+np.pi,100) ]

            out = []

            for r, th in coord:
                out0 = self.analytic_funcdict[name](r, th, 0.)
                out.extend([((th-th0)*180./np.pi), out0])

            return np.reshape((np.asarray(out)),(int(len(out)/2),2))
        else:
            coord = [[ix,iy,iz] for ix in [self.sigma_core] for iy in np.linspace(0,np.pi,50) for iz in np.linspace(0,2.*np.pi,100) ]
            
            out = []

            for r, th, ph in coord:
                out0 = self.analytic_funcdict[name](r, th, ph)
                out.extend([th, ph, out0])

            return np.reshape((np.asarray(out)),(int(len(out)/3),3))


    def potential_outside_2sympatches(self, r, th, phi):

        sigma = self.sigma_core; Zp = self.patch_charge; Zc = self.colloid_charge
        a = self.ecc; kappa = self.kappa;
        
        out0 = ((Zc+2*Zp[0]))*(np.exp(self.kappa*sigma)/(1.+self.kappa*sigma))*(np.exp(-self.kappa*r)/r)
        out1 = 0.

        for l in range(2,self.lmax,2):
            out1 += (np.power(a[0]/sigma,l)*((2*l+1.)*(kv(l+0.5,kappa*r)/kv(l+1.5,kappa*sigma)))*eval_legendre(l,np.cos(th)))
        out0 += (2.*Zp[0])/(self.kappa*sigma*sigma)*out1
        #out0 *= (self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB)

        return out0

    def potential_outside_2sympatches_yukawa(self, r, th, ph):

        sigma = self.sigma_core; Zp = np.ravel(self.patch_charge); Zc = self.colloid_charge
        a = np.ravel(self.ecc); kappa = self.kappa; eps = self.eps; lmax = self.lmax

        el = 1.
        out0 = (Zc+2*Zp[0])
        out1 = 0.

        for l in range(2,lmax,2):
            out1 += np.power(a[0]/sigma,l)*(2*l+1)*eval_legendre(l,np.cos(th))

        out0 += 2.*Zp[0]*out1
        out0 *= (np.exp(kappa*sigma)/(1.+kappa*sigma))*(np.exp(-kappa*r)/(r))
        #out0 *= (self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB) 

        return out0

        ##NOT WORKING
    def potential_numeric_2patches(self, r, th, ph):

        Zc = self.colloid_charge; Zp = self.patch_charge; sigma = self.sigma_core
        sqsigma = np.sqrt(sigma)
        ## system of 2 equations for l,m: better to invert it analytically for higher precision
        #a_coeff = []
        #b_coeff = []

        out = 0. + 0.*1j
        pref = 4.*np.pi

        ## case l=0
        B0 = 2.828427125*(Zc+np.sum(Zp))*np.exp(self.kappa*sigma)*np.sqrt(self.kappa)/((1.+self.kappa*sigma))  
       
        #print("old", ((Zc+np.sum(Zp)))*(np.exp(self.kappa*sigma)/(1.+self.kappa*sigma))*(np.exp(-self.kappa*r)/r) )
        #print("new", B0*sph_harm(0,0,ph,th)*kv(0.5,self.kappa*r)/np.sqrt(r))
        
        out += B0*sph_harm(0,0,ph,th)*kv(0.5,self.kappa*r)/np.sqrt(r) 

        _mat_1 = np.empty((2,2), dtype = float)
        _known = np.zeros((2,1), dtype = float)
        _sol = np.empty((2,1), dtype = complex)
       
        for l in range(1, self.lmax):

            tl = 2*l+1
            
            p0 = kv(l,self.kappa*r)
            bessel_l = kn(l,self.kappa*sigma); bessel_der = self.kappa*kvp(l,self.kappa*sigma)

            #p0 = kv(l+0.5, self.kappa*r)/np.sqrt(r) 
            #bessel_l = kv(l+0.5, self.kappa*sigma)/sqsigma
            #bessel_der = (kvp(l+0.5,self.kappa*sigma)*self.kappa/sqsigma - 0.5*bessel_l/sigma)
            det = -1.*np.power(sigma,l)*bessel_der + l*np.power(sigma,l-1)*bessel_l

            ###each of these should be divided by det; will be divided later
            _mat_1[0,0] = np.power(sigma,l) ; _mat_1[0,1] = -bessel_l  ##TODO change
            _mat_1[1,0] = l*np.power(sigma,l-1); _mat_1[1,1] = -self.kappa*bessel_der

            _known0 = -pref/(tl*np.power(sigma, l+1))
            _known1 = pref*(l+1)/(tl*np.power(sigma, l+2))

            _known[0,0] = _known0; _known[1,0] = _known1

            for m in range(-l,l+1):

                temp = 0.
                
                for k in range(self.npatch):

                    sph = np.conjugate(sph_harm(m,l,self.sph_phi[k],self.sph_theta[k]))    ##   
                    temp += Zp[k]*np.power(self.ecc[k],l)*sph
                    #temp = (Zp[0]*np.power(self.ecc[0],l)*sph1+ Zp[1]*np.power(self.ecc[1],l)*sph2)

                b1 = _known[0,0]*temp
                b2 = _known[1,0]*temp

                _sol[0,0] = (_mat_1[1,1]*b1 + _mat_1[0,1]*b2)/det
                _sol[1,0] = (-_mat_1[1,0]*b1 + _mat_1[0,0]*b2)/det

                out += _sol[1,0]*p0*sph_harm(m, l, ph, th) 

        return np.real(out) #*(self.e_charge*self.permittivity/(self.eps*self.real_size))/(self.temperature*self.kB)
