### compute analytical solution as in chapter

import numpy as np
from scipy.special import eval_legendre, lpmv, kn, kvp, sph_harm
'''
def mod_kn(n, x):
    out = np.ones_like(x)  ##k = 0
    for k in range(0,n):
        ##out += exp((n+k)*log(n+k) - k*log(k) - (n-k)*log(n-k) - k)/(2*x)**k
        #out += (np.math.factorial(n+k)/(np.math.factorial(k)*np.math.factorial(n-k)))/(2*x)**k
        j=0; dum = 1.
        while (k-j) >= 1:
            dum *= float(n+k-j)/float(k-j)
            j=j+1
            if j >= k+1 :
                print 'error 1'
                exit(1)
        j=0
        while (n-j) >= (n-k+1):
            dum *= float(n-j)
            j=j+1
            if j >= k+1:
                print 'error 2'
                exit(1)
        out += dum/(2*x)**k

    return np.sqrt(np.pi/(2.*x))*np.exp(-x)*out

#def der_mod_kv(n, x):
#    val0 = np.sqrt(np.pi/(2*x))*mod_kv(n+1, x);  ##Kl+1
#    val1 = np.sqrt(np.pi/(2*x))*mod_kv(n-1, x)   ##Kl-1
#    return  (n*val1 - (n+1)*val0)/(2*n+1)  ###(n+0.5)*val0/x - val1
'''
''' 
def sph_harm(l,m,th,ph):
    ##_fr = np.math.factorial(l-m)/np.math.factorial(l+m)
    _fr = 1.
    m1 = m

    if m < 0:
        m1 = -m
   
    for k in range(l-m1+1,l+m1+1):
        _fr *= 1./float(k)
   
    if m < 0:
        return np.sqrt((2.*l+1)/(4.*np.pi)*_fr)*np.exp(1j*m1*ph)*((-1)**m*_fr)*lpmv(m1,l,np.cos(th))
    else:
        return np.sqrt((2.*l+1)/(4.*np.pi)*_fr)*np.exp(1j*m1*ph)*lpmv(m1,l,np.cos(th))
'''

class Mixin:

    def print_potential(self, name):

        coord = [[ix,iy] for ix in [self.sigma_core] for iy in np.linspace(0,np.pi,100) ]
    
        out = []

        if name == 'analytic':
            for r, th in coord:
                out0 = self.potential_outside_2sympatches(r, th, 0.)
                out.extend([(th*180./np.pi), out0])
        elif name == 'yukawa':
            for r, th in coord:
                out0 = self.potential_outside_2sympatches_yukawa(r, th, 0.)
                out.extend([(th*180./np.pi), out0])
        elif name == 'numerical':
            for r, th in coord:
                out0 = self.potential_numeric_2patches(r, th, 0.)
                out.extend([(th*180./np.pi), out0])
        else:
            print "wrong mode"
            exit(1)

        return np.reshape((np.asarray(out)),(len(out)/2,2))

    def potential_outside_2sympatches(self, r, th, phi):

        sigma = self.sigma_core; Zp = self.patch_charge; Zc = self.colloid_charge 
        a = self.ecc; kappa = self.kappa; eps = self.eps; lmax = self.lmax
        el = 1.
        out0 = ((Zc+2*Zp[0])*el/eps)*(np.exp(kappa*sigma)/(1.+kappa*sigma))*(np.exp(-kappa*r)/r)
        out1 = 0.

        for l in range(2,lmax,2):
            out1 += ((a[0]/sigma)**l)*((2*l+1)*np.exp(np.log(kn(l,kappa*r))-np.log(kn(l+1,kappa*sigma)))*eval_legendre(l,np.cos(th)))
        out0 += (2.*Zp[0]/eps)/(kappa*sigma*np.sqrt(r*sigma))*out1
        
        return out0 

    def potential_outside_2sympatches_yukawa(self, r, th, ph):

        sigma = self.sigma_core; Zp = np.ravel(self.patch_charge); Zc = self.colloid_charge
        a = np.ravel(self.ecc); kappa = self.kappa; eps = self.eps; lmax = self.lmax
        
        el = 1.
        out0 = (Zc+2*Zp[0])
        out1 = 0.

        for l in range(2,lmax,2):
            out1 += ((a[0]/sigma)**l)*((2*l+1))*eval_legendre(l,np.cos(th))

        out0 += 2.*Zp[0]*out1
        out0 *= (np.exp(kappa*sigma)/(1.+kappa*sigma))*(np.exp(-kappa*r)/(self.eps*r))

        return out0

    ##NOT WORKING
    def potential_numeric_2patches(self, r, th, ph):

        Zc = self.colloid_charge; Zp = self.patch_charge; sigma = self.sigma_core
        
        ## system of 2 equations for l,m: better to invert it analytically for higher precision 
        a_coeff = []
        b_coeff = []
        
        ## case l=0
        A0 = 0 ##TODO CHANGE 
        B0 = 0 
        a_coeff.extend([A0]); b_coeff.extend([B0])
    
        _mat_1 = np.empty((2,2), dtype = float)
        _known = np.empty((2,1), dtype = float)
        _sol = np.empty((2,1), dtype = complex) 
        pref = 4.*np.pi/self.eps

        for l in range(self.lmax):
        
            tl = 2*l+1
            bessel_l = kn(l,self.kappa*sigma); bessel_l1 = kn(l+1,self.kappa*sigma); 
            det = sigma**l*self.kappa*(l/(self.kappa*sigma)*bessel_l - bessel_l1) + l*sigma**(l-1)*bessel_l

            ###each of these should be divided by det; will be divided later
            _mat_1[0,0] = 0; _mat_1[0,1] = 0  ##TODO change
            _mat_1[1,0] = -l*sigma**(l-1); _mat_1[1,1] = sigma**l

            _known[0,0] = -pref/(tl*sigma**(l+1))
            _known[1,0] = pref*(l+1)/(tl*sigma**(l+2))
    
            for m in range(-l,l+1):
                sph1 = np.conjugate(sph_harm(m,l,0., np.pi/2))   ##sph_harm(l,m,np.pi/2.,0.))
                sph2 = np.conjugate(sph_harm(m,l,np.pi, np.pi/2))  ##sph_harm(l,m,np.pi/2.,np.pi))
                
                temp = (Zp[0]*self.ecc[0]**l*sph1+ Zp[1]*self.ecc[1]**l*sph2)

                b1 = _known[0,0]*temp
                b2 = _known[1,0]*temp
            
                _sol[0,0] = (_mat_1[0,0]*b1 + _mat_1[0,1]*b2)/det
                _sol[1,0] = (_mat_1[1,0]*b1 + _mat_1[1,1]*b2)/det
            
                a_coeff.extend([_sol[0,0]]); b_coeff.extend([_sol[1,0]])

        #for l in range(0,self.lmax):
        #    mc = 0
        #    for m in range(-l,l+1):
        #        print l, m,mc, l*l+mc, b_coeff[l*l+mc]
        #        mc += 1
        
        pb = 0; pa=0
        for l in range(0,self.lmax):
            p0 = kn(l,self.kappa*r)
            mc = 0; pb_0 = 0
            for m in range(-l,l+1):
                pt = b_coeff[l*l+mc]*sph_harm(m,l,ph,th)  ##sph_harm(l,m, th, ph)
                pb_0 += pt
                mc += 1
            pb += pb_0*p0
        
        return np.absolute(pb)


