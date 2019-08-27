### compute analytical solution as in chapter

import numpy as np
from scipy.special import eval_legendre, lpmv

def mod_kv(n, x):
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

def der_mod_kv(n, x):
    val0 = mod_kv(n, x); val1 = mod_kv(n+1, x)
    return (n+0.5)*val0/x - val1
    
def sph_harm(l,m,th,ph):
    ##_fr = np.math.factorial(l-m)/np.math.factorial(l+m)
    _fr = 1.
    for k in range(l-m+1,l+m+1):
        _fr *= 1./float(k)
    
    return np.sqrt((2.*l+1)/(4.*np.pi)*_fr)*np.exp(1j*m*ph)*lpmv(m,l,np.cos(th))


class Mixin:

    def print_potential(self, name):

        coord = [[ix,iy] for ix in np.linspace(self.sigma,self.sigma,1) for iy in np.linspace(0,np.pi,100) ]
    
        out = []

        if name == 'analytic':
            for r, th in coord:
                out0 = self.potential_outside_2sympatches(r, th, 0.)
                out.extend([r,(th*180./np.pi), out0])
            out = np.reshape(np.real(np.asarray(out)),(len(out)/3,3))
        elif name == 'yukawa':
            for r, th in coord:
                out0 = self.potential_outside_2sympatches_yukawa(r, th, 0.)
                out.extend([r,(th*180./np.pi), out0])
            out = np.reshape(np.real(np.asarray(out)),(len(out)/3,3))
        else:
            print "mode not implemented"
            exit(1)

        return out

    def potential_outside_2sympatches(self, r, th, phi):

        sigma = self.sigma; Zp = self.patch_charge; Zc = self.colloid_charge 
        a = self.ecc; kappa = self.kappa; eps = self.eps; lmax = self.lmax
        el = 1.
        out0 = ((Zc+2*Zp[0])*el/eps)*(np.exp(kappa*sigma)/(1.+kappa*sigma))*(np.exp(-kappa*r)/r)
        out1 = 0.

        for l in range(2,lmax,2):
            out1 += ((a[0]/sigma)**l)*((2*l+1)*np.exp(np.log(mod_kv(l,kappa*r))-np.log(mod_kv(l+1,kappa*sigma)))*eval_legendre(l,np.cos(th)))
        out0 += (2.*Zp[0]/eps)/(kappa*sigma*np.sqrt(r*sigma))*out1
        
        return out0 

    def potential_outside_2sympatches_yukawa(self, r, th, ph):

        sigma = self.sigma; Zp = np.ravel(self.patch_charge); Zc = self.colloid_charge
        a = np.ravel(self.ecc); kappa = self.kappa; eps = self.eps; lmax = self.lmax
        el = 1.
        out0 = ((Zc+2*Zp[0])*el/eps)
        out1 = 0.

        for l in range(2,lmax,2):
            out1 += ((a[0]/sigma)**l)*((2*l+1))*eval_legendre(l,np.cos(th))

        out0 += (2.*Zp[0]/eps)*out1
        out0 *= (np.exp(kappa*sigma)/(1.+kappa*sigma))*(np.exp(-kappa*r)/r)

        return out0

'''
##NOT WORKING
def potential_numeric_2patches(sigma, Zp, Zc, a, kappa, eps, lmax):

    ## system of 2 equations for l,m: better to invert it analytically for higher precision 
   
    a_coeff = []
    b_coeff = []
    ## case l=0
    
    A0 = -(Zc+2*Zp)/(eps*sigma)*(kappa*sigma)/(1+kappa*sigma)
    B0 = ((Zc + 2*Zp)/eps)*(np.exp(kappa*sigma)/(1.+kappa*sigma))*np.sqrt(kappa)
    a_coeff.extend([A0])
    b_coeff.extend([B0])
    
    _mat_1 = np.empty((2,2), dtype = float)
    _known = np.empty((2,1), dtype = float)
    _sol = np.empty((2,1), dtype = complex)
    sq = np.sqrt(2.*kappa/np.pi) 
    pref = 2./eps #4.*np.pi/eps

    for l in range(lmax):
        
        mod_bessel_l = mod_kv(l,kappa*sigma); mod_bessel_der = der_mod_kv(l, kappa*sigma)
        #print mod_bessel_l, mod_bessel_der

        bessel_ratio = mod_bessel_der/mod_bessel_l ##np.exp(np.log(mod_bessel_der) - np.log(mod_bessel_l))
        det = (sigma**(l-1)*(sq)*(l - sigma*bessel_ratio))

        ###each of these should be divided by det; will be divided later
        _mat_1[0,0] = -sq*bessel_ratio
        _mat_1[0,1] = sq
        _mat_1[1,0] = -(l*sigma**(l-1)/mod_bessel_l)
        _mat_1[1,1] = (sigma**(l)/mod_bessel_l)

        _known[0,0] = -pref*(1.)*(Zp*a**l/sigma**(l+1))  ##1./(2.*l+1)
        _known[1,0] = pref*(1.)*((l+1)*(Zp*a**l/sigma**(l+2)))

        for m in range(-l,l+1):
            sph1 = np.conjugate(sph_harm(l,m,np.pi/2.,0))
            sph2 = np.conjugate(sph_harm(l,m,np.pi/2.,np.pi))
            
            b1 = _known[0,0]*(sph1+sph2)
            b2 = _known[1,0]*(sph1+sph2)
            
            _sol[0,0] = _mat_1[0,0]*b1 + _mat_1[0,1]*b2;
            _sol[1,0] = (_mat_1[1,0]*b1 + _mat_1[1,1]*b2)/det
            #print l, _sol
            a_coeff.extend([_sol[0,0]])
            b_coeff.extend([_sol[1,0]])

    coord = [[ix,iy,iz] for ix in np.linspace(sigma,sigma,1) for iy in np.linspace(0,np.pi,100) for iz in np.linspace(np.pi/4,np.pi/2.,1) ]
    out = list()

    for r, th, ph in coord:
        pb = 0; pa=0
        for l in range(0,lmax):
            p0 = kn(l,kappa*r)*sq ##np.power(r,-0.5)
            mc = 0
            for m in range(-l,l+1):
                pt = b_coeff[l*l+mc]*sph_harm(l,m, th, ph)*p0
                ##pp = b_coeff[l]*eval_legendre(l,np.cos(th))*p0
                pb += pp
                pt = a_coeff[l*l+mc]*sph_harm(l,m, th, ph)*r**l + fa
                mc += 1
        
        

        out.extend([r,(th*180./np.pi),ph,np.absolute(pb), np.absolute(pa)])
                                
    return np.reshape((np.asarray(out)),(len(out)/4,4))
'''

