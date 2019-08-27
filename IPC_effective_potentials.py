import numpy as np

from scipy.special import eval_legendre

class Mixin:

    def effective_potential_2patches(self):

        sigma = self.sigma; Zc = self.colloid_charge
        Zp = np.ravel(self.patch_charge); a = np.ravel(self.ecc)
        lmax = self.lmax
        dlvo_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)
   
        th0 = np.pi/2.;
        for r in np.linspace(sigma,1.5*sigma,100): 
            VV = np.zeros(3)
            a1 = np.arcsin(a/np.sqrt(r**2+a**2)); r1 = np.sqrt(r**2+a**2)
            ### EQUATIORIAL-EQUATORIAL, POLAR-POLAR, EQUATORIAL-POLAR
            th = [[th0, th0-a1[0], th0+a1[1], -th0, -th0+a1[0], -th0-a1[1]], [0.,0.,0.,0.,0.,0.], [th0,th0,th0, 0, -a1[0],+a1[1]]]
            dd = [[r,r1[0],r1[1],r,r1[0],r1[1]],[r,r-a[0],r+a[1],r,r-a[0],r+a[1]], [r,r-a[0],r+a[1],r,r1[0],r1[0]]]
        
            for i in range(3):
                V1 = 0
                V1 += Zc*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][0], th[i][0], 0.)
                V1 += Zp[0]*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][1], th[i][1], 0.)
                V1 += Zp[1]*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][2], th[i][2], 0.)

                V2 = 0
                V2 += Zc*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][3], th[i][3], 0.)
                V2 += Zp[0]*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][4], th[i][4], 0.)
                V2 += Zp[1]*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][5], th[i][5], 0.)
           
                VV[i] = 0.5*(V1+V2)

            self.effective_potential.extend([r,VV[0],VV[1],VV[2]])

        self.effective_potential = np.reshape(np.asarray(self.effective_potential),(len(self.effective_potential)/4,4))
        self.effective_potential[:,1:] = self.effective_potential[:,1:]/np.fabs(self.effective_potential[0,3])


    def effective_potential(self):

        sigma = self.sigma; Zp = np.ravel(self.patch_charge); Zc = self.colloid_charge 
        a = np.ravel(self.ecc); lmax = self.lmax; nconf = 3
        dlvo_c = np.exp(self.kappa*self.sigma_core)/(1.+self.kappa*self.sigma_core)

        th0 = np.pi/2.;
        for r in np.linspace(sigma,1.5*sigma,100):
            VV = np.zeros(nconf)
        
            a1 = np.arcsin(a/np.sqrt(r**2+a**2)); r1 = np.sqrt(r**2+a**2)
            th = [[th0, th0-a1[0], th0+a1[1], -th0, -th0+a1[0], -th0-a1[1]], [0.,0.,0.,0.,0.,0.], [th0,th0,th0, 0, -a1[0],+a1[1]]]
            dd = [[r,r1[0],r1[1],r,r1[0],r1[1]],[r,r-a[0],r+a[1],r,r-a[0],r+a[1]], [r,r-a[0],r+a[1],r,r1[0],r1[1]]]

            for i in range(nconf):
                V1 = 0
                V1 += Zc*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][0], th[i][0], 0.)
                V1 += Zp*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][1], th[i][1], 0.)
                V1 += Zp*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][2], th[i][2], 0.)

                V2 = 0
                V2 += Zc*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][3], th[i][3], 0.)
                V2 += Zp*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][4], th[i][4], 0.)
                V2 += Zp*dlvo_c*self.potential_outside_2sympatches_yukawa(dd[i][5], th[i][5], 0.)

                VV[i] = 0.5*(V1+V2)

            self.effective_potential.extend([r,VV[0],VV[1],VV[2]])

        self.effective_potential = np.reshape(np.asarray(self.effective_potential),(len(self.effective_potential)/4,4))
        self.effective_potential[:,1:] = self.effective_potential[:,1:]/np.fabs(self.effective_potential[0,3])


'''
def effective_potentials_yukawa(Zc, Zp, kappa, sigma_c, a, eps):

    sigma = 2.*sigma_c; lmax = 80
    dlvo_c = np.exp(kappa*sigma_c)/(1.+kappa*sigma_c)
    V = []
     
    for r in np.linspace(sigma,1.5*sigma,100): 
        VV = np.zeros(3)
        a1 = np.arcsin(a/np.sqrt(r**2+a**2)); th0 = np.pi/2.; r1 = np.sqrt(r**2+a**2)
        ### EQUATIORIAL-EQUATORIAL, POLAR-POLAR, EQUATORIAL-POLAR
        th = [[th0, th0-a, th0+a, -th0, -th0+a, -th0-a], [0.,0.,0.,0.,0.,0.], [th0,th0,th0, 0, -a,+a]]  
        dd = [[r,r1,r1,r,r1,r1], [r,r-a,r+a,r,r-a,r+a], [r,r-a,r+a,r,r1,r1]]

        for i in range(3):
            V0 = Zc**2; V00 = 0.
            V1 = Zc*Zp; V01 = 0.
            V2 = Zc*Zp; V02 = 0.
            for l in range(0,lmax,2):
                V00 += (2*l+1)*((a/sigma_c)**l)*eval_legendre(l,np.cos(th[i][0]))
                V01 += (2*l+1)*((a/sigma_c)**l)*eval_legendre(l,np.cos(th[i][1]))
                V02 += (2*l+1)*((a/sigma_c)**l)*eval_legendre(l,np.cos(th[i][2]))

            V0 += V00*2.*Zc*Zp
            V1 += V01*2.*Zp**2; V1 *= np.exp(-kappa*(dd[i][1]-r))/(dd[i][1]/r)
            V2 += V02*2.*Zp**2; V2 *= np.exp(-kappa*(dd[i][2]-r))/(dd[i][2]/r)

            VV1 = (V0+V1+V2)*(dlvo_c**2)*np.exp(-kappa*r)/(eps*r)
       
            V0 = Zc**2; V00 = 0.
            V1 = Zc*Zp; V01 = 0.
            V2 = Zc*Zp; V02 = 0.
            for l in range(0,lmax,2):
                V00 += (2*l+1)*((a/sigma_c)**l)*eval_legendre(l,np.cos(th[i][3]))
                V01 += (2*l+1)*((a/sigma_c)**l)*eval_legendre(l,np.cos(th[i][4]))
                V02 += (2*l+1)*((a/sigma_c)**l)*eval_legendre(l,np.cos(th[i][5]))

            V0 += V00*2.*Zc*Zp
            V1 += V01*2.*Zp**2; V1 *= np.exp(-kappa*(dd[i][4]-r))/(dd[i][4]/r)
            V2 += V02*2.*Zp**2; V2 *= np.exp(-kappa*(dd[i][5]-r))/(dd[i][5]/r)

            VV2 = (V0+V1+V2)*(dlvo_c**2)*np.exp(-kappa*r)/(eps*r)

            VV[i] = 0.5*(VV1+VV2)
        
        V.extend([r,VV[0],VV[1],VV[2]])

    return np.reshape(np.asarray(V),(len(V)/4,4))
'''
