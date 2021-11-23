import numpy as np
#from numpy.linalg import det, inv
#from scipy.integrate import odeint
from constants import c

class PulsarRadiation:
    def __init__(self,t0,h0,alpha,i,omega,mu):
        self.t0=t0
        self.h0=h0
        self.alpha=alpha
        self.i=i
        self.omega=omega
        self.e_cross=np.array([[0,np.cos(i),-np.sin(i)],[np.cos(i),0,0],[-np.sin(i),0,0]])
        self.e_plus=np.array([[1,0,0],[0,-np.cos(i)**2,np.sin(i)*np.cos(i)],[0,np.sin(i)*np.cos(i),-np.sin(i)**2]])
        self.mu=mu
        
    def h_plus(self,t):
        first_freq=self.h0*np.sin(self.alpha)*(1/2)*np.cos(self.alpha)*np.sin(self.i)*np.sin(self.omega*(t-self.t0))
        second_freq=-self.h0*np.sin(self.alpha)*np.cos(self.i)*np.sin(self.alpha)*np.sin(2*self.omega*(t-self.t0))
        return first_freq+second_freq
    
    def h_cross(self,t):
        first_freq=self.h0*np.sin(self.alpha)*(1/2)*np.cos(self.i)*np.cos(self.alpha)*np.cos(self.omega*(t-self.t0))*np.sin(self.i)
        second_freq=self.h0*np.sin(self.alpha)*(-1/2)*(1+np.cos(self.i)**2)*np.cos(2*self.omega*(t-self.t0))*np.sin(self.alpha)
        return first_freq+second_freq
    
    def dot_h_plus(self,t):
        first_freq=self.h0*np.sin(self.alpha)*(1/2)*self.omega*np.cos(self.alpha)*np.cos(self.omega*(t-self.t0))*np.sin(self.i)
        second_freq=self.h0*np.sin(self.alpha)*(-2)*self.omega*np.cos(self.i)*np.cos(2*self.omega*(t-self.t0))*np.sin(self.alpha)
        return first_freq+second_freq
    
    def dot_h_cross(self,t):
        first_freq=self.h0*np.sin(self.alpha)*(-1/2)*self.omega*np.cos(self.i)*np.cos(self.alpha)*np.sin(self.i)*np.sin(self.omega*(t-self.t0))
        second_freq=self.h0*np.sin(self.alpha)*self.omega*(1+np.cos(self.i)**2)*np.sin(self.alpha)*np.sin(2*self.omega*(t-self.t0))
        return first_freq+second_freq
    
    def ddot_h_plus(self,t):
        first_freq=self.h0*np.sin(self.alpha)*(-1/2)*self.omega**2*np.cos(self.alpha)*np.sin(self.i)*np.sin(self.omega*(t-self.t0))
        second_freq=self.h0*np.sin(self.alpha)*4*self.omega**2*np.cos(self.i)*np.sin(self.alpha)*np.sin(2*self.omega*(t-self.t0))
        return first_freq+second_freq
    
    def ddot_h_cross(self,t):
        first_freq=self.h0*np.sin(self.alpha)*(-1/2)*self.omega**2*np.cos(self.i)*np.cos(self.alpha)*np.cos(self.omega*(t-self.t0))*np.sin(self.i)
        second_freq=self.h0*np.sin(self.alpha)*2*self.omega**2*(1+np.cos(self.i)**2)*np.cos(2*self.omega*(t-self.t0))*np.sin(self.alpha)
        return first_freq+second_freq
    
    def sigma_re(self,t):
        sigma_re=[h_plus*self.e_plus for h_plus in self.h_plus(t)]
        return sigma_re
    
    def sigma_im(self,t):
        sigma_im=[1j*h_cross*self.e_cross for h_cross in -self.h_cross(t)]
        return sigma_im
    
    def dot_sigma_re(self,t):
        dot_sigma_re=[h_plus*self.e_plus for h_plus in self.dot_h_plus(t)]
        return dot_sigma_re
    
    def dot_sigma_im(self,t):
        dot_sigma_im=[1j*h_cross*self.e_cross for h_cross in -self.dot_h_cross(t)]
        return dot_sigma_im
    
    def ddot_sigma_re(self,t):
        ddot_sigma_re=[h_plus*self.e_plus for h_plus in self.ddot_h_plus(t)]
        return ddot_sigma_re
    
    def ddot_sigma_im(self,t):
        ddot_sigma_im=[1j*h_cross*self.e_cross for h_cross in -self.ddot_h_cross(t)]
        return ddot_sigma_im
    
    def sigma(self,t):
        sigma=[self.sigma_re(t)[i]+self.sigma_im(t)[i] for i in range(len(t))]
        return np.array(sigma)
    
    def dsigma(self,t):
        dsigma=[self.dot_sigma_re(t)[i]+self.dot_sigma_im(t)[i] for i in range(len(t))]
        return np.array(dsigma)
    
    def ddsigma(self,t):
        ddsigma=[self.ddot_sigma_re(t)[i]+self.ddot_sigma_im(t)[i] for i in range(len(t))]
        return np.array(ddsigma)
    
    def get_sigma_matrix(self,t):
        return self.sigma(t),self.dsigma(t),self.ddsigma(t)
    
##########################
####Perturbative Method###    
##########################
    def RotationMatrix(self,t):
        P=[]
        for i in t:
            P.append([[np.cos(self.omega*(i-self.t0)),-np.cos(self.alpha)*np.sin(self.omega*(i-self.t0)),-np.sin(self.alpha)*np.sin(self.omega*(i-self.t0))], \
               [np.sin(self.omega*(i-self.t0)),np.cos(self.alpha)*np.cos(self.omega*(i-self.t0)),np.sin(self.alpha)*np.cos(self.omega*(i-self.t0))], \
                   [0                             ,-np.sin(self.alpha)                              ,np.cos(self.alpha)                               ]])
        
        return P
        
    def psi0(self,t):
        
        P=self.RotationMatrix(t)
        eta_im=[]
        for i in range(len(t)):
            eta_im.append(np.dot(P[i],np.array([0,0,self.mu])))
        
        return 0+1j*np.array(eta_im)
    
    def psi1(self,t):
        dt=abs(t[1]-t[0])*np.ones(3)#len(t)
        dpsi0=np.array(list(map(np.gradient,self.psi0(t).T,dt)))
        return dpsi0.T/(np.sqrt(2)*c)
    
    def psi2(self,t):
       dt=abs(t[1]-t[0])*np.ones(3)#len(t)
       dpsi1=np.array(list(map(np.gradient,self.psi1(t).T,dt)))
       return -dpsi1.T/(np.sqrt(2)*c)
   
    def psi1_2order(self,t): #No se usa por ahora
        dt=abs(t[1]-t[0])
        ddeta=-self.psi2(t)
        sigma=self.sigma(t)
        sigma_ddeta=np.array([np.dot(sigma[i],ddeta[i]) for i in len(t)])
        second_order_term=np.gradient(self.psi0(t),dt)/2+(3/20)*sigma_ddeta
        return self.psi1(t)+second_order_term
    
    def psi2_2order(self,t):#No se usa por ahora
        dt=abs(t[1]-t[0])
        return -np.gradient(self.psi1_2order(t),dt)
"""                      
########################
####Numerical Method###    
########################
    def RotationMatrix(self,i):

        P=[[np.cos(self.omega*(i-self.t0)),-np.cos(self.alpha)*np.sin(self.omega*(i-self.t0)),-np.sin(self.alpha)*np.sin(self.omega*(i-self.t0))], \
               [np.sin(self.omega*(i-self.t0)),np.cos(self.alpha)*np.cos(self.omega*(i-self.t0)),np.sin(self.alpha)*np.cos(self.omega*(i-self.t0))], \
                   [0                             ,-np.sin(self.alpha)                              ,np.cos(self.alpha)                               ]]
        
        return P 
    
    def psi0(self,t,mu):
        
        P=self.RotationMatrix(t)
        eta_im=np.dot(P,np.array([0,0,mu]))
        
        return 0+1j*np.array(eta_im)
    
    def sigma_inv(self,t):
        sigma=self.sigma(t)
        dets=list(map(det,sigma))
        i=0
        new_times=[]
        nonnull_sigma=[]
        for ele in dets:
            if ele!=0: 
                nonnull_sigma.append(sigma[i])
                new_times.append(t[i])
            i+=1
        inv_sigma=inv(nonnull_sigma)
        
        return new_times,nonnull_sigma,inv_sigma
    
    def dot_psi0(self,t,mu):
        return np.diff(self.psi0(t,mu))/np.diff(t)
    
    def dot_psi1(self,psi1,t,mu,sigma_inv):
        
        dot_psi0=self.dot_psi0(t,mu)
        inhomogenity=(-5/3)*np.dot(sigma_inv,dot_psi0)
        function=(10/3)*np.dot(sigma_inv,psi1)
        return inhomogenity+function
    
    def SolvePsi1(self,t,mu):
        y0=[0,0,0]
        psi1=odeint(self.dot_psi1,y0,t,args=(mu, self.sigma_inv))
        return psi1
        
    def psi2(self,t,mu):
        return -np.diff(self.psi1(t,mu)/np.diff(t))"""