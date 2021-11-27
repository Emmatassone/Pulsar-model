import numpy as np
#from numpy.linalg import det, inv
#from scipy.integrate import odeint
#from constants import c

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
