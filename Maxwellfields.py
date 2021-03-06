import numpy as np
from constants import c
from numpy.linalg import det, inv
from scipy.integrate import odeint
##########################
####Perturbative Method###    
##########################
class MaxwellFields:
    def __init__(self,t0,h0,alpha,i,omega,mu):
        self.t0=t0
        self.alpha=alpha
        self.i=i
        self.omega=omega
        self.mu=mu

#perturbative solution
    class perturbative:    
        def RotationMatrix(self,t):
            P=[]
            for i in t:
                P.append([[np.cos(self.omega*(i-self.t0)),-np.cos(self.alpha)*np.sin(self.omega*(i-self.t0)),-np.sin(self.alpha)*np.sin(self.omega*(i-self.t0))], \
                   [np.sin(self.omega*(i-self.t0)),np.cos(self.alpha)*np.cos(self.omega*(i-self.t0)),np.sin(self.alpha)*np.cos(self.omega*(i-self.t0))], \
                       [0                             ,-np.sin(self.alpha)                              ,np.cos(self.alpha)                               ]])
            
            return P
        
        def DotRotationMatrix(self,t):
            om = self.omega
            dot_P=[]
            for i in t:
                dot_P.append([[-om*np.sin(self.omega*(i-self.t0)),-om*np.cos(self.alpha)*np.cos(self.omega*(i-self.t0)),-om*np.sin(self.alpha)*np.cos(self.omega*(i-self.t0))], \
                   [om*np.cos(self.omega*(i-self.t0)),-om*np.cos(self.alpha)*np.sin(self.omega*(i-self.t0)),-om*np.sin(self.alpha)*np.sin(self.omega*(i-self.t0))], \
                       [0                             ,0                              ,0                               ]])
            
            return dot_P
        
        def DDotRotationMatrix(self,t):
            om2 = self.omega**2
            ddot_P=[]
            for i in t:
                ddot_P.append([[-om2*np.cos(self.omega*(i-self.t0)),om2*np.cos(self.alpha)*np.sin(self.omega*(i-self.t0)),om2*np.sin(self.alpha)*np.sin(self.omega*(i-self.t0))], \
                   [-om2*np.sin(self.omega*(i-self.t0)),-om2*np.cos(self.alpha)*np.cos(self.omega*(i-self.t0)),-om2*np.sin(self.alpha)*np.cos(self.omega*(i-self.t0))], \
                       [0                             ,0                              ,0                               ]])
            
            return ddot_P
        
        def psi0(self,t):
            
            P=self.RotationMatrix(t)
            eta_im=[]
            for i in range(len(t)):
                eta_im.append(np.dot(P[i],np.array([0,0,self.mu])))
            return 0+1j*np.array(eta_im)
        
        def psi1(self,t):
            #dt=abs(t[1]-t[0])*np.ones(3)#len(t)
            #dpsi0=np.array(list(map(np.gradient,self.psi0(t).T,dt)))
            #return dpsi0.T/(np.sqrt(2)*c)
            dot_P = self.DotRotationMatrix(t)
            dmu = []
            for i in range(len(t)):
                dmu.append(np.dot(dot_P[i],np.array([0,0,self.mu])))
            return 0-1j*np.array(dmu)/(np.sqrt(2)*c)
            
        def psi2(self,t):
           #dt=abs(t[1]-t[0])*np.ones(3)#len(t)
           #dpsi1=np.array(list(map(np.gradient,self.psi1(t).T,dt)))
           #return -np.sqrt(2)*dpsi1.T/c
           ddot_P = self.DDotRotationMatrix(t)
           ddmu= []
           for i in range(len(t)):
               ddmu.append(np.dot(ddot_P[i],np.array([0,0,self.mu])))
           return 0+1j*np.array(ddmu)/c**2
        
        def psi1_2order(self,t):
            ppmu=self.psi2(t)
            pmu=-self.psi1(t)
            sigma=self.sigma(t)
            sigma_ddeta=np.array([np.dot(sigma[i],ppmu[i]) for i in range(len(t))])
            second_order_term=pmu+(3/20)*sigma_ddeta
            return second_order_term#+self.psi1(t)
        
        def psi2_2order(self,t):
            dt=abs(t[1]-t[0])*np.ones(3)
            dpsi1=np.array(list(map(np.gradient,self.psi1_2order(t).T,dt)))
            return -np.sqrt(2)*dpsi1.T/c

#Numerical solution 
        class numerical:
            
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
                return -np.diff(self.psi1(t,mu)/np.diff(t))