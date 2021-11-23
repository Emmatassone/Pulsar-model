import numpy as np
from scipy.integrate import cumtrapz
from constants import c,G,epsilon
from sigma_pulsar_model import PulsarRadiation

class EvolutionEquation(PulsarRadiation):
    
    def __init__(self,t0,h0,alpha,i,omega,mu):
        super().__init__(t0,h0,alpha,i,omega,mu)
       
        
    def LoadSigma(self,t):
        sigma2,dsigma2,ddsigma2=self.get_sigma_matrix(t)
        self.sr=sigma2.real
        self.si=sigma2.imag
        self.dsr=dsigma2.real
        self.dsi=dsigma2.imag
        self.ddsr=ddsigma2.real
        self.ddsi=ddsigma2.imag
        self.p1r=self.psi1(t).real
        self.p1i=self.psi1(t).imag
        self.p2r=self.psi2(t).real
        self.p2i=self.psi2(t).imag
      

    @staticmethod
    def dot_D(P,Q,p1r,p1i,p2r,p2i):
        term1=np.sqrt(2)*P/c
        term2=-(1/(3*c**2))*np.einsum('ijk,j,k->i',epsilon,p2i,p1r)
        term3=-(1/(3*c**2))*np.einsum('ijk,j,k->i',epsilon,p1i,p2r)
        term4=-np.sqrt(2)*Q*p2r/(3*c**2)
        return term1+term2+term3+term4
    
    @staticmethod
    def dot_J(Q,sr,si,dsr,dsi,p1r,p1i,p2r,p2i):
        term1=(1/(3*c**2))*np.einsum('ijk,j,k->i',epsilon,p1i,p2i)
        term2=(1/(3*c**2))*np.einsum('ijk,j,k->i',epsilon,p1r,p2r)
        term3=-(c**2/(5*G))*np.einsum('ijk,jl,lk->i',epsilon,si,dsi)
        term4=-(c**2/(5*G))*np.einsum('ijk,jl,lk->i',epsilon,sr,dsr)
        term5=np.sqrt(2)*Q*p2i/(3*c**2)
        return term1+term2+term3+term4+term5
    
    @staticmethod    
    def dot_P(dsr,dsi,p1i,p2r):
        term1=-np.sqrt(2)*c**2/(15*G)*np.einsum('ijk,jl,lk->i',epsilon,dsi,dsr)
        term2=c**4/(3*np.sqrt(2)*G)*np.einsum('ijk,jl,lk->i',epsilon,p1i,p2r)
        return term1+term2
    
    @staticmethod    
    def dot_M(dsr,dsi,p2r,p2i):
        term1=-(c/(10*G))*np.einsum('ij,ij',dsi,dsi)
        term2=-(c/(10*G))*np.einsum('ij,ij',dsr,dsr)
        term3=-(1/(3*c**2))*np.einsum('i,i',p2i,p2i)
        term4=-(1/(3*c**2))*np.einsum('i,i',p2r,p2r)
        return (term1+term2+term3+term4)/c**2
    
    @staticmethod
    def R(R,Q,D,M,P,sr,si,p1r,p1i,p2r,p2i):#chequear formula
        term1=(1/M)*D
        term2=(1/M)*4*np.sqrt(2)*np.einsum('i,ij->j',P,sr)/(5*c)
        term3=(1/M)*p2i*np.einsum('i,i',R,p1i)/(10*np.sqrt(2)*c**2)
        term4=-(1/M)*p1i*np.einsum('i,i',R,p2i)/(15*np.sqrt(2)*c**2)
        term5=(1/M)*R*np.einsum('i,i',p1i,p2i)/(10*np.sqrt(2)*c**2)
        term6=-(1/M)*Q*np.einsum('ijk,j,k',epsilon,R,p2i)/(6*c**2)
        term7=(1/M)*p2r*np.einsum('i,i',R,p1r)/(10*np.sqrt(2)*c**2)
        term8=-(1/M)*np.einsum('ijk,lk,l,j',epsilon,sr,p2i,p1r)/(30*c**2)
        term9=-(1/M)*np.einsum('ijk,lk,j,l',epsilon,sr,p2i,p1r)/(30*c**2)
        term10=(1/M)*np.einsum('jkl,il,j,k',epsilon,sr,p2i,p1r)/(30*c**2)
        term11=-(1/M)*p1r*np.einsum('i,i',R,p2r)/(15*np.sqrt(2)*c**2)
        term12=(1/M)*R*np.einsum('i,i',p1r,p2r)/(10*np.sqrt(2)*c**2)
        term13=(1/M)*Q*np.einsum('ij,j',sr,p2r)/(15*np.sqrt(2)*c**2)
        term14=(1/M)*np.einsum('ijk,lk,l,j',epsilon,sr,p1i,p2r)/(30*c**2)
        term15=(1/M)*np.einsum('ijk,lk,j,l',epsilon,sr,p1i,p2r)/(30*c**2)
        term16=(1/M)*np.einsum('jkl,il,j,k',epsilon,sr,p1i,p2r)/(30*c**2)
        return term1+term2+term3+term4+term5+term6+term7+term8+term9+term10+term11+term12+term13+term14+term15+term16
    
    @staticmethod
    def S(J,R,Q,P,sr,si,p1r,p1i,p2r,p2i):
        term0=J
        term1=np.einsum('ijk,j,k->i',epsilon,P,R)/c
        term2=-Q*np.einsum('ij,j->i',sr,p2i)/(15*np.sqrt(2)*c**2)
        term3=-np.einsum('ijl,lk,k,j->i',sr,p1i,p2i)/(30*c**2)
        term4=-np.einsum('ijl,lk,j,k->i',sr,p1i,p2i)/(30*c**2)
        term5=-np.einsum('jkl,il,j,k->i',sr,p1i,p2i)/(30*c**2)
        term6=p1r*np.einsum('i,i',R,p2i)/(15*np.sqrt(2)*c**2)
        term7=p2r*np.einsum('i,i',R,p1i)/(10*np.sqrt(2)*c**2)
        term8=-p2i*np.einsum('i,i',R,p1r)/(10*np.sqrt(2)*c**2)
        term9=-R*np.einsum('j,j',p2i,p1r)/(10*np.sqrt(2)*c**2)
        term10=-p1i*np.einsum('j,j',R,p2r)/(15*np.sqrt(2)*c**2)
        term11=R*np.einsum('j,j',p1i,p2r)/(10*np.sqrt(2)*c**2)
        term12=-Q*np.einsum('ijk,j->i',R,p2r)/(6*c**2)
        term13=-np.einsum('ikl,lj,j,k->i',sr,p1r,p2r)/(30*c**2)
        term14=-np.einsum('ijl,lk,j,k->i',sr,p1r,p2r)/(30*c**2)
        term15=-np.einsum('jkl,il,j,k->i',sr,p1r,p2r)/(30*c**2)
        
        return term0+term1+term2+term3+term4+term5+term6+term7+term8+term9+term10+term11+term12+term13+term14+term15
       
    def get_dot_D(self,P,Q,p1r,p1i,p2r,p2i):
        return np.array(list(map(self.dot_D,P,Q,p1r,p1i,p2r,p2i)))

    def get_dot_J(self,Q,sr,si,dsr,dsi,p1r,p1i,p2r,p2i):
        return np.array(list(map(self.dot_J,Q,sr,si,dsr,dsi,p1r,p1i,p2r,p2i)))

    def get_dot_P(self,dsr,dsi,p1i,p2r):
        return np.array(list(map(self.dot_P,dsr,dsi,p1i,p2r)))

    def get_dot_M(self,dsr,dsi,p2r,p2i):
        return np.array(list(map(self.dot_M,dsr,dsi,p2r,p2i)))
    
    def get_J(self,t,Q,sr,si,dsr,dsi,p1r,p1i,p2r,p2i,initial_J=0):
        return cumtrapz(self.get_dot_J(Q,sr,si,dsr,dsi,p1r,p1i,p2r,p2i).T,t,initial=initial_J)

    def get_P(self,t,dsr,dsi,p1i,p2r,initial_P=0):
        return cumtrapz(self.get_dot_P(dsr,dsi,p1i,p2r).T,t,initial=initial_P)
    
    def get_M(self,t,dsr,dsi,p2r,p2i,initial_M=0):
        return cumtrapz(self.get_dot_M(dsr,dsi,p2r,p2i),t,initial=initial_M)
        
