from sigma_pulsar_model import PulsarRadiation
from amplitude_models import PulsarAmplitudeModels
from Evolution_equations import EvolutionEquation
import numpy as np
import matplotlib.pyplot as plt
from constants import Msun,c,G,mu0
#Ref. https://www.cv.nrao.edu/~sransom/web/Ch6.html

pulsar={'NAME':'J0002+6216',
        'P0(s)':0.1153635682680,
        'F0(hz)':8.6682478274,
        'DIST(kpc)':6.357,
        'AGE(Yr)':3.06e+05,
        'BSURF(G)':8.4e+11,
        'EDOT(erg/s)':1.5e+35}

#parameter in SI units
betas=[1/5,1,100,500]
radius=1000*100 #typical NS radius in cm
age=pulsar['AGE(Yr)']*3.154e+7 #3.154e+7 s = 1yr
alphas=np.array([10,30,50,70,90])*np.pi/180.
inclinations=np.array([0,45,90])*np.pi/180.
omega=pulsar['F0(hz)']*2*np.pi
mu=pulsar['BSURF(G)']*(radius**3)
mu=2*np.pi*mu*3e9/(mu0*c)
print(mu)

amplitude=PulsarAmplitudeModels.MagneticFieldInducedDeformation(betas[0],radius,age,alphas[0])
pulsarmodel=PulsarRadiation(0, amplitude, alphas[0], inclinations[1], omega, mu)

EE=EvolutionEquation(0, amplitude, alphas[0], inclinations[1], omega, mu)
t=np.linspace(0,pulsar['P0(s)'],100)

EE.LoadSigma(t)
#Only sigma
#Mdot=EE.get_dot_M(EE.dsr,EE.dsi,np.eye(len(t),3)*0,np.eye(len(t),3)*0)
#M=EE.get_M(t, EE.dsr, EE.dsi, np.eye(len(t),3)*0,np.eye(len(t),3)*0,initial_M=1.4*Msun)
#Only phi
null_sigmas=np.zeros_like(EE.dsr)
Mdot=EE.get_dot_M(null_sigmas,null_sigmas,EE.p2r,EE.p2i)
M=EE.get_M(t, null_sigmas,null_sigmas, EE.p2r, EE.p2i,initial_M=0)#+1.4*Msun
#Both
#Mdot=EE.get_dot_M(EE.dsr,EE.dsi,EE.p2r,EE.p2i)
#M=EE.get_M(t, EE.dsr, EE.dsi, EE.p2r, EE.p2i,initial_M=0)#+1.4*Msun
#mu=np.sqrt(4*np.pi*3*c**2*I*P*Pdot/(mu0*8*np.pi**2*np.sin(alpha)**2)) , I: momento de Inercia
#nt,s,si=pulsarmodel.sigma_inv(t)
# plotting 
term2=-(c/(10*G))*np.einsum('ij,ij',EE.dsr[1],EE.dsr[1])
term3=-(1/(6*c))*np.einsum('i,i',EE.p2i[1],EE.p2i[1])
print('sigma term:',term2)
print('phi term:',term3)
fig ,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(t[1:], Mdot[1:])
ax1.set_xlabel("t")
ax1.set_ylabel("$\dot{M}$",rotation=90,loc='top')
ax2.plot(t[1:], M[1:])
ax2.set_xlabel("t")
ax2.set_ylabel("$M$",rotation=90,loc='top')
fig.tight_layout()
fig.savefig("t_vs_M.png")