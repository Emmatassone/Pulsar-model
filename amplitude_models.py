import numpy as np
from constants import c,mu0


class PulsarAmplitudeModels:

    def MagneticFieldInducedDeformation(beta,radius,age,alpha): #age=period/2*dot_period
        sigma0=3*beta*radius**2/(c*age*np.sin(alpha)**2)
        return sigma0
    
    def IncompressibleMagnetizedFluid(north_pole_magnetic_field,period,radius,constant_mass_density,magnetic_moment=0):
        
        if magnetic_moment==0:
            sigma0=6*np.pi**2*radius**3*north_pole_magnetic_field**2/(mu0*c**4*constant_mass_density*period**2)
        else:
            sigma0=3*mu0*magnetic_moment**2/(2*c**4*constant_mass_density*radius**3*period**2)
        return sigma0
        
