import numpy as np
epsilon = np.array([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]])

c = 2.99792458e10
G = 6.6725985e-8
Msun = 1.98847e33     #Solar mass in CGS
mu0_mks=1.25663706212e-6 #Vacuum magnetic permeability in MKS= N/(A²).
mu0=mu0_mks*1e5 #Vacuum magnetic permeability in CGS. 1N=10^5 dyne
