# ======================================================
#	Black Hole Accretion Disc Spectrum
#	Computational Physics Year-III (PHYS-3561)
#	Thomas Davies (2022)
# ====================================================== 

### === Imports === ###
import matplotlib.pyplot as plt
import numpy as np

### === Physical Constants === ###
sigma = 5.6704e-8 # W m^-2 K^-4 
G = 6.6743e-11	# N m^2 kg^-2
h = 6.6261e-34 # J/Hz
k = 1.3807e-23 # J/K
c = 3e8 # m/s

### === Simulation === ###
def temp_pow4(r, m, mdot):
	return (G * m * mdot) / (8 * np.pi * (r**3) * sigma)

def blackbody(T, nu, m, mdot):
	num = (2*np.pi*h * (nu**3)) / (c**2) 
	denom = np.exp( (h*nu) / (k*T) ) - 1
	return num / denom

bin_count = 1e3
def sample_spectrum(nu, rin, rout, m, mdot,rg):
	dr = (rout - rin) / bin_count
	sval = 0
	r = rin

	while(r <= rout):
		T4 = temp_pow4(r + .5*dr, m, mdot)
		T = np.power(T4, 0.25)
		suffix = 4*np.pi*(rg**2)
		sval = sval + blackbody(T, nu, m, mdot)*suffix*r*dr
		r = r + dr

	return sval


m = 10*2e30 # black hole mass [kg]
mdot = 1e15 # accretion rate [kg/s]
rg = (G*m) / (c**2) 
rin,rout = 6*rg, 1e5*rg # integral limits
f0,f1,df = 1e14, 1e19,1e15
nu = f0

# compute the intensities over a range of photon frequencies. 
nus, spectrum = [], []
while(nu <= f1):
	intensity = sample_spectrum(nu, rin, rout, m, mdot, rg)	
	nus.append(nu)
	spectrum.append(intensity)
	nu = nu + df

plt.plot(nus, spectrum)
plt.show()