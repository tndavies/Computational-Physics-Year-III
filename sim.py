# ======================================================
#	Black Hole Accretion Disc Spectrum
#	Computational Physics Year-III (PHYS-3561)
#	Thomas Davies (2022)
# ====================================================== 

### === Imports === ###
import matplotlib.pyplot as plt
import numpy as np

### === Physical Constants (SI UNITS) === ###
SOLAR_MASS = 2e30
SIGMA = 5.6704e-8
G = 6.6743e-11
h = 6.6261e-34
k = 1.3807e-23
c = 3e8

# Supervisor Meeting Questions
# 1) For large frequencies, np.exp in temp function overflows ... how to handle it?
# 2) How to integrate spectrum plot as its a discrete data set ... maybe linearly interpolate between nearest data points?
# 3) Explain physics of line emission

### === Simulation Parameters === ###
FREQ_SAMPLES = 1e3
FNT_SIZE = 22

bhMass = 10*SOLAR_MASS
bhAccRate = 1e15
Rg = (G*bhMass)/(c**2)
Rl = 6*Rg # inner disk radius 
Ru = (1e5)*Rg # outter disk radius
Fl = 1e14 # photon frequency range start (Hz) 
Fu = 1e19 # photon frequency range end (Hz) 

# @optimise: Upgrade to RK4 for faster convergence
def integrator(f, a, b, bN):
	bWidth = (b - a) / bN
	binx, sum = a, 0

	while(binx <= b):
		fVal = f(binx, bWidth)
		sum = sum + fVal*bWidth
		binx = binx + bWidth

	return sum

# @note: this is the non-viscous temperature function!
def dktemperature(R):
	n = G * bhMass * bhAccRate
	d = 8*np.pi*SIGMA*(R**3)

	return np.power(n/d, 0.25)

def emission_intensity(T, nu):
	n = (2*np.pi*h*(nu**3)) / (c**2)
	m = (h*nu) / (k*T)
	d = np.exp(m) - 1

	return n / d

# @optimise: Switch integral to use log bins for faster possible faster convergence? 
def spectral_intensity(nu, bN):
	int = integrator(
			lambda bO,bW : bO*emission_intensity(dktemperature(bO+0.5*bW), nu),
			Rl, Ru,
			bN
		)

	return 4*np.pi*int

def compute_spectrum(bN):
	frequencies = np.linspace(Fl, Fu, FREQ_SAMPLES)
	luminosities = []
	
	for nu in frequencies:
		Lnu = spectral_intensity(nu, bN)
		luminosities.append(Lnu)

	return frequencies, luminosities

### === Obtain Emission Spectrum === ###
freqs, lums = compute_spectrum(10)
plt.figure()
plt.title("Accretion Disk Spectrum (RAW)", fontsize=FNT_SIZE)
plt.xlabel(r'$\nu$',fontsize=FNT_SIZE)
plt.ylabel(r'$\L_{\nu}$',fontsize=FNT_SIZE)
plt.plot(freqs, lums, "x--")
plt.show()

### === Obtain Total Disk Emission Luminosity === ###
specA,specB=1.2e13,1e18

#def spectrum_function(nu):
	# get two nearest frequency data samples
	# linearly interpolate between them
	# return the interpolated intensity value

#tLum = integrator(spectrum_function, specA, specB, TLUM_BIN_COUNT)
#print("Total Accretion Disk Luminosity: " + str(tLum) + " [W??]")

### === Demonstrate Convergence for Bin Counts === ###
diffMetric = 0
for idx,f in enumerate(freqs):
	diffMetric = diffMetric + abs( lums[idx] - lums2[idx] )

print("Difference metric: " + str(diffMetric))
