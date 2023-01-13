# ======================================================
#	Black Hole Accretion Disc Spectrum
#	Computational Physics Year-III (PHYS-3561)
#	Thomas Davies (2022)
# ====================================================== 

import matplotlib.pyplot as plt
import numpy as np

# Physical Constants
solar_mass = 2e30
sb = 5.6704e-8
G = 6.6743e-11
h = 6.6261e-34
k = 1.3807e-23
pi = np.pi
c = 3e8

# Numerical evaluation parameters
IntegralSampleDensity = 100
SamplesPerDecade = 50

FNT_SIZE = 15

class BlackHole:
	def __init__(self, bk_mass, bk_accrate, bk_spin):
		self.mass = bk_mass
		self.accrate = bk_accrate
		self.spin = bk_spin

		# Here we compute several constants that depend on the
		# black hole's spin state ... for example rin is the
		# inner radius of the accretion disk, and is used in all
		# cases ... whereas most others are only used when the
		# GR temperature model is used during simulation.
		self.aS = abs(self.spin)

		n = np.arccos(self.aS)
		self.y1 = 2*np.cos( (n-pi)/3  )
		self.y2 = 2*np.cos( (n+pi)/3  )
		self.y3 = -2*np.cos( n/3 )

		za = np.cbrt(1-self.aS**2)
		zb = np.cbrt(1+self.aS)
		zc = np.cbrt(1-self.aS)

		z1 = 1 + za*(zb + zc)
		z2 = np.sqrt( 3*(self.aS**2) + z1**2 )

		self.rms = 0
		prograde = True if(self.spin>0) else False
		if(prograde): self.rms = (3 + z2) - np.sqrt((3-z1)*(3+z1+2*z2))
		else:  		  self.rms = (3 + z2) + np.sqrt((3-z1)*(3+z1+2*z2))	

		self.yms = np.sqrt(self.rms)

		self.rout = 1e5 # Outer disk radius (scaled units R/Rg) 
		self.Rg = (G*self.mass)/(c**2)

def RadDist(t, v):
	n = (2*pi*h*(v**3))/c**2
	p = (h*v)/(k*t)

	# If p takes a value bigger than THRESHOLD
	# then exp(p) will be too big for us to store
	# in memory.
	THRESHOLD = 709

	if(p >= THRESHOLD):
		return n*np.exp(-p)
	else:
		return n/(np.exp(p) - 1)

# Newtonian derived temperature model
def TempDistNw(r, bk):
	n = 3*G*bk.mass*bk.accrate
	d = 8*pi*sb*((bk.Rg*r)**3)
	t4 = (n/d)*(1-np.power(bk.rms/r,0.5))
	return np.power(t4, 0.25)

# General Relativity derived temperature model
def TempDistGR(r, bk):
	y = np.sqrt(r)

	A = 1 - (bk.yms / y) - (3*bk.aS*np.log(y/bk.yms))/(2*y)

	Bn1 = 3 * (bk.y1-bk.aS)**2 * np.log((y-bk.y1)/(bk.yms-bk.y1))
	Bd1 = y*bk.y1*(bk.y1-bk.y2)*(bk.y1-bk.y3)
	Bn2 = 3 * (bk.y2-bk.aS)**2 * np.log((y-bk.y2)/(bk.yms-bk.y2))
	Bd2 = y*bk.y2*(bk.y2-bk.y1)*(bk.y2-bk.y3)
	Bn3 = 3 * (bk.y3-bk.aS)**2 * np.log((y-bk.y3)/(bk.yms-bk.y3))
	Bd3 = y*bk.y3*(bk.y3-bk.y1)*(bk.y3-bk.y2)
	B = (Bn1/Bd1) + (Bn2/Bd2) + (Bn3/Bd3)

	C = 1 - (3/r) + (2*bk.aS) / np.power(r, 1.5)

	corrFactor = (A-B)/C 

	return TempDistNw(r, bk) * corrFactor

def ComputePhotonLuminosity(v, bk, tModel):
	a,b = np.log(bk.rms), np.log(bk.rout) 	# logarithmic integral bounds

	def integrand(u):
		r = np.exp(u)
		t = tModel(r, bk)
		f = RadDist(t, v)
		return f * (r**2)

	# The integrand is undefined at u=a as TempDist(a)=0 in the model,
	# thus we bias the lower bound slightly as to avoid this.
	LowerBoundBias = 1.001

	# Sample integrand over integration range [a,b]
	uvals,samples = np.linspace(LowerBoundBias*a, b, IntegralSampleDensity),[]
	for u in uvals: samples.append(integrand(u))

	# Approximate integral using trapezium rule
	prefactor = 4*pi*(bk.Rg**2)
	int = np.trapz(samples, uvals)
	lv = prefactor * int

	return(lv)

def ComputeEmissionSpectrum(bk, tModel):
	freqs, lums, flums = [], [], []
	fexp0, fexp1 = 14, 19			# spectrum frequency range

	for exp in range(fexp0,fexp1):
		# boundary values for current frequency decade (start, end)
		ds = 10**exp  
		de = ds*10

		for f in np.linspace(ds, de, SamplesPerDecade):
			l = ComputePhotonLuminosity(f, bk, tModel)
			freqs.append(f)
			lums.append(l)
			flums.append(f*l)

	tlum = np.trapz(lums,freqs)

	return freqs,lums,flums,tlum


# === SIMULATION ===#

# Plot of Newtonian and GR temperature models,
# over the milestone blackhole accretion disk.
def GenTempPlot(spin):	
	bk = BlackHole(10*solar_mass,1e15,spin)
	tempsN, tempsGR = [], []
	
	radii = np.linspace(bk.rms,bk.rout,15*(bk.rout-bk.rms))
	for r in radii:
		tempsN.append(TempDistNw(r,bk))
		tempsGR.append(TempDistGR(r,bk))

	return radii,tempsN,tempsGR

radii0,tempsN0,tempsGR0 = GenTempPlot(-0.998)
radii1,tempsN1,tempsGR1 = GenTempPlot(0)
radii2,tempsN2,tempsGR2 = GenTempPlot(0.998)

plt.figure()
plt.xlabel(r'$log_{10} \; r$', fontsize=FNT_SIZE)
plt.ylabel("T [K]", fontsize=FNT_SIZE)

s0,c0 = r'$a^* = -0.998$', 'r'
plt.plot(np.log10(radii0), tempsGR0, ""+c0, label=s0)
plt.plot(np.log10(radii0), tempsN0, "--"+c0)

s1,c1 = r'$a^* = 0$', 'g'
plt.plot(np.log10(radii1), tempsGR1, ""+c1, label=s1)
plt.plot(np.log10(radii1), tempsN1, "--"+c1)

s2,c2 = r'$a^* = +0.998$', 'b'
plt.plot(np.log10(radii2), tempsGR2, ""+c2, label=s2)
plt.plot(np.log10(radii2), tempsN2, "--"+c2)

plt.legend()
plt.savefig("fig1.png")

	# Plots of the milestone black hole spectrum in both the Newtonian
	# and GR models.
	# freqs0,lums0,flums0,tlum0 = ComputeEmissionSpectrum(bk, TempDistNw)
	# freqs1,lums1,flums1,tlum1 = ComputeEmissionSpectrum(bk, TempDistGR)
	# plt.figure()
	# plt.xlabel("log10 frequency [Hz]")
	# plt.ylabel("log10 Luminosity?? [??]")
	# plt.plot(np.log10(freqs0), np.log10(flums0), label="Newtonian Model")
	# plt.plot(np.log10(freqs1), np.log10(flums1), label="General Relativity Model")
	# plt.savefig("fig2.png")

# Plot of how the inner accretion disk radius changes as the milestone
# black hole spins.
spins, rins = np.linspace(-0.998,0.998,1000), []
for s in spins: 
	blackhole = BlackHole(10*solar_mass, 1e15, s) 
	rins.append(blackhole.rms)
plt.figure()
plt.xlabel(r'$a^*$', fontsize=FNT_SIZE)
plt.ylabel(r'$r_{in}$', fontsize=FNT_SIZE)
plt.plot(spins, rins)
plt.savefig("fig3.png")

# Plot of how the total disk emission luminosity changes with black hole spin.
spins, tlums = np.linspace(-0.998,0.998,50), []
for s in spins:
 	blackhole = BlackHole(10*solar_mass, 1e15, s)
 	freqs,lums,flums,tlum = ComputeEmissionSpectrum(blackhole, TempDistGR)
 	tlums.append(tlum)
plt.figure()
plt.xlabel(r'$a^*$', fontsize=FNT_SIZE)
plt.ylabel('$L_{tot} \; [W]$', fontsize=FNT_SIZE)
plt.plot(spins, tlums)
plt.savefig("fig4.png")
