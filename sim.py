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

### === Simulation Parameters === ###

bhMass = 10*SOLAR_MASS
bhAccRate = 1e15 			# [kg/s]
Rg = (G*bhMass)/(c**2)
rin, rout = 6,1e5			# Accretion disk inner/outer radii (in units of Rg)

PTXTSIZE = 20 				# Size of text on plots
PTRANS = False				# Make background of plots transparent

###  ======================= ###

def integrator(f, a, b, bN):
	bWidth = (b - a) / bN
	binx, sum = a, 0

	while(binx <= b):
		fVal = f(binx, bWidth)
		sum = sum + fVal*bWidth
		binx = binx + bWidth

	return sum

def temp(r):
	n = 3 * G * bhMass * bhAccRate
	d = 8*np.pi*SIGMA*(Rg**3)*(r**3)
	m = (n / d) * (1 - np.sqrt(rin / r))

	return np.power(m, 0.25)

def blackbody(T, v):
	n = (2*np.pi*h)/(c**2)
	m = (h*v) / (k*T)
	r = 0

	# exp(>700) can't be stored with a 64-bit float
	EXP_F64_LIMIT = 709
	if(m >= EXP_F64_LIMIT):
		r = (n*(v**3))*np.exp(-m)
	else:
		r = (n*(v**3))/(np.exp(m)-1)

	return r

def compute_spectrum(bN):
	lums, freqs = [], [] # spectrum results
	lfs,lfe=14,19 # log10 of lower/upper frequencies
	# iterate over each decade between 10^0 and 10^15
	decade_sample_count = 100
	for i in range(lfs, lfe):
		_g = 4*np.pi*(Rg**2)
		decade_start = 10**i
		decade_end = 10**(i+1)
		
		fs = np.linspace(decade_start,decade_end,decade_sample_count)
		for v in fs:
			def Foo(bO,bW):
				u = bO+0.5*bW
				x = temp(np.exp(u))
				y = blackbody(x,v)
				return y*np.exp(u)
			vlum = _g*integrator(Foo, np.log(rin), np.log(rout), bN)
			freqs.append(v)
			lums.append(vlum)
	return freqs, lums


### === Accretion Disk Temeprature Profile === ###

# iterate over each decade between 10^0 and 10^15
radii, temps = [], []
decade_sample_count = 500
for i in range(0, 7):
	decade_start = rin * (10**i)
	decade_end = rin * (10**(i+1))
			
	rs = np.linspace(decade_start,decade_end,decade_sample_count)
	ts = temp(rs)

	for j in rs: radii.append(j)
	for j in ts: temps.append(j)

plt.figure()
plt.xlabel(r'$log_{10}(r_s)$', fontsize=PTXTSIZE)
plt.ylabel("T [K]", fontsize=PTXTSIZE)
plt.plot(np.log10(radii), temps, "-")
plt.savefig("TvsR", transparent=PTRANS, bbox_inches='tight')

### === Accretion Disk Spectrum Plot === ###
freqs, lums = compute_spectrum(1000)

fs, ls = freqs.copy(), lums.copy()
for idx, v in enumerate(fs):
	ls[idx] = ls[idx]*v

plt.figure()
plt.xlabel(r'$\log_{10}(\nu) \; [\log_{10} \: Hz]$', fontsize=PTXTSIZE)
plt.ylabel(r'$\log_{10}(\nu \: L_{\nu}) \; [\log_{10} \: W]$', fontsize=PTXTSIZE)
plt.plot(np.log10(fs), np.log10(ls), "-")
plt.savefig("spectrum", transparent=PTRANS, bbox_inches='tight')

### === Accretion Disk Total Luminosity Value === ###
total_lum = np.trapz(lums,freqs)
print("Accretion Disk Luminosity: " + str(total_lum) + " [Watts]")







