# ======================================================
#	Black Hole Accretion Disc Spectrum
#	Computational Physics Year-III (PHYS-3561)
#	Thomas Davies (2022)
# ====================================================== 

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math

# ----------------------------- PHYSICAL CONSTANTS (SI) -----------------------------#
MSOL = 1.99e30
sb = 5.6704e-8
G = 6.6743e-11
h = 6.6261e-34
k = 1.3807e-23
pi = np.pi
c = 3e8

# ----------------------------- COMPUTATION ROUTINES -----------------------------#

def Fade(col, n, bias=0.1, m=0.7):
	alpha = 1.0 - 1.0 / np.power(n, m)
	alpha = min(alpha + bias, 1.0)
	return (*col, alpha)

class BlackHole:
	def __init__(self, bk_mass, bk_accrate, bk_spin):
		self.mass = bk_mass
		self.accrate = bk_accrate
		self.spin = bk_spin

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

		# Both branches yield same rms value for spin=0.
		# rms is calculated in units of Rg.
		if(self.spin >= 0): self.rms = (3 + z2) - np.sqrt((3-z1)*(3+z1+2*z2)) 	# prograde case
		else: 				self.rms = (3 + z2) + np.sqrt((3-z1)*(3+z1+2*z2))	# retrograde case

		self.yms = np.sqrt(self.rms)
		self.rout = 1e5 # Outer disk radius (scaled units R/Rg) 
		self.Rg = (G*self.mass)/(c**2)

def RadDist(t, v):
	n = (2*pi*h*(v**3))/c**2
	p = (h*v)/(k*t)

	THRESHOLD = 709
	if(p >= THRESHOLD): return n*np.exp(-p)
	else: return n/(np.exp(p) - 1)

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
	n = 3*G*bk.mass*bk.accrate
	d = 8*pi*sb*((bk.Rg*r)**3)
	t4 = (n/d)*corrFactor
	
	return np.power(t4, 0.25)

def ComputePhotonLuminosity(v, bk, tModel, binCount):
	def integrand(u):
		r = np.exp(u)
		return RadDist(tModel(r, bk), v) * (r**2)

	# Emission integral (logarithmic) bounds
	a,b = np.log(bk.rms), np.log(bk.rout) 	

	# Generate bins and sample integrand function
	BinStart = a 
	u_vals, f_vals = [], []
	BinWidth = (b-a) / binCount
	for i in range(binCount):
		u_vals.append(BinStart + 0.5*BinWidth ) 
		f_vals.append( integrand(u_vals[-1]) )
		BinStart += BinWidth

	# Approximate integral using trapezium rule
	emission_integral = 4*pi*(bk.Rg**2) * np.trapz(f_vals, u_vals)

	return emission_integral

def ComputeEmissionSpectrum(bk, expA, expB, BinCount, DecadeDensity):
	freqs, lums, flums = [], [], []

	for i in range(expA, expB):
		ds, de = 10**i, 10**(i+1)
		for f in np.linspace(ds,de,DecadeDensity):
			freqs.append(f)
	
	for f in freqs:
		l = ComputePhotonLuminosity(f, bk, TempDistGR, BinCount)
		lums.append(l)
		flums.append(f*l)

	tlum = np.trapz(lums, freqs)

	return freqs,lums,flums,tlum

# ----------------------------- PLOTTING ROUTINE -----------------------------#

BK_MASS, BK_ACCRATE = 7.0*MSOL, 8.0e12 
plot_col = (0.458824, 0.121569, 0.768627)

# Plot of how inner disk radius (Rin) changes with spin parameter (a*)
def InnerRadiusPlot():
	plt.figure()
	plt.xlabel(r"$a*$")
	plt.ylabel(r"$R_{in} \;\; [R_g]$")

	# Generates a Rin vs a* plot for a given number of spins.
	def GenPlot(expN):
		Rin_Values = []
		SpinStates = np.linspace(-0.998, 0.998, 10**expN)
		for spin in SpinStates:
			bk = BlackHole(BK_MASS, BK_ACCRATE, spin)
			Rin_Values.append(bk.rms)
		return SpinStates, Rin_Values

	for idx,expN in enumerate(range(1,4)):
		res = GenPlot(expN)
		col = Fade(plot_col, idx + 1, 0.3)
		plt.plot(res[0], res[1], color=col, label=r"$log \: N=$"+str(expN))
		
	plt.legend()
	plt.show()

# # Plot of Newtonian & GR Thermal models for our black hole in the no-spin case
# def Spinless_tProfiles():
# 	plt.figure()
	
# 	Datasets = []
# 	scale = 1e6 # for plot aethsetics
# 	bk = BlackHole(BK_MASS, BK_ACCRATE, 0)
# 	for expN in tqdm(np.linspace(1,6,12), desc="spinless_tProf"):
# 		radii = np.linspace(bk.rms, bk.rout, 10**expN)
# 		tNw, tGR = [],[]

# 		for r in radii:
# 			tNw.append(TempDistNw(r, bk) / scale)
# 			tGR.append(TempDistGR(r, bk) / scale)

# 		Datasets.append( (expN,radii,tNw,tGR) )

# 	for idx,ds in enumerate(Datasets):
# 		expN, radii, tNw, tGR = ds # unpack dataset
# 		sRadii = np.log10(radii)

# 		Opts = "--"
# 		LineWidth = 0.8
# 		lbl = r"$\log \: N =$"+str(round(expN,2))
# 		if(idx == len(Datasets) - 1):
# 			LineWidth = 1.2
# 			Opts = "-"

# 		c0 = Fade(plot_col, idx + 1, 0.3, 0.5)
# 		c1 = Fade(plot_col, idx + 1, 0.3, 0.5)

# 		plt.subplot(2,1,1)
# 		plt.plot(sRadii, tNw, Opts, linewidth=LineWidth,label=lbl,color=c0)

# 		plt.subplot(2,1,2)
# 		plt.plot(sRadii, tGR, Opts, linewidth=LineWidth,label=lbl,color=c1)

# 	plt.subplot(2,1,1)
# 	plt.ylabel("T [MK]")
# 	plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5))

# 	plt.subplot(2,1,2)
# 	plt.xlabel(r"$\ln(r) \;\; [\ln \: R_g]$")
# 	plt.ylabel("T [MK]")
# 	plt.legend(loc='center left', bbox_to_anchor=(0.98, 0.5))

# 	plt.show()

# Thermal model plots for our blackhole with various spin-states
def tProfiles():
	plt.figure()

	pidx = 1
	scale = 1e6
	Rout_Probe = 1e5
	SpinStates = [-0.998,0.0,0.998]
	cols = [plot_col, (0.901961, 0.337255, 0.298039),(0.235294, 0.545098, 0.901961)]
	for s in tqdm(SpinStates, desc="tProfiles"):
		bk = BlackHole(BK_MASS, BK_ACCRATE, s)
		Datasets = []
		
		for expN in np.linspace(1,6,8):
			temps = []
			radii = np.linspace(bk.rms, Rout_Probe, 10**expN)
			for r in radii: temps.append( TempDistGR(r, bk) / scale ) 
			Datasets.append( (expN, radii, temps) )

		plt.subplot(2,2,pidx)
		for idx,ds in enumerate(Datasets):
			expN, radii, temps = ds

			Opts = "--"
			LineWidth = 0.9
			lbl = r"$\log \: N =$"+str(round(expN,2))
			if(idx == len(Datasets) - 1):
				LineWidth = 1.3
				Opts = "-"

			# c0 = Fade(plot_col, idx + 1, 0.3)
			plt.plot(np.log10(radii), temps, Opts, linewidth=LineWidth,label=lbl,color=cols[pidx-1])

		pidx += 1

	plt.subplot(2,2,1)
	# plt.title(r"$a^*=-0.998$")
	plt.xlabel(r"$\log(r) \;\; [\ln \: R_g]$")
	plt.ylabel("T [MK]")

	plt.subplot(2,2,2)
	# plt.title(r"$a^*=0$")
	plt.xlabel(r"$\log(r) \;\; [\ln \: R_g]$")
	plt.ylabel("T [MK]")

	plt.subplot(2,2,3)
	# plt.title(r"$a^*=+0.998$")
	plt.xlabel(r"$\log(r) \;\; [\ln \: R_g]$")
	plt.ylabel("T [MK]")

	plt.show()

# Plots how total disk luminosity varies with spin parameter, and includes
# a convergence figure.
def LuminositySpinPlot():
	spins = np.linspace(-0.998,0.998,4)
	fExps = (14,19)

	def calc_tlums(BinCount,DecadeDensity):
		tlums = []
		for s in spins:
			bk = BlackHole(BK_MASS, BK_ACCRATE, s)
			spec = ComputeEmissionSpectrum(bk,fExps[0],fExps[1],BinCount,DecadeDensity)
			tlums.append(spec[-1])
		return tlums

	def CompareDatasets(y_refs,y_ds):
		metric = 0
		for j in range(len(y_refs)): 
			metric += abs(y_ds[j] - y_refs[j])
		
		return metric

	# ------------------------------------------------------------- #

	expBinCounts = [1,2,3]
	expDecadeDensities = [1,2,3,3.3,3.5]

	# Computes how the L(a*) plot varies as the BinCount increases.
	first, refDS, BC_metrics = True, [], [] 	
	for expBC in tqdm(expBinCounts, desc="bc_optimal"):
		bc = math.floor(10**expBC)
		dd = math.floor(10**expDecadeDensities[0])

		tlums = calc_tlums(bc, dd)
		if(first): 
			refDS = tlums.copy()
			first = False

		CMPmetric = CompareDatasets(refDS, tlums) 
		BC_metrics.append(CMPmetric)

	# Computes how the L(a*) plot varies as the DecadeDensity increases.
	# doesn't compute for last value in expDecadeDensities! 
	first, refDS, DC_metrics = True, [], []
	for idx in tqdm(range(len(expDecadeDensities) - 1),desc="LA-convergence"):
		dd = math.floor(10**expDecadeDensities[idx])
		bc = math.floor(10**expBinCounts[-1])

		tlums = calc_tlums(bc, dd)
		if(first): 
			refDS = tlums.copy()
			first = False

		CMPmetric = CompareDatasets(refDS, tlums) 
		DC_metrics.append(CMPmetric)

	# Compute the last value in expDecadeDensities
	tlums_final = calc_tlums(math.floor(10**expBinCounts[-1]), math.floor(10**expDecadeDensities[-1])) 
	CMPmetric = CompareDatasets(refDS, tlums_final) 
	DC_metrics.append(CMPmetric)

	scale = 1e9
	plt.figure()
	plt.xlabel(r"$a*$")
	plt.ylabel(r"$L \;\; [GW]$")
	ys = [y/scale for y in tlums_final]
	plt.plot(spins, ys, color=plot_col)
	plt.show()

	plt.figure()
	plt.subplot(2,1,1)
	plt.xlabel(r"$\log \: N$")
	plt.ylabel(r"$\sigma$")
	plt.plot(expBinCounts, BC_metrics, "o-", color=plot_col)

	plt.subplot(2,1,2)	
	plt.xlabel(r"$\log M$")
	plt.ylabel(r"$\sigma$")
	plt.plot(expDecadeDensities, DC_metrics, "o-", color=plot_col)
	plt.show()

	return spins, tlums_final

# Plot of the accretion efficiency (eta) as a function of spin state
def DiskEfficiency(spins, tlums):
	etas = []
	for L in tqdm(tlums, desc="diskEff"):
		eff = L / (BK_ACCRATE*(c**2))
		etas.append(eff)

	# eta is a dimensionless quantity.
	plt.figure()
	plt.xlabel(r"$a*$")
	plt.ylabel(r"$\eta$")
	plt.plot(spins, etas, color=plot_col)
	plt.show()

# ----------------------------- RESULTS CODE -----------------------------#
plt.rcParams.update({'font.size': 22})
# InnerRadiusPlot()
# Spinless_tProfiles()
tProfiles()
# la_data = LuminositySpinPlot()
# DiskEfficiency(*la_data)