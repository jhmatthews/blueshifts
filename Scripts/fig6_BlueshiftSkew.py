import astropy.io.ascii as io 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy import units as u
import py_plot_util as util
import sys, os
import matplotlib
import py_read_output as rd 
import jm_util
import blueshift_util
import astropy.io.fits as pyfits
from tqdm import tqdm

jm_util.set_mod_defaults()
jm_util.set_times()

def make_figure():
	data_dir = "{}/allspec/".format(blueshift_util.g_DataDir)
	cmap_names = blueshift_util.cmap_dict
	th_label = [70, 45, 20]


	plt.figure()
	angles_all = np.arange(5,90,5)

	plt.figure(figsize=(6,5))

	for itb, tb in enumerate(["thmin70", "thmin45", "thmin20"]):
	#for itb, tb in enumerate(["thmin70"]):
		fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and tb in f and "tau" not in f]

		norm, mappable = blueshift_util.get_norm_mappable(cmap=cmap_names[tb])
		ilabel = 0
		ball = []
		sall = []

		print ("Computing blueshifts and skews for all models with {}".format(tb))

		for j,f in enumerate(tqdm(fnames)):

			try:
				s = io.read("{}/{}".format(data_dir, f))
			except io.ascii.core.InconsistentTableError:
				print ("Error for", f)
			#angles = [int(sys.argv[1])]

			wavelength = s["Lambda"]
			velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs 
			velocity = velocity.to(u.km/u.s)

			nn = int(f[3:6].strip("_"))

			pf_root = f[:-5]
			pf = rd.read_pf("{}/{}".format(data_dir, pf_root))
			theta1 = float(pf["SV.thetamin(deg)"])
			theta2 = float(pf["SV.thetamax(deg)"])
			theta_b = 0.5 * (theta1 + theta2)
			launch = float(pf["SV.diskmin(units_of_rstar)"])

			if launch == 50:
				edgecolor="k"
				ls = "-"
			else:
				edgecolor="None"
				ls = "--"

			angles = np.arange(5,int(theta1)+5,5)
			#angles = angles_all[(angles_all>theta2) + (angles_all<theta1)]
			costheta = np.cos(np.radians(angles))

			blueshifts = np.zeros_like(angles, dtype=float)
			skews = np.zeros_like(angles, dtype=float)
			ews = np.zeros_like(angles, dtype=float)

			for i in range(len(angles)):
				colname = "A{:02d}P0.50".format(angles[i])

				fcont = blueshift_util.fit_continuum(wavelength, s[colname])
				try:
					flux = savgol_filter(s[colname]/fcont, 5, 3)
				except: 
					flux = s[colname]/fcont
				# get blueshift 
				blueshifts[i] =  blueshift_util.get_blueshift(velocity/(u.km / u.s), s[colname], fcont)
				#skews[i] = shape(wavelength,flux-fcont,blueshifts[i],1500,1600,1550)[0]
				vpeak, fpeak, skew = blueshift_util.get_skew(velocity/(u.km / u.s), flux, blueshifts[i])
				
				#_, _, vpeak = blueshift_util.shape(wavelength,flux-fcont,blueshifts[i],1500,1600,1550)
				skews[i] = (-vpeak)-blueshifts[i]
				ews[i] = blueshift_util.get_ew(wavelength, velocity/(u.km / u.s), s[colname], fcont)
			#print (skews)
				#print (blueshift)

			#ews = np.ones_like(blueshifts) * np.mean(ews) / costheta

			theta_winds = np.ones_like(angles) * theta_b

			#if np.all(ews > 0):
			edgecolor = "None"
			edgecolor="k"
			if ilabel == 0:
				label = r"$\theta_{{\rm min}} = {:.0f}^\circ$".format(th_label[itb])
				ilabel+=1
			else:
				label = None
			EW_CRIT = 20
			#plt.plot(blueshifts[ews>EW_CRIT], skews[ews>EW_CRIT], color=mappable.to_rgba(angles[3]), label=label)
			plot_kwargs = {"facecolors": mappable.to_rgba(40), "edgecolors":  "None", "s": 70, "alpha": 0.8}
			plt.scatter(1e50*blueshifts[ews>EW_CRIT], skews[ews>EW_CRIT], zorder=2, label=label, **plot_kwargs)
			#plt.scatter(blueshifts[ews>EW_CRIT], skews[ews>EW_CRIT], facecolors=mappable.to_rgba(angles[ews>EW_CRIT]), edgecolors=edgecolor, s=60, alpha=0.7, zorder=2)
			if nn > 81:
				plt.scatter(blueshifts[ews>EW_CRIT], skews[ews>EW_CRIT], zorder=2, **plot_kwargs, marker="^")
			else:
				plt.scatter(blueshifts[ews>EW_CRIT], skews[ews>EW_CRIT], zorder=2, **plot_kwargs)


	print ("Reading observational data and finishing plot...")

	blueshifts, ews, skew, vpeak = blueshift_util.measure_blueshifts()
	plt.scatter(blueshifts, (-vpeak)-blueshifts, c="k", zorder=3, marker="x", label="Composites")

	#cbar = plt.colorbar(mappable=mappable)
	#cbar.set_label(r"$\theta_w$ (half-opening angle)", fontsize=16)
	plt.legend(frameon=False, loc=2, handletextpad=0.1, fontsize=18, borderaxespad=0.1, borderpad=0.1)
	#plt.subplots_adjust(top=0.98, right=0.96, bottom=0.12, left=0.15)
	plt.xlabel(r"C~\textsc{iv}~Blueshift~(km~s$^{-1}$)", fontsize=18)
	plt.ylabel(r"C~\textsc{iv}~Skew~(km~s$^{-1}$)", fontsize=18, labelpad=-3)
	plt.xlim(-2000,2999)
	plt.ylim(-4000,4000)
	plt.grid(ls=":")
	plt.tight_layout(pad=0.05)
	blueshift_util.save_paper_figure("fig6.pdf")
	print ("All Done.")

if __name__ == "__main__":
	make_figure()