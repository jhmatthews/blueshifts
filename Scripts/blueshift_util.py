import numpy as np 
import astropy.io.ascii as io 
import astropy.io.fits as pyfits
from astropy import constants as const
from astropy import units as u
import py_read_output as rd 
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

#cmap_names = ["YlGn_r", "YlOrRd_r", "YlGnBu_r"]
cmap_names = ["YlGn_r", "RdPu_r", "Blues_r"]

cmap_dict = {"thmin70": "Blues_r", "thmin45": "RdPu_r", "thmin20": "YlGn_r"}
fiducial_angle = 15

g_DataDir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', '_Data'))
g_FigureDir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', 'Figures'))
def rho2nh():
    z,abun,mass = np.genfromtxt("{}/elements.txt".format(g_DataDir), unpack=True, usecols=(1,3,4))
    x = 0
    y = 0
    xx = 0
    for i in range(len(abun)):
        x += ((10.0 ** abun[i]) * mass[i])
        xx +=  ((10.0 ** abun[i]) * z[i])
    y = (10.0 ** abun[0])
    x *= const.m_p.cgs.value
    return (y/x, xx/y)


def save_paper_figure(savename, fig = None, figure_dir = g_FigureDir, **savefig_kwargs):
    if fig == None:
        fig = plt.gcf()
    
    full_savename = "{}/{}".format(figure_dir, savename)
    fig.savefig(full_savename, **savefig_kwargs)
     


def get_filenames_for_alpha(system_args = ["dummy", "0.5"], override=None):
    # CHOOSE YOUR FIGHTER
    # by which I mean, choose which value of alpha to plot 
    if len(system_args) > 1:
        Alpha = system_args[1]
    else:
        Alpha = "0.5"

    if override is not None:
        Alpha = override

    speclist_dir = "{}/speclists/".format(g_DataDir)

    if Alpha == "-1":
        AlphaString = "combo"
        speclist_column = 0
        data_dir = data_dir  = "{}/allspec/".format(g_DataDir)
        speclist = np.genfromtxt("{}/speclist_combo.txt".format(speclist_dir), dtype=str, usecols = (speclist_column,), unpack=True)
        Alpha = float(Alpha)
    elif Alpha == "-1.5":
        AlphaString = "alpha1.5"
        speclist_column = 0
        data_dir  = "{}/{}/".format(g_DataDir, AlphaString)
        speclist = np.genfromtxt("{}/speclist_alpha1.5_9.txt".format(speclist_dir), dtype=str, usecols = (speclist_column,), unpack=True)
        Alpha = 1.5
    else:
        AlphaString = "alpha{}".format(Alpha)
        speclist_column = int ( (float(Alpha) + 0.01) // 0.5) - 1 # gives 0,1,2 for 0.5,1,2 as desired
        # folder containing spectra for this value of alpha 
        data_dir  = "{}/{}/".format(g_DataDir, AlphaString)
        # load list of spectra with column set by value of alpha 
        speclist = np.genfromtxt("{}/speclist.txt".format(speclist_dir), dtype=str, usecols = (speclist_column,), unpack=True)
        Alpha = float(Alpha)
        
    return (Alpha, AlphaString, data_dir, speclist)

def get_data_dir_for_alpha(alpha=0.5):
    if alpha==0.5:
        AlphaString = "alpha{:.1f}".format(alpha)
    else:
        AlphaString = "alpha{:.0f}".format(int(alpha))
    # folder containing spectra for this value of alpha 
    data_dir  = "{}/{}/".format(g_DataDir, AlphaString)
    return (data_dir)


def get_norm_mappable(cmap):
    norm = matplotlib.colors.Normalize(vmin=5, vmax=90)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    return (norm, mappable)

def read_fits_spectrum(f = "h19_c3_19061908_1525_a.fits"):
    """Read an observed spectrum from a fits file

    Args:
        f (str, optional): filename. Defaults to "h19_c3_19061908_1525_a.fits".

    Returns:
        w, fl: wavelength and flux arrays 
    """
    with pyfits.open("{}".format(f)) as hdulist:
        hdr = hdulist[0].header
        wfirst = hdr["COEFF0"]
        wlast = wfirst + (hdr["COEFF1"] * hdr["NAXIS1"])
        w = np.logspace(wfirst, wlast, hdr["NAXIS1"], endpoint=False)
        fl = hdulist[0].data[0]

    return (w, fl)

def get_line_lum(wavelength, velocity, f, fcont, d=100.0):
    select = (np.fabs(velocity)<1e4) 
    integrand = f - fcont
    integrand = integrand[select]
    integral = np.fabs(np.trapz(integrand, wavelength[select]))
    pc = const.pc.cgs.value
    lum = integral * 4.0 * np.pi * (d * d * pc * pc)
    return (lum)


def get_ew(wavelength, velocity, f, fcont):
    """Calculate equivalent width

    Args:
        wavelength (array-like): wavelength array
        velocity (array-like): velocity array
        f (array-like): flux
        fcont (array-like): continuum flux 

    Returns:
        integral (float): equivalent width
    """
    integrand = (1.0 - f/fcont)
    select = (np.fabs(velocity)<1e4)
    integrand = integrand[select]
    integral = np.trapz(integrand, wavelength[select])

    return (integral)


def calc_blueshift(f, data_dir):
	try:
		s = io.read("{}/{}".format(data_dir, f))
	except io.core.InconsistentTableError:
		print ("Error for", f)

	wavelength = s["Lambda"]
	velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs 
	velocity = velocity.to(u.km/u.s)

	pf_root = f[:-5]
	pf = rd.read_pf("{}/{}".format(data_dir, pf_root))
	theta1 = float(pf["SV.thetamin(deg)"])
	theta2 = float(pf["SV.thetamax(deg)"])

	edgecolor="k"
	ls = "-"

	angles = np.arange(5,int(theta1)+5,5)

	blueshifts = np.zeros_like(angles, dtype=float)
	ews = np.zeros_like(angles, dtype=float)

	for i in range(len(angles)):
		colname = "A{:02d}P0.50".format(angles[i])

		fcont = fit_continuum(wavelength, s[colname])
		flux = savgol_filter(s[colname]/fcont, 5, 3)

		# get blueshift 
		blueshifts[i] =  get_blueshift(velocity/(u.km / u.s), s[colname], fcont)
		ews[i] = get_ew(wavelength, velocity/(u.km / u.s), s[colname], fcont)
		#print (blueshift)

	return (blueshifts, ews, angles)

#@jit(nopython=True)
def get_blueshift(velocity, f, fcont):
    line_flux = f-fcont
    line_flux = line_flux[(np.fabs(velocity)<1e4)]
    v_use = velocity[(np.fabs(velocity)<1e4)]

    # interpolate on wavelengths
    interp_func = interp1d(v_use, line_flux)
    velocity_new = np.linspace(np.min(v_use), np.max(v_use), 100000)
    fnew = interp_func(velocity_new)

    cdf = np.cumsum(fnew) / np.sum(fnew)
    halfway = np.argmin(np.fabs(cdf - 0.5))

    #plt.plot(velocity[(np.fabs(velocity)<1e4)], cdf)
    #plt.plot(line_flux)
    return (-velocity_new[halfway])

def order(mi,ma):
    if mi>ma:
        mi,ma=ma,mi
    return mi,ma

#@jit(nopython=True)
def fit_continuum(w, f, lines=[1215,1240,1400,1640,1753,1909,2800,1550,1030,4863,6563], 
                  window_length=9, polyorder=5, mask_size = 70):
	f_filter = savgol_filter(f, window_length, polyorder) 
	mask = np.ones_like(w, dtype=bool)
	for l in lines:
		mask *= (np.fabs(l - w) >  mask_size)

	interp_func = interp1d(w[mask], f_filter[mask], kind="slinear", fill_value="extrapolate")
	fcont = interp_func(w)
	return (fcont)

def getind(lst,val):
    diff=abs(lst-val)
    closest=min(diff)
    diff=list(diff)
    return diff.index(closest)

def shape(w,det,bls,lmin,lmax,l0, return_arg = False):
    xmin,xmax=order(getind(w,lmax),getind(w,lmin))
    reduced=det[xmin:xmax]
    argmax = list(reduced).index(np.max(reduced))
    argmax = np.argmax(reduced)
    ws=w[xmin:xmax][argmax]

    print (xmin, xmax, argmax)

    vpeak = const.c.cgs * (ws-l0)/l0

    vpeak = vpeak.cgs.value / 1e5

    if return_arg:
        return vpeak - bls, ws, vpeak, argmax, xmin, xmax
    else:
        return vpeak - bls, ws, vpeak

def get_skew(velocity, flux, blueshift):
    line_flux = flux
    line_flux = line_flux[(np.fabs(velocity)<1e4)]
    v_use = velocity[(np.fabs(velocity)<1e4)]

    # interpolate on wavelengths
    interp_func = interp1d(v_use, line_flux)
    velocity_new = np.linspace(np.min(v_use), np.max(v_use), 100000)
    fnew = interp_func(velocity_new)

    argmax = np.argmax(fnew)
    vpeak = velocity_new[argmax]
    fpeak = fnew[argmax]
    skew = vpeak - blueshift

    return (vpeak, fpeak, skew)

def get_flux_at_wavelength(lambda_array, flux_array, w):

    '''
    Find the flux at wavelength w

    Parameters:
        lambda_array: array-like    
            array of wavelengths in angstroms. 1d 

        flux_array: array-like 
            array of fluxes same shape as lambda_array 

        w: float 
            wavelength in angstroms to find
    
    Returns:
        f: float 
            flux at point w
    '''

    i = np.abs(lambda_array - w).argmin()

    return flux_array[i]

def measure_blueshifts(subfolder = "/composites33/", fits_structure = True):
    """measure blueshifts in observational data from a file 

    Args:
        folder (str, optional): 
        folder name within the global data directory. 
        Defaults to composites33.

    Returns:
        blueshifts, ews, skew, vpeak:
            arrays holding the blueshifts, EWs, skews and peak velocities
            for each spectrum in the folder.
    """
    # list all files 
    folder = "{}/{}/".format(g_DataDir, subfolder)

    if fits_structure:
        files = os.listdir(folder)
        N = len(files)
    else:
        N = 33
        wavelength = np.genfromtxt("{}/jm_33comps_norm.wav".format(folder))
        fluxes = np.genfromtxt("{}/jm_33comps_norm.dat".format(folder), unpack=True)

    # blank arrays
    blueshifts = np.zeros(N)
    ews = np.zeros(N)
    skew = np.zeros(N)
    lambda0 = 1550
    vpeak = np.zeros(N)

    for i in range(N):
        if fits_structure:
             f = files[i]
             wavelength,fl = read_fits_spectrum(folder+f)
        else:
             fl = fluxes[i]
             
        fcont = fit_continuum(wavelength, fl)
        velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs 
        velocity = velocity.to(u.km/u.s)

        blueshifts[i] =  get_blueshift(velocity/(u.km / u.s), fl, fcont)
        ews[i] = get_ew(wavelength, velocity/(u.km / u.s), fl, fcont)
        vpeak[i], fpeak, skew[i] = get_skew(velocity/(u.km / u.s), fl/fcont, blueshifts[i])
        #skew[i], _, vpeak[i] = shape(wavelength,fl-fcont,blueshifts[i],1500,1600,1550)

    return (blueshifts, ews, skew, vpeak)

def gradient_fill(x, y, fill_color=None, ax=None, mappable = None, alpha = None, zorder = None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    kwargs2 = kwargs.copy() 
    kwargs2["label"] = None
    kwargs2["lw"] = 0
    line, = ax.plot(x, y, **kwargs2)
    if fill_color is None:
        fill_color = line.get_color()

    if zorder is None: 
        zorder = line.get_zorder()
    if alpha is None:
        alpha = line.get_alpha()
        alpha = 1.0 if alpha is None else alpha


    if mappable is None:
        # z = np.empty((100, 1, 4), dtype=float)
        # rgb = mcolors.colorConverter.to_rgb(fill_color)
        # z[:,:,:3] = rgb
        # z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

        Nz = 10000
        z = np.empty((Nz, 1, 4), dtype=float)
        #print (fill_color)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        rgb = fill_color[:3]
        #print (rgb)
        z[:,:,:3] = rgb

        z[:,:,-1] = np.linspace(0, 0.85, Nz)[:,None] ** 1.5
        
    else:
        xx, yy = np.meshgrid(x, y)
        z = mappable.to_rgba(xx[::-1])
        z[:,:,-1] = alpha
    #z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]


    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, 0, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, 0], xy, [xmax, 0], [xmin, 0]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    line, = ax.plot(x, y, color = fill_color, **kwargs, zorder=zorder)

    #ax.autoscale(True)
    return line, im

