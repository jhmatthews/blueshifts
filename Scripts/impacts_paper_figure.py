import matplotlib.pyplot as p
from astropy import units as u
import numpy as np 
import py_plot_util as util 
import os
from plot_norm import *
from scipy.signal import savgol_filter
import jm_util
from astropy import constants as const
import os
import py_read_output as rd
sys.path.append("/Users/matthewsj/winds/c4_blue/Scripts")
import blueshift_util
import matplotlib.patheffects as pe
jm_util.set_mod_defaults()
jm_util.set_times()
W0 = 1450

data_dir  = "/Users/matthewsj/winds/c4_blue/_Data/restart_without/"


runs = ["run47","run128"]

suffixes= [["","_restart_norot", "_restart_nolineopacity"],["_diskmax_3e16","_restart_nolineopacity_nodisk"]]
suffixes= [["","_restart_norot", "_restart_nolineopacity"],]

labels = [[["Model e)", "No rotation", "No line RT"],],
          [["Model n)", "No rotation", "No line RT"],]]
#fnames = ["run128_restart_nolineopacity_nodisk", "run47_restart_nolineopacity", "run47_restart_norot", "run47_thmin45_rv1e19_vinf1p0_mdotw5p0"]
#fnames = ["run128_restart_nolineopacity_nodisk", "run47_restart_nolineopacity", "run47_restart_norot", "run47_thmin45_rv1e19_vinf1p0_mdotw5p0"]

jm_util.set_cmap_cycler("Spectral_r", 5)
colors = ["C2", "C0", "C4"]

fig, axes = plt.subplots(figsize=(6,4), nrows=1, ncols = 2)

for irun,run in enumerate(runs):
    for i,suffix in enumerate(suffixes):
        #plt.subplot(2,2,i+1)
        iplot = irun + (i * 2)
        ax = axes[irun]

        for i_s,s in enumerate(suffix):
            f = data_dir + run + s
            print (f)
            s1 = rd.read_spectrum(f)

            angle = "A15P0.50"
            LW = 3.5

            print (irun, i)
            plot_label = labels[irun][i][i_s]

            fl = s1["{}".format(angle)]
            w = s1["Lambda"]
            w = w[::-1]
            fl = fl[::-1]
            cc = blueshift_util.fit_continuum(w, fl, window_length=15, polyorder=7)
            fl = savgol_filter(fl/cc, 15, 3)

            if i_s == 0:
                path = [pe.Stroke(linewidth=6, foreground='#7f7f7f'), pe.Normal()]
            else:
                path = None

            velocity = (w - 1550.0) / 1550.0 * const.c.cgs 
            velocity = velocity.to(u.km/u.s)

            ax.plot(velocity/1e3, fl, alpha=1, zorder=5, label=plot_label, lw=3, path_effects = path, c=colors[i_s])
            ax.set_xlim(-8,8)

        ax.set_ylim(0.8, 6)
        # plt.gca().axvline(0, ls="--", alpha=0.5, color="k")
        ax.axvline(0, ls="--", alpha=1, c="k")
        ax.legend(loc="upper right", labelspacing=0.25, borderpad=0.25)
        #ax.grid()
        if irun == 0:
            ax.set_ylabel(r"Normalised $F_\lambda$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Velocity~($10^3$~km~s$^{-1}$)")

figure_dir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', 'Figures'))
plt.tight_layout(pad=0.05)
plt.savefig("{}/impacts.pdf".format(figure_dir), dpi=300)